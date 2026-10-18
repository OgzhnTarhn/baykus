"""ByteTrack referans koşucusu (`trackers==2.6.1`) ve MOT değerlendirmesi (tasarım §4.3).

Girdi: tespit dosyasının (`baykus-engine detect`) kareleri; takip katmanı ham skor ≥ 0,1, dosyadaki
sırayla. Eşikler ham skora uygulanır. Her dizi yeni bir takipçiyle başlar (`reset()`; kimlikler
0'dan). Çıktı girdi indeksiyle eşlenir: 2.6.1 sonucu `detections[idx]` ile kurduğu için indeks
`sv.Detections.data` içinde taşınır. C++ eşliği için istenen karelerin iç durumu dökülebilir.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
from trackers import ByteTrackTracker
from trackers.core.bytetrack.tracklet import ByteTrackTracklet
from trackers.eval.evaluate import evaluate_mot_sequences

TRACK_FLOOR = 0.1  # takip katmanı (tasarım §4.1)
CFG_KEYS = (
    "lost_track_buffer",
    "frame_rate",
    "track_activation_threshold",
    "minimum_consecutive_frames",
    "minimum_iou_threshold",
    "high_conf_det_threshold",
)
GT_VIS = 0.25  # M3OT süzgeci (docs/data/m3ot.md): flag 1, sınıf 1, görünürlük > 0,25
METRICS = ["CLEAR", "HOTA", "Identity"]


def load_cfg(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in CFG_KEYS if k not in cfg]
    if missing or set(cfg) - set(CFG_KEYS):
        raise ValueError(f"{path}: takip parametreleri {list(CFG_KEYS)} olmalı; eksik {missing}")
    return cfg


def _detections(dets: list[dict]) -> sv.Detections:
    if not dets:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([d["box"] for d in dets], np.float64),
        confidence=np.array([d["score_raw"] for d in dets], np.float64),
        data={"index": np.array([d["index"] for d in dets], np.int64)},
    )


def _track_rows(tracker: ByteTrackTracker) -> list[list[float]]:
    return [[*map(float, t.get_state_bbox()), int(t.tracker_id)] for t in tracker.tracks]


def run_tracks(
    frames: list[dict], cfg: dict, dump_frames: set[int] = frozenset()
) -> tuple[list[dict], dict[int, dict]]:
    tracker = ByteTrackTracker(**cfg)
    tracker.reset()
    calls: list[tuple[np.ndarray, list]] = []
    associate = tracker._get_associated_indices

    def recording(sim: np.ndarray, thresh: float):
        out = associate(sim, thresh)
        calls.append((np.array(sim), [list(map(int, m)) for m in out[0]]))
        return out

    tracker._get_associated_indices = recording  # yalnız döküm için gözlem; sonuç değişmez
    out, dumps = [], {}
    for fr in frames:
        dets = [d for d in fr["dets"] if d["score_raw"] >= TRACK_FLOOR]
        before = _track_rows(tracker)
        calls.clear()
        res = tracker.update(_detections(dets))
        idx = res.data.get("index", np.array([], np.int64)) if len(res) else []
        tracks = [
            {"index": int(i), "track_id": int(t)} for i, t in zip(idx, res.tracker_id, strict=True)
        ]
        out.append({"frame": fr["frame"], "tracks": tracks})
        if fr["frame"] in dump_frames:
            (iou1, a1), (iou2, a2) = calls if len(calls) == 2 else ((np.zeros((0, 0)), []),) * 2
            dumps[fr["frame"]] = {
                "tracks_before": before,
                "tracks_after": _track_rows(tracker),
                "dets": [{"index": d["index"], "score_raw": d["score_raw"]} for d in dets],
                "iou1": iou1.astype(np.float64).tolist(),
                "assign1": a1,
                "iou2": iou2.astype(np.float64).tolist(),
                "assign2": a2,
            }
    return out, dumps


def to_mot(tracks: list[dict], dets: list[dict]) -> str:
    """MOT satırları (`frame,id,x,y,w,h,conf,-1,-1,-1`); kimliği -1 olanlar yazılmaz."""
    by_frame = {fr["frame"]: {d["index"]: d for d in fr["dets"]} for fr in dets}
    lines = []
    for fr in tracks:
        for t in fr["tracks"]:
            if t["track_id"] < 0:
                continue
            d = by_frame[fr["frame"]][t["index"]]
            x1, y1, x2, y2 = d["box"]
            box = f"{x1},{y1},{x2 - x1},{y2 - y1}"
            lines.append(f"{fr['frame']},{t['track_id']},{box},{d['score_raw']},-1,-1,-1")
    return "\n".join(lines) + ("\n" if lines else "")


def gt_mot(tracks_csv: Path, seq: str) -> str:
    """`pack_labels` iz dosyasından MOT GT (yalnız AP etiketleriyle aynı süzgeçten geçenler)."""
    lines = []
    with Path(tracks_csv).open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["seq"] != seq or int(r["flag"]) != 1 or int(r["cls"]) != 1:
                continue
            if float(r["vis"]) <= GT_VIS:
                continue
            x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
            lines.append(f"{r['frame']},{r['track_id']},{x},{y},{w},{h},1,1,{float(r['vis'])}")
    return "\n".join(lines) + ("\n" if lines else "")


def _metrics(res) -> dict:
    return {"HOTA": res.HOTA.HOTA, "MOTA": res.CLEAR.MOTA, "IDF1": res.Identity.IDF1}


def evaluate_mot(seqs: dict[str, tuple[list[dict], list[dict]]], gt_csv: Path) -> dict:
    """`trackers eval` (TrackEval eşdeğeri): dizi başına ve birleşik HOTA, MOTA, IDF1."""
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir, trk_dir = Path(tmp) / "gt", Path(tmp) / "trk"
        gt_dir.mkdir()
        trk_dir.mkdir()
        for seq, (tracks, dets) in seqs.items():
            (gt_dir / f"{seq}.txt").write_text(gt_mot(gt_csv, seq), encoding="utf-8")
            (trk_dir / f"{seq}.txt").write_text(to_mot(tracks, dets), encoding="utf-8")
        res = evaluate_mot_sequences(gt_dir, trk_dir, metrics=METRICS)
    return {
        "per_seq": {s: _metrics(r) for s, r in res.sequences.items()},
        "combined": _metrics(res.aggregate),
    }


def lsap_cases(n: int = 200, seed: int = 42) -> list[dict]:
    """Atama çözücüsü birim fikstürü: rastgele IoU matrisleri (yarısında eşit değerler),
    SciPy `linear_sum_assignment(maximize=True)` sonucu. Değerler float32 (takipçideki gibi)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        r, c = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        m = rng.uniform(0, 1, (r, c))
        if i % 2:
            m = np.round(m * 4) / 4  # 0, .25, .5, .75, 1: eşit değerli
        m = m.astype(np.float32)
        rows, cols = linear_sum_assignment(m, maximize=True)
        out.append(
            {"iou": m.astype(np.float64).tolist(), "rows": rows.tolist(), "cols": cols.tolist()}
        )
    return out


KALMAN_INIT = [100.0, 50.0, 140.0, 80.0]


def kalman_steps(n: int = 20, seed: int = 42) -> list[dict]:
    """ByteTrack iz Kalman'ı: `KALMAN_INIT` ile açılan izde rastgele predict / update adımları;
    her adımdan sonra durum ve kovaryans (float64)."""
    rng = np.random.default_rng(seed)
    t = ByteTrackTracklet(np.array(KALMAN_INIT, np.float64))
    box = np.array(KALMAN_INIT, np.float64)
    steps = []
    for _ in range(n):
        if rng.uniform() < 0.5:
            t.predict()
            rec = {"op": "predict"}
        else:
            box = box + rng.normal(2.0, 1.5, 4)
            t.update(box.copy())
            rec = {"op": "update", "bbox": box.tolist()}
        kf = t.state_estimator.kf
        rec["state"] = kf.state.ravel().tolist()
        rec["cov"] = kf.state_covariance.tolist()
        steps.append(rec)
    return steps
