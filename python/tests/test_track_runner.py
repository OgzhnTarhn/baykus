import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from trackers import ByteTrackTracker

from baykus.track.runner import (
    TRACK_FLOOR,
    evaluate_mot,
    gt_mot,
    kalman_steps,
    load_cfg,
    lsap_cases,
    oracle_frames,
    run_tracks,
    to_mot,
)

CFG = Path(__file__).resolve().parents[2] / "configs" / "engine" / "tracker.json"


def _d(index, box, score=0.9):
    return {"index": index, "cls_id": 0, "score_raw": score, "score": score, "box": list(box)}


def _moving(n=6, step=4.0):
    """İki araç sağa gidiyor; her karede aynı iki tespit (indeksler değişik sırada)."""
    frames = []
    for f in range(1, n + 1):
        a = _d(10 + f, (10 + step * f, 10, 40 + step * f, 30))
        b = _d(20 + f, (200, 100 + step * f, 230, 120 + step * f), 0.8)
        frames.append({"frame": f, "dets": [b, a] if f % 2 else [a, b]})
    return frames


def test_cfg_matches_trackers_defaults():
    cfg = load_cfg(CFG)
    sig = inspect.signature(ByteTrackTracker.__init__).parameters
    assert cfg == {k: sig[k].default for k in cfg}
    assert set(cfg) == {
        "lost_track_buffer",
        "frame_rate",
        "track_activation_threshold",
        "minimum_consecutive_frames",
        "minimum_iou_threshold",
        "high_conf_det_threshold",
    }


def test_runner_maps_outputs_by_index():
    out, _ = run_tracks(_moving(), load_cfg(CFG))
    assert [sorted(t["index"] for t in fr["tracks"]) for fr in out][:2] == [[11, 21], [12, 22]]
    ids = {t["index"]: t["track_id"] for fr in out for t in fr["tracks"]}
    assert ids[11] == ids[21] == -1  # ilk kare: olgunlaşmamış
    assert {ids[12], ids[22]} == {0, 1}  # ikinci güncellemede kimlik (0'dan)
    assert ids[16] == ids[12] and ids[26] == ids[22]


def test_runner_filters_track_floor():
    frames = [
        {"frame": 1, "dets": [_d(1, (0, 0, 10, 10), TRACK_FLOOR - 1e-6), _d(2, (50, 50, 60, 60))]}
    ]
    out, _ = run_tracks(frames, load_cfg(CFG))
    assert [t["index"] for t in out[0]["tracks"]] == [2]


def test_runner_resets_between_sequences():
    cfg = load_cfg(CFG)
    a, _ = run_tracks(_moving(), cfg)
    b, _ = run_tracks(_moving(), cfg)
    assert a == b  # ikinci dizide kimlikler baştan


def test_runner_empty_frames():
    out, _ = run_tracks([{"frame": 1, "dets": []}, {"frame": 2, "dets": []}], load_cfg(CFG))
    assert out == [{"frame": 1, "tracks": []}, {"frame": 2, "tracks": []}]


def test_runner_dump_has_iou_and_assignments():
    _, dumps = run_tracks(_moving(), load_cfg(CFG), dump_frames={3})
    d = dumps[3]
    assert set(d) >= {"tracks_before", "tracks_after", "dets", "iou1", "assign1", "iou2", "assign2"}
    assert len(d["iou1"]) == 2 and len(d["assign1"]) == 2  # 2 iz × 2 yüksek skorlu tespit
    for t, c in d["assign1"]:
        assert d["iou1"][t][c] > 0.1
    assert all(len(t) == 5 for t in d["tracks_after"])


def test_to_mot_skips_unconfirmed():
    frames = _moving(3)
    out, _ = run_tracks(frames, load_cfg(CFG))
    lines = to_mot(out, frames).strip().splitlines()
    assert len(lines) == 4  # kare 2 ve 3, iki iz; kare 1 -1
    f, tid, x, y, w, h, conf, *_ = lines[0].split(",")
    assert int(f) == 2 and float(w) == pytest.approx(30.0) and float(conf) > 0


def test_gt_mot_from_tracks_csv(tmp_path: Path):
    csv = tmp_path / "tracks_ir.csv"
    csv.write_text(
        "split,modality,seq,frame,stem,track_id,x,y,w,h,flag,cls,vis\n"
        "val,ir,1_08,1,1_08_000001,5,10,20,30,40,1,1,1.0\n"
        "val,ir,1_08,1,1_08_000001,6,10,20,30,40,1,1,0.1\n"  # görünürlük ≤ 0,25: etiketlerde de yok
        "val,ir,2_08,1,2_08_000001,7,1,2,3,4,1,1,1.0\n",
        encoding="utf-8",
    )
    got = gt_mot(csv, "1_08")
    assert got.strip().splitlines() == ["1,5,10.0,20.0,30.0,40.0,1,1,1.0"]


def test_oracle_frames_gt_boxes_score_one(tmp_path: Path):
    csv = tmp_path / "tracks_ir.csv"
    csv.write_text(
        "split,modality,seq,frame,stem,track_id,x,y,w,h,flag,cls,vis\n"
        "val,ir,1_08,1,1_08_000001,5,10,20,30,40,1,1,1.0\n"
        "val,ir,1_08,1,1_08_000001,6,50,60,10,10,1,1,0.1\n"  # görünürlük ≤ 0,25: GT'de de yok
        "val,ir,1_08,3,1_08_000003,5,12,20,30,40,1,1,1.0\n"
        "val,ir,2_08,1,2_08_000001,7,1,2,3,4,1,1,1.0\n",
        encoding="utf-8",
    )
    frames = oracle_frames(csv, "1_08", [1, 2, 3])
    assert [fr["frame"] for fr in frames] == [1, 2, 3]  # kutusuz kare de takipçiye gider
    assert frames[0]["dets"] == [
        {"index": 0, "cls_id": 0, "score_raw": 1.0, "score": 1.0, "box": [10.0, 20.0, 40.0, 60.0]}
    ]
    assert frames[1]["dets"] == [] and frames[2]["dets"][0]["box"] == [12.0, 20.0, 42.0, 60.0]


def test_oracle_tracks_evaluate_well(tmp_path: Path):
    csv = tmp_path / "tracks_ir.csv"
    rows = ["split,modality,seq,frame,stem,track_id,x,y,w,h,flag,cls,vis"]
    rows += [f"val,ir,1_08,{f},1_08_{f:06d},3,{10 + 2 * f},20,30,40,1,1,1.0" for f in range(1, 9)]
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    dets = oracle_frames(csv, "1_08", list(range(1, 9)))
    tracks, _ = run_tracks(dets, load_cfg(CFG))
    s = evaluate_mot({"1_08": (tracks, dets)}, csv)["per_seq"]["1_08"]
    # ByteTrack ilk karelerde izi onaylamadan yazmaz (minimum_consecutive_frames); gerisi kusursuz
    assert s["IDF1"] > 0.8 and s["MOTA"] > 0.7


def test_evaluate_mot_perfect(tmp_path: Path):
    csv = tmp_path / "tracks_ir.csv"
    rows = ["split,modality,seq,frame,stem,track_id,x,y,w,h,flag,cls,vis"]
    frames = []
    for f in range(1, 6):
        rows.append(f"val,ir,1_08,{f},1_08_{f:06d},3,{10 * f},20,30,40,1,1,1.0")
        frames.append({"frame": f, "tracks": [{"index": f, "track_id": 0}]})
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    dets = [{"frame": f, "dets": [_d(f, (10 * f, 20, 10 * f + 30, 60))]} for f in range(1, 6)]
    rec = evaluate_mot({"1_08": (frames, dets)}, csv)
    s = rec["per_seq"]["1_08"]
    assert s["HOTA"] == pytest.approx(1.0) and s["MOTA"] == pytest.approx(1.0)
    assert s["IDF1"] == pytest.approx(1.0) and rec["combined"]["MOTA"] == pytest.approx(1.0)


def test_lsap_cases_match_scipy():
    cases = lsap_cases(n=20, seed=42)
    assert len(cases) == 20
    assert any(len(set(np.ravel(c["iou"]))) < np.size(c["iou"]) for c in cases)  # eşit değerli
    for c in cases:
        r, k = linear_sum_assignment(np.array(c["iou"], np.float32), maximize=True)
        assert c["rows"] == r.tolist() and c["cols"] == k.tolist()


def test_kalman_steps_shape():
    steps = kalman_steps(n=20, seed=42)
    assert len(steps) == 20 and {s["op"] for s in steps} <= {"predict", "update"}
    assert all(len(s["state"]) == 8 and len(s["cov"]) == 8 for s in steps)
    json.dumps(steps, allow_nan=False)
