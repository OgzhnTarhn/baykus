"""M3OT öğretmen etiketi yardımcıları (Faz 4, Görev 15B).

docs/design/faz4-kapsama.md §3 B "Veri" maddesinin saf (yan etkisiz) parçaları:

- Her M3OT train izinden (dizi + iz kimliği) eşit aralıklı en çok 10 kare.
- Kutunun merkezinde 160 × 128 pencere (görüntü sınırına kaydırılarak); RGB ve IR aynı
  koordinatla kırpılır.
- O karenin oyu: kutuyla (pencere koordinatında) IoU ≥ 0,3 olanların en yüksek IoU'lusu,
  ham skoru ≥ 0,5 ise sınıfı; değilse oy yok.
- İzin sınıfı: oyların çoğunluğu (eşitlikte skor toplamı büyük olan, o da eşitse küçük
  ``cls_id``); hiç oy yoksa ``car`` (0).

Bu modül modele / ONNX'e / pipeline'a dokunmaz; yalnız saf fonksiyonlar içerir.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from baykus.data import m3ot
from baykus.data.multichannel import read_pair

WINDOW = (160, 128)
FRAMES_PER_TRACK = 10
IOU_MIN = 0.3
SCORE_MIN = 0.5
FLOOR = 0.1

CAR_CLS_ID = 0


def _iou(a: list[float], b: list[float]) -> float:
    """İki xyxy kutunun IoU'su (birleşim sıfırsa 0,0)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = ix1 - ix0
    ih = iy1 - iy0
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _clamp_axis(lo: float, size: int, win: int) -> tuple[int, int]:
    """[lo, lo + win] aralığını [0, size] içine kaydırıp tam sayıya çevirir."""
    if size <= win:
        return 0, int(size)
    lo = int(round(lo))
    if lo < 0:
        lo = 0
    if lo + win > size:
        lo = size - win
    return lo, lo + win


def window(box: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """xyxy kutunun merkezinde 160 × 128 pencere; görüntüye sığması için kaydırılır.

    Görüntü pencereden küçükse o eksende ``0..boyut`` döner.
    """
    win_w, win_h = WINDOW
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    x0, x1 = _clamp_axis(cx - win_w / 2.0, int(img_w), win_w)
    y0, y1 = _clamp_axis(cy - win_h / 2.0, int(img_h), win_h)
    return x0, y0, x1, y1


def frames_for_track(frames: list[int], k: int = FRAMES_PER_TRACK) -> list[int]:
    """İz karelerinden sıralı, eşit aralıklı, ilk ve son dahil en çok ``k`` kare.

    ``k``'dan az kare varsa hepsi (tekrarsız) döner.
    """
    fs = sorted(set(int(f) for f in frames))
    if k <= 0:
        return []
    if len(fs) <= k:
        return fs
    if k == 1:
        return [fs[0]]
    n = len(fs)
    idx = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
    out: list[int] = []
    for i in idx:
        if not out or out[-1] != fs[i]:
            out.append(fs[i])
    return out


def frame_vote(dets: list[dict], box: list[float]) -> tuple[int, float] | None:
    """Bir karenin oyu: en yüksek IoU'lu eşleşme, ham skoru yeterliyse (cls_id, score_raw).

    ``dets`` kayıtları ``{cls_id, score_raw, box}``; ``box`` pencere koordinatında.
    IoU ≥ ``IOU_MIN`` olan yoksa ya da en iyi eşleşmenin ``score_raw`` < ``SCORE_MIN``
    ise ``None``.
    """
    best: dict | None = None
    best_iou = 0.0
    for det in dets:
        iou = _iou(box, det["box"])
        if iou >= IOU_MIN and (best is None or iou > best_iou):
            best = det
            best_iou = iou
    if best is None:
        return None
    score = float(best["score_raw"])
    if score < SCORE_MIN:
        return None
    return int(best["cls_id"]), score


def track_class(votes: list[tuple[int, float]]) -> int:
    """Oyların çoğunluğu; eşitlikte skor toplamı büyük, o da eşitse küçük ``cls_id``.

    Hiç oy yoksa ``car`` (0).
    """
    if not votes:
        return CAR_CLS_ID
    counts: dict[int, int] = {}
    sums: dict[int, float] = {}
    for cls_id, score in votes:
        cid = int(cls_id)
        counts[cid] = counts.get(cid, 0) + 1
        sums[cid] = sums.get(cid, 0.0) + float(score)
    best = min(
        counts,
        key=lambda cid: (-counts[cid], -sums[cid], cid),
    )
    return best


def track_rows(root: Path, split: str) -> dict[str, list[dict]]:
    """``root/meta/m3ot_tracks.csv`` → ``{"<seq>:<track_id>": [{frame, stem, box}, ...]}``.

    Yalnız ``split`` eşit, ``modality == 'ir'``, ``flag == 1``, ``cls == 1`` ve
    ``vis > m3ot.MIN_VISIBILITY`` satırları alınır; her izin listesi kareye göre sıralıdır.
    ``box`` xyxy (``[x, y, x + w, y + h]``). ``split == 'test'`` ise ``ValueError``.
    """
    if split == "test":
        raise ValueError("test splitinde iz etiketi yok")
    path = Path(root) / "meta" / "m3ot_tracks.csv"
    out: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            if row["modality"] != "ir":
                continue
            if int(row["flag"]) != 1 or int(row["cls"]) != 1:
                continue
            if float(row["vis"]) <= m3ot.MIN_VISIBILITY:
                continue
            key = f"{row['seq']}:{row['track_id']}"
            x = float(row["x"])
            y = float(row["y"])
            w = float(row["w"])
            h = float(row["h"])
            out.setdefault(key, []).append(
                {
                    "frame": int(row["frame"]),
                    "stem": row["stem"],
                    "box": [x, y, x + w, y + h],
                }
            )
    for rows in out.values():
        rows.sort(key=lambda r: r["frame"])
    return out


def crop_pair(
    root: Path, split: str, stem: str, win: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """RGB + IR çiftini okuyup aynı pencereyle kırpar (``np.ascontiguousarray``)."""
    root = Path(root)
    bgr, ir = read_pair(
        root / "rgb/images" / split / f"{stem}.jpg",
        root / "ir/images" / split / f"{stem}.png",
    )
    x0, y0, x1, y1 = win
    return (
        np.ascontiguousarray(bgr[y0:y1, x0:x1]),
        np.ascontiguousarray(ir[y0:y1, x0:x1]),
    )


def label_tracks(
    root: Path,
    split: str,
    pipe,
    frames_per_track: int = FRAMES_PER_TRACK,
) -> dict[str, dict]:
    """Her iz için seçilen karelerde tespit → oy → iz sınıfı.

    Dönen sözlük ``{anahtar: {"cls_id", "votes", "frames"}}``; ``votes`` her oy için
    ``[cls_id, score_raw]`` listesi, ``frames`` seçilen kare numaraları.
    """
    root = Path(root)
    tracks = track_rows(root, split)
    out: dict[str, dict] = {}
    for key, rows in tracks.items():
        by_frame = {r["frame"]: r for r in rows}
        frames = frames_for_track([r["frame"] for r in rows], frames_per_track)
        votes: list[tuple[int, float]] = []
        img_w = 0
        img_h = 0
        for i, frame in enumerate(frames):
            row = by_frame[frame]
            if i == 0:
                _, ir_full = read_pair(
                    root / "rgb/images" / split / f"{row['stem']}.jpg",
                    root / "ir/images" / split / f"{row['stem']}.png",
                )
                img_h, img_w = ir_full.shape[:2]
            win = window(row["box"], img_w, img_h)
            x0, y0, _, _ = win
            bgr, ir = crop_pair(root, split, row["stem"], win)
            track_box = [
                row["box"][0] - x0,
                row["box"][1] - y0,
                row["box"][2] - x0,
                row["box"][3] - y0,
            ]
            dets = [
                {
                    "cls_id": int(d.cls_id),
                    "score_raw": float(d.score_raw),
                    "box": [
                        float(d.box[0]) - x0,
                        float(d.box[1]) - y0,
                        float(d.box[2]) - x0,
                        float(d.box[3]) - y0,
                    ],
                }
                for d in pipe.detect(bgr, ir, FLOOR)["dets_all"]
            ]
            vote = frame_vote(dets, track_box)
            if vote is not None:
                votes.append(vote)
        out[key] = {
            "cls_id": track_class(votes),
            "votes": [[int(c), float(s)] for c, s in votes],
            "frames": frames,
        }
    return out


def audit_crops(
    root: Path, split: str, keys: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Her izin ortanca karesi için ``(ir, bgr)`` kırpılmış çifti."""
    root = Path(root)
    tracks = track_rows(root, split)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in keys:
        rows = tracks.get(key)
        if not rows:
            continue
        row = rows[len(rows) // 2]
        _, ir_full = read_pair(
            root / "rgb/images" / split / f"{row['stem']}.jpg",
            root / "ir/images" / split / f"{row['stem']}.png",
        )
        img_h, img_w = ir_full.shape[:2]
        win = window(row["box"], img_w, img_h)
        bgr, ir = crop_pair(root, split, row["stem"], win)
        out[key] = (ir, bgr)
    return out
