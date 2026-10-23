"""M3OT araç kapsamı ölçümü (Faz 4 tasarımı §1, plan F4 Görev 15A).

Üç koşul yan yana: temiz, RGB kapalı, RGB yerine IR. Her dizi ve koşul için AP50, bulma
oranı (ham skor eşikleri × IoU ≥ 0,5 / 0,3) ve kare başına tespit sayısı; "combined" iki
dizinin toplam sayılarıyla. Etiketler IR (`labels/<split>_ir.json`), sınıfsız; değerlendirme
katmanı (ham ≥ 0,001). Test dizisi reddedilir.
tiles4 — görüntü değişmez; tespit 4 karoya bölünerek yapılır (pipe.detect(..., tiles=4)).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from baykus.engine_ref.compose import decode_pair
from baykus.engine_ref.measure import ap50_sequence, load_gt
from baykus.engine_ref.pipeline import Det, Pipeline

CONDITIONS = ("clean", "rgb_off", "ir_as_rgb", "tiles4")
SCORES = (0.05, 0.1, 0.3, 0.5, 0.7)
IOU_MINS = (0.5, 0.3)


def apply(cond: str, bgr: np.ndarray, ir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Koşulu uygular: temiz aynı, RGB kapalı siyah, RGB yerine IR gri→BGR, tiles4 aynı."""
    if cond not in CONDITIONS:
        raise ValueError(f"bilinmeyen koşul: {cond!r}; seçenekler: {CONDITIONS}")
    if cond in ("clean", "tiles4"):
        return bgr, ir
    if cond == "rgb_off":
        return np.zeros_like(bgr), ir
    return cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR), ir


def _iou(a: list[float], b: list[float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def recall(dets: list[list[float]], gts: list[list[float]], iou_min: float) -> int:
    """IoU'ya göre azalan açgözlü bire bir eşleme; IoU ≥ iou_min olan GT sayısı."""
    pairs = sorted(
        ((_iou(d, g), i, j) for i, d in enumerate(dets) for j, g in enumerate(gts)),
        reverse=True,
    )
    used_d: set[int] = set()
    used_g: set[int] = set()
    n = 0
    for iou, i, j in pairs:
        if iou < iou_min:
            break
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        n += 1
    return n


def _coco(dets: list[Det], image_id: int) -> list[dict]:
    return [
        {
            "image_id": image_id,
            "category_id": 1,
            "bbox": [d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1]],
            "score": d.score_raw,
        }
        for d in dets
    ]


def _gt_boxes(gt, image_id: int) -> list[list[float]]:
    return [
        [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
        for a in gt.loadAnns(gt.getAnnIds(imgIds=[image_id]))
    ]


def _rates(hits: dict, gt_n: int) -> dict:
    return {
        str(s): {f"iou_{m}": hits[s][m] / gt_n if gt_n else 0.0 for m in IOU_MINS} for s in SCORES
    }


def _per_frame(dets_n: dict, frames: int) -> dict:
    return {str(s): dets_n[s] / frames if frames else 0.0 for s in SCORES}


def run_coverage(
    seq_dirs: list[Path],
    labels_dir: Path,
    pipe: Pipeline,
    conds: list[str],
    floor: float = 0.001,
) -> dict:
    """Dizi × koşul AP50 / bulma oranı / kare başına tespit; "combined" toplamla."""
    per_seq: dict[str, dict] = {}
    tot = {
        c: {
            "frames": 0,
            "gt": 0,
            "hits": {s: {m: 0 for m in IOU_MINS} for s in SCORES},
            "dets": {s: 0 for s in SCORES},
            "coco": [],
            "ids": [],
        }
        for c in conds
    }
    splits: set[str] = set()
    for seq_dir in seq_dirs:
        seq_dir = Path(seq_dir)
        index = json.loads((seq_dir / "index.json").read_text(encoding="utf-8"))
        seq, idx, split = index["name"], list(index["frames"]), index["split"]
        if split == "test":
            raise ValueError("test bölümü kapsam ölçümünde kullanılmaz")
        splits.add(split)
        gt = load_gt(labels_dir, split)
        stem_ids = {img["file_name"]: img["id"] for img in gt.loadImgs(gt.getImgIds())}
        ids = [stem_ids[f"{seq}_{f:06d}"] for f in idx]
        gts = [_gt_boxes(gt, img_id) for img_id in ids]
        frames = [
            decode_pair(
                (seq_dir / "rgb" / f"{f:06d}.jpg").read_bytes(),
                (seq_dir / "ir" / f"{f:06d}.png").read_bytes(),
            )
            for f in idx
        ]
        per_seq[seq] = {}
        for cond in conds:
            coco: list[dict] = []
            hits = {s: {m: 0 for m in IOU_MINS} for s in SCORES}
            dets_n = {s: 0 for s in SCORES}
            gt_n = 0
            for (bgr, ir), img_id, g in zip(frames, ids, gts, strict=True):
                b, i = apply(cond, bgr, ir)
                if cond == "tiles4":
                    dets = pipe.detect(b, i, floor, tiles=4)["dets_all"]
                else:
                    dets = pipe.detect(b, i, floor)["dets_all"]
                coco += _coco(dets, img_id)
                gt_n += len(g)
                for s in SCORES:
                    ds = [d.box for d in dets if d.score_raw >= s]
                    dets_n[s] += len(ds)
                    for m in IOU_MINS:
                        hits[s][m] += recall(ds, g, m)
            per_seq[seq][cond] = {
                "ap50": ap50_sequence(coco, gt, ids),
                "recall": _rates(hits, gt_n),
                "dets_per_frame": _per_frame(dets_n, len(idx)),
            }
            t = tot[cond]
            t["frames"] += len(idx)
            t["gt"] += gt_n
            t["coco"] += coco
            t["ids"] += ids
            for s in SCORES:
                t["dets"][s] += dets_n[s]
                for m in IOU_MINS:
                    t["hits"][s][m] += hits[s][m]
    if len(splits) != 1:
        raise ValueError(f"dizilerin split'i {sorted(splits)}, tek split bekleniyor")
    gt_all = load_gt(labels_dir, splits.pop())
    combined = {
        c: {
            "ap50": ap50_sequence(tot[c]["coco"], gt_all, tot[c]["ids"]),
            "recall": _rates(tot[c]["hits"], tot[c]["gt"]),
            "dets_per_frame": _per_frame(tot[c]["dets"], tot[c]["frames"]),
        }
        for c in conds
    }
    return {**per_seq, "combined": combined}
