"""M3OT ölçüm adaptörü (Faz 3 tasarımı §6.3, plan F3 Görev 11).

Tespit dosyası (`baykus-engine detect` ya da C++ `baykus_detect`: `{"seq", "split", "frames":
[{"frame", "dets"}]}`) → sınıfsız COCO tahminleri (bütün tahminler kategori 1, skor ham) →
AP50; ECE Faz 2 protokolüyle (ham skor ≥ 0,25, 15 aralık) ham ve kalibre skorla. Etiketler
`pack_labels` çıktısından (`labels/<split>_ir.json`, IR etiketi). Değerlendirme katmanı
(ham ≥ 0,001) kullanılır; API yanıtından (conf süzgeçli) AP hiçbir zaman hesaplanmaz.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

from baykus.eval.bootstrap import ap50, collect_matches
from baykus.health.calibrate import CONF_MIN, ece, match_labels

TEST_FLAG = "--i-know-this-is-test"
DUP_IOU = 0.7
DUP_FLOOR = 0.1  # takip girdisi (tasarım §4.3; baykus.track.runner.TRACK_FLOOR)


def load_gt(labels_dir: Path, split: str) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(Path(labels_dir) / f"{split}_ir.json"))


def _stem_ids(gt: COCO) -> dict[str, int]:
    return {img["file_name"]: img["id"] for img in gt.loadImgs(gt.getImgIds())}


def to_coco(dets: dict, gt: COCO) -> list[dict]:
    """Kare kimliği COCO `file_name`'inden (`<dizi>_<kare:06d>`); xyxy → xywh; sınıfsız."""
    ids = _stem_ids(gt)
    out = []
    for fr in dets["frames"]:
        stem = f"{dets['seq']}_{fr['frame']:06d}"
        if stem not in ids:
            raise ValueError(f"{stem}: etiket dosyasında yok")
        for d in fr["dets"]:
            x1, y1, x2, y2 = d["box"]
            out.append(
                {
                    "image_id": ids[stem],
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": d["score_raw"],
                    "score_cal": d["score"],
                }
            )
    return out


def _image_ids(dets: dict, gt: COCO) -> list[int]:
    ids = _stem_ids(gt)
    return [ids[f"{dets['seq']}_{fr['frame']:06d}"] for fr in dets["frames"]]


def ap50_sequence(coco_dets: list[dict], gt: COCO, img_ids: list[int] | None = None) -> float:
    return ap50(collect_matches(gt, coco_dets, img_ids))


def _ece(coco_dets: list[dict], gt: COCO, calibrated: bool) -> float | None:
    if not coco_dets:
        return None
    raw, labels = match_labels(gt, coco_dets)
    keep = raw >= CONF_MIN
    if not calibrated:
        return ece(raw[keep], labels[keep])
    # Kalibre skor ham skorun fonksiyonu (izotonik adım): aynı tespitler, eşlenmiş skor
    cal = {d["score"]: d["score_cal"] for d in coco_dets}
    return ece(np.array([cal[s] for s in raw[keep]], np.float64), labels[keep])


def ece_raw(coco_dets: list[dict], gt: COCO) -> float | None:
    return _ece(coco_dets, gt, calibrated=False)


def ece_calibrated(coco_dets: list[dict], gt: COCO) -> float | None:
    return _ece(coco_dets, gt, calibrated=True)


def _iou(a: list[float], b: list[float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def duplicate_rate(dets: dict, iou: float = DUP_IOU, conf: float = DUP_FLOOR) -> float:
    """Aynı karede IoU ≥ iou olan farklı sınıflı tespit çiftleri / tespit (ham ≥ conf; varsayılan
    takip girdisi, tasarım §4.3). Takipçi sınıfları birleştirdiği için aynı aracın iki sınıflı
    kopyası takipte çift iz, sınıfsız değerlendirmede yanlış pozitif olur."""
    pairs = n = 0
    for fr in dets["frames"]:
        ds = [d for d in fr["dets"] if d["score_raw"] >= conf]
        n += len(ds)
        for i, a in enumerate(ds):
            for b in ds[i + 1 :]:
                if a["cls_id"] != b["cls_id"] and _iou(a["box"], b["box"]) >= iou:
                    pairs += 1
    return pairs / n if n else 0.0


def _summary(coco: list[dict], gt: COCO, img_ids: list[int], files: list[dict]) -> dict:
    return {
        "ap50": ap50_sequence(coco, gt, img_ids),
        "ece_raw": ece_raw(coco, gt),
        "ece_calibrated": ece_calibrated(coco, gt),
        "duplicate_rate": duplicate_rate({"frames": [fr for f in files for fr in f["frames"]]}),
        "n_images": len(img_ids),
        "n_dets": len(coco),
    }


def measure(
    dets_files: list[dict], labels_dir: Path, *, split: str, allow_test: bool = False
) -> dict:
    """Dizi başına ve birleşik ölçüm; etiketler yalnız bu dizilerin kareleriyle sınırlanır."""
    if split == "test" and not allow_test:
        raise ValueError(f"test bölümü yalnız son ölçümde {TEST_FLAG} ile ölçülür")
    other = sorted({d["split"] for d in dets_files} - {split})
    if other:
        raise ValueError(f"tespit dosyalarının split'i {other}, beklenen {split!r}")
    gt = load_gt(labels_dir, split)
    per_seq, all_coco, all_ids = {}, [], []
    for d in dets_files:
        coco, ids = to_coco(d, gt), _image_ids(d, gt)
        per_seq[d["seq"]] = _summary(coco, gt, ids, [d])
        all_coco += coco
        all_ids += ids
    rec = _summary(all_coco, gt, all_ids, dets_files)
    return {
        "split": split,
        "label_set": "ir",
        "class_agnostic": True,
        "sequences": [d["seq"] for d in dets_files],
        **rec,
        "per_seq": per_seq,
    }
