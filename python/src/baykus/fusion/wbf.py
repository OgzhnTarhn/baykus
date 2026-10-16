"""WBF sarmalayıcısı (FR-V7). (S) görevi; kullanıcının isteğiyle Claude yazdı (2026-10-16).

Bir görüntünün model başına COCO tahminleri → ensemble-boxes weighted_boxes_fusion → COCO
tahminleri. Kütüphane 0-1 aralığında x1, y1, x2, y2 ister; dönüşüm ve doğrulama burada.
"""

from __future__ import annotations

from collections.abc import Sequence


def fuse_image(
    dets_per_model: Sequence[Sequence[dict]],
    weights: Sequence[float],
    *,
    width: int,
    height: int,
    iou_thr: float,
    skip_box_thr: float = 0.0,
) -> list[dict]:
    """Bir görüntünün tahminlerini birleştirir.

    dets_per_model[k]: k. modelin bu görüntüdeki tahminleri; her biri
        {"category_id": int, "bbox": [x, y, w, h] (piksel), "score": float}.
    weights[k]: k. modelin ağırlığı.
    Dönüş: {"category_id", "bbox" [x, y, w, h] piksel, "score"} listesi, skora göre azalan.

    Sözleşme (tests/test_wbf.py):
    - Ağırlık sayısı model sayısından farklıysa ya da bir ağırlık <= 0 ise ValueError
      (ensemble-boxes yanlış sayıda ağırlığı sessizce 1'e çeviriyor).
    - Kutu görüntüye kırpılır; kırpınca alanı 0 kalan kutu ve skoru <= 0 olan kutu atılır
      (kütüphane uyarı basmasın, 0'a bölüp NaN üretmesin).
    - conf_type="avg", allows_overflow=False (kütüphane varsayılanı).
    - Girdi listeleri değiştirilmez; çıktıda sayılar Python int/float.
    - Hiç kutu yoksa [].
    """
    if len(weights) != len(dets_per_model) or any(w <= 0 for w in weights):
        raise ValueError(
            f"ağırlık: model başına bir pozitif değer gerekli; {len(dets_per_model)} model, "
            f"ağırlıklar {list(weights)}"
        )
    boxes, scores, labels = [], [], []
    for dets in dets_per_model:
        b, s, c = [], [], []
        for d in dets:
            x, y, w, h = d["bbox"]
            x1, y1 = max(x, 0.0), max(y, 0.0)  # görüntüye kırp (piksel)
            x2, y2 = min(x + w, width), min(y + h, height)
            if x2 <= x1 or y2 <= y1 or d["score"] <= 0:
                continue  # boş kutu ya da 0 skor: kütüphane uyarır / 0'a böler
            b.append([x1 / width, y1 / height, x2 / width, y2 / height])
            s.append(float(d["score"]))
            c.append(int(d["category_id"]))
        boxes.append(b)
        scores.append(s)
        labels.append(c)
    if not any(boxes):
        return []
    from ensemble_boxes import weighted_boxes_fusion  # numba'yı ilk kullanımda yükle

    fb, fs, fl = weighted_boxes_fusion(
        boxes,
        scores,
        labels,
        weights=[float(w) for w in weights],
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        conf_type="avg",
    )
    out = []
    for (x1, y1, x2, y2), score, label in zip(fb, fs, fl, strict=True):
        x1, x2 = float(x1) * width, float(x2) * width
        y1, y2 = float(y1) * height, float(y2) * height
        out.append({"category_id": int(label), "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score)})  # fmt: skip
    return out  # kütüphane skora göre azalan sıralı döner
