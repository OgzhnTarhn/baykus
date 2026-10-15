"""WBF sarmalayıcısı (FR-V7). (S) görevi: implementasyonu kullanıcı yazar, Claude test yazar.

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
    raise NotImplementedError("(S) görevi: implementasyonu kullanıcı yazar")
