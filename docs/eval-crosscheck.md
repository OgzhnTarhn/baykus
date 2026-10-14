# Ultralytics val vs pycocotools (FR-V6)

Kabul ölçütü (spec §22 #17): fark ≤ 0,01 ya da açıklaması yazılmış.

| Çalıştırma | Etiket | Ultralytics mAP50 | pycocotools mAP50 | fark | Durum |
|---|---|---|---|---|---|
| 2026-10-11_e1-rgb_s_49da0cb | ir | 0,7639 | 0,7613 | 0,0026 | ≤ 0,01 ✓ |

Kaynak: `results/runs/2026-10-11_e1-rgb_s_49da0cb_val_ir.json` → `crosscheck`. Değerlendirme 2026-10-14, Colab A100,
Ultralytics 8.4.177, commit `d99411b`, `conf=0.001`, `max_det=1000`, IR etiketleri etkin. `dropped_predictions = 0`
(tahminler dosya adıyla eşlendi; dışlanan 36 dosya kimlikleri kaydırmadı).

## Ek gözlem: maxDets = 100 skoru düşürüyor

Aynı çalıştırmada Ultralytics `save_json=True` ile kendi COCO değerlendirmesini de yaptı (faster-coco-eval 1.8.0,
`maxDets=100`, Colab konsol çıktısı): AP50 = 0,758, AP50-95 = 0,481. pycocotools değerlendirmemiz `maxDets=1000`
ile 0,7613 / 0,4827. Fark kalabalık sahnelerde görüntü başına 100 tespit sınırından geliyor; "maxDets en az 1000"
kuralının (CLAUDE.md) ölçülmüş gerekçesi.
