# Ultralytics val vs pycocotools (FR-V6)

> Taslak (2026-10-14): Colab çıktısından alınan sayılar. Run JSON'ları (`results/runs/*_val_ir.json` → `crosscheck`) indirilince tablo kesin değerlerle güncellenir.

| Çalıştırma | Etiket | Ultralytics mAP50 | pycocotools mAP50 | fark | Durum |
|---|---|---|---|---|---|
| 2026-10-11_e1-rgb_s_49da0cb | ir | 0,764 (konsol, 3 basamak) | 0,7613 (konsol) | ≈ 0,003 | ≤ 0,01 ✓ |

Kaynak: `02_train` 4) değerlendirme hücresi, 2026-10-14, A100, Ultralytics 8.4.177, `conf=0.001`, `max_det=1000`, IR etiketleri etkin.

## Ek gözlem: maxDets = 100 skoru düşürüyor

Aynı çalıştırmada Ultralytics `save_json=True` ile kendi COCO değerlendirmesini de yaptı (faster-coco-eval 1.8.0, `maxDets=100`):
AP50 = 0,758, AP50-95 = 0,481. Bizim pycocotools değerlendirmemiz `maxDets=1000` ile 0,7613. Fark kalabalık sahnelerde
görüntü başına 100 tespit sınırından geliyor; CLAUDE.md'deki "maxDets en az 1000" kuralının ölçülmüş gerekçesi.
