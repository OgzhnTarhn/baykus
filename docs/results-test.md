# Sonuçlar (test, ir etiketleri)

> Bu dosya `baykus-report` ile `results/runs/*.json`'dan üretildi; elle düzenlenmez.
> Köşeli parantez: görüntü düzeyinde bootstrap %95 güven aralığı (FR-V8).

| Çalıştırma | Mod | Split | Etiket | mAP50 [%95 GA] | mAP50-95 | Gündüz | Gece | Karanlık gece |
|---|---|---|---|---|---|---|---|---|
| 2026-10-11_e1-rgb_s_49da0cb | rgb | test | ir | 73.3 [72.1, 74.5] | 47.1 | 80.0 | 74.6 | 66.9 |
| 2026-10-14_e2-ir_s_02c4098 | ir | test | ir | 81.0 [79.9, 82.1] | 60.8 | 79.8 | 78.5 | 82.5 |
| 2026-10-14_e3-early-fusion_s_67ace37 | early_fusion | test | ir | 82.3 [81.2, 83.4] | 62.4 | 83.2 | 79.0 | 82.9 |
| 2026-10-15_e3c-modality-dropout_s_d4e1280 | early_fusion | test | ir | 82.5 [81.4, 83.5] | 62.6 | 82.8 | 79.6 | 83.1 |
| 2026-10-16_e4-late-fusion_s_b346199 | late_fusion | test | ir | 82.0 [81.0, 83.0] | 61.6 | 82.8 | 80.1 | 82.0 |
| 2026-10-16_e4b-late-fusion-illum_s_b346199 | late_fusion | test | ir | 82.2 [81.2, 83.2] | 61.8 | 83.3 | 80.1 | 82.0 |

![Aydınlatma kırılımı](figures/results/illumination-test.png)

## Etiket seti karşılaştırması (E5)

> Aynı tahminler iki etiket setine karşı (FR-V3). Δ = RGB − IR, mAP50 puanı.

| Çalıştırma | Mod | mAP50 IR [%95 GA] | mAP50 RGB [%95 GA] | Δ (RGB − IR) | Δ gündüz | Δ gece | Δ karanlık gece |
|---|---|---|---|---|---|---|---|
| 2026-10-11_e1-rgb_s_49da0cb | rgb | 73.3 [72.1, 74.5] | 76.3 [75.1, 77.4] | +3.0 | +2.3 | +0.7 | +2.7 |
| 2026-10-14_e2-ir_s_02c4098 | ir | 81.0 [79.9, 82.1] | 69.9 [68.6, 71.3] | -11.1 | -2.8 | -2.6 | -18.1 |
| 2026-10-14_e3-early-fusion_s_67ace37 | early_fusion | 82.3 [81.2, 83.4] | 72.2 [70.9, 73.6] | -10.1 | -1.9 | -1.9 | -18.3 |
| 2026-10-15_e3c-modality-dropout_s_d4e1280 | early_fusion | 82.5 [81.4, 83.5] | 72.1 [70.9, 73.5] | -10.4 | -1.9 | -2.0 | -18.7 |
| 2026-10-16_e4-late-fusion_s_b346199 | late_fusion | 82.0 [81.0, 83.0] | 77.8 [76.7, 78.9] | -4.2 | -0.6 | -1.0 | -8.5 |
| 2026-10-16_e4b-late-fusion-illum_s_b346199 | late_fusion | 82.2 [81.2, 83.2] | 78.1 [76.9, 79.1] | -4.2 | +0.1 | -0.9 | -8.5 |

## Veri setleri arası test (E7, M3OT)

> DroneVehicle'da eğitilen modeller M3OT test bölümünde, sınıfsız `vehicle` AP (spec §8.3): rgb RGB etiketiyle, diğer modlar IR etiketiyle. M3OT kutuları çok küçük (< 1024 px²): mutlak değer değil modlar arası sıra yorumlanır.

| Çalıştırma | Mod | Etiket seti | vehicle AP50 [%95 GA] | AP50-95 |
|---|---|---|---|---|
| 2026-10-11_e1-rgb_s_49da0cb | rgb | rgb | 85.2 [83.8, 86.6] | 32.8 |
| 2026-10-14_e2-ir_s_02c4098 | ir | ir | 98.1 [97.3, 98.5] | 57.4 |
| 2026-10-14_e3-early-fusion_s_67ace37 | early_fusion | ir | 97.1 [96.5, 98.0] | 50.1 |
| 2026-10-15_e3c-modality-dropout_s_d4e1280 | early_fusion | ir | 88.9 [87.7, 90.2] | 43.3 |
| 2026-10-16_e4-late-fusion_s_b346199 | late_fusion | ir | 97.1 [96.5, 97.7] | 57.1 |
