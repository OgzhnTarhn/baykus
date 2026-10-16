# Sonuçlar (val, ir etiketleri)

> Bu dosya `baykus-report` ile `results/runs/*.json`'dan üretildi; elle düzenlenmez.
> Köşeli parantez: görüntü düzeyinde bootstrap %95 güven aralığı (FR-V8).

| Çalıştırma | Mod | Split | Etiket | mAP50 [%95 GA] | mAP50-95 | Gündüz | Gece | Karanlık gece |
|---|---|---|---|---|---|---|---|---|
| 2026-10-11_e1-rgb_s_49da0cb | rgb | val | ir | 76.1 [73.6, 78.9] | 48.3 | 81.8 | 76.0 | 70.8 |
| 2026-10-14_e2-ir_s_02c4098 | ir | val | ir | 83.9 [81.7, 85.9] | 62.2 | 82.4 | 80.3 | 87.1 |
| 2026-10-14_e3-early-fusion_s_67ace37 | early_fusion | val | ir | 85.7 [83.8, 87.6] | 64.5 | 86.7 | 79.2 | 87.3 |
| 2026-10-15_e3c-modality-dropout_s_d4e1280 | early_fusion | val | ir | 85.1 [83.1, 86.9] | 63.8 | 86.0 | 80.2 | 86.7 |
| 2026-10-16_e4-late-fusion_s_b346199 | late_fusion | val | ir | 84.8 [82.8, 86.7] | 62.9 | 85.3 | 81.1 | 86.5 |
| 2026-10-16_e4b-late-fusion-illum_s_b346199 | late_fusion | val | ir | 85.0 [83.1, 86.9] | 62.9 | 85.7 | 81.2 | 86.5 |

![Aydınlatma kırılımı](figures/results/illumination.png)

## Etiket seti karşılaştırması (E5)

> Aynı tahminler iki etiket setine karşı (FR-V3). Δ = RGB − IR, mAP50 puanı.

| Çalıştırma | Mod | mAP50 IR [%95 GA] | mAP50 RGB [%95 GA] | Δ (RGB − IR) | Δ gündüz | Δ gece | Δ karanlık gece |
|---|---|---|---|---|---|---|---|
| 2026-10-11_e1-rgb_s_49da0cb | rgb | 76.1 [73.6, 78.9] | 78.0 [75.5, 80.8] | +1.8 | +1.3 | +1.6 | +0.3 |
| 2026-10-14_e2-ir_s_02c4098 | ir | 83.9 [81.7, 85.9] | 72.9 [70.4, 76.0] | -11.0 | -2.3 | -2.7 | -19.8 |
| 2026-10-14_e3-early-fusion_s_67ace37 | early_fusion | 85.7 [83.8, 87.6] | 75.8 [73.6, 78.7] | -9.8 | -2.8 | -2.2 | -18.0 |
| 2026-10-15_e3c-modality-dropout_s_d4e1280 | early_fusion | 85.1 [83.1, 86.9] | 75.9 [73.6, 78.6] | -9.2 | -2.0 | -2.2 | -17.8 |
| 2026-10-16_e4-late-fusion_s_b346199 | late_fusion | 84.8 [82.8, 86.7] | 79.9 [77.8, 82.3] | -4.8 | -0.6 | -0.9 | -11.6 |
| 2026-10-16_e4b-late-fusion-illum_s_b346199 | late_fusion | 85.0 [83.1, 86.9] | 80.2 [78.1, 82.6] | -4.8 | -0.1 | -0.8 | -11.6 |

## Eşleştirilmiş farklar

> `baykus-eval compare`: aynı bootstrap örnekleriyle A − B, mAP50 puanı [%95 GA]; `*` aralık 0'ı içermiyor (anlamlı). Tek seed: seed varyansı dahil değil.

| Karşılaştırma (A − B) | Genel | Gündüz | Gece | Karanlık gece |
|---|---|---|---|---|
| e2_ir-vs-e1_rgb_val_ir | +7.8 [+5.4, +9.9] * | +0.6 [-1.6, +2.8] | +4.3 [+0.8, +8.5] * | +16.3 [+12.8, +20.3] * |
| e3_early_fusion-vs-e1_rgb_val_ir | +9.5 [+7.3, +11.7] * | +4.9 [+2.9, +7.0] * | +3.2 [-0.2, +7.9] | +16.5 [+13.2, +20.2] * |
| e3_early_fusion-vs-e2_ir_val_ir | +1.8 [+0.8, +2.8] * | +4.3 [+2.5, +6.3] * | -1.1 [-2.6, +0.8] | +0.1 [-1.3, +1.4] |
| e3c_modality_dropout-vs-e3_early_fusion_val_camloss_ir | +35.1 [+33.1, +37.1] * | +42.2 [+39.4, +45.1] * | +32.7 [+29.0, +35.9] * | +29.6 [+27.4, +31.6] * |
| e3c_modality_dropout-vs-e3_early_fusion_val_drop_ir_ir | +59.1 [+56.4, +61.9] * | +57.3 [+54.0, +60.9] * | +58.0 [+52.3, +63.0] * | +60.5 [+56.0, +64.5] * |
| e3c_modality_dropout-vs-e3_early_fusion_val_drop_rgb_ir | +11.1 [+8.9, +13.5] * | +27.0 [+23.5, +30.6] * | +7.3 [+4.5, +10.0] * | -1.3 [-2.7, +0.5] |
| e3c_modality_dropout-vs-e3_early_fusion_val_ir | -0.6 [-1.6, +0.4] | -0.7 [-2.0, +0.5] | +0.9 [-0.7, +2.1] | -0.6 [-2.1, +1.0] |
| e4_late_fusion-vs-e2_ir_val_ir | +0.9 [+0.2, +1.6] * | +2.9 [+1.8, +4.0] * | +0.8 [-0.0, +1.5] | -0.7 [-1.4, +0.0] |
| e4_late_fusion-vs-e3_early_fusion_val_camloss_ir | +37.7 [+35.7, +39.8] * | +43.4 [+40.5, +46.5] * | +36.4 [+32.6, +39.4] * | +33.1 [+30.6, +35.3] * |
| e4_late_fusion-vs-e3_early_fusion_val_drop_ir_ir | +63.3 [+60.3, +66.3] * | +58.2 [+54.7, +62.0] * | +63.4 [+57.1, +68.6] * | +66.9 [+62.1, +70.8] * |
| e4_late_fusion-vs-e3_early_fusion_val_drop_rgb_ir | +12.1 [+10.0, +14.7] * | +28.6 [+25.0, +32.3] * | +9.4 [+6.1, +12.6] * | -0.8 [-2.3, +1.1] |
| e4_late_fusion-vs-e3_early_fusion_val_ir | -0.9 [-1.8, +0.0] | -1.4 [-2.8, -0.3] * | +1.9 [+0.0, +3.3] * | -0.8 [-2.2, +0.6] |
| e4_late_fusion-vs-e3c_modality_dropout_val_camloss_ir | +2.6 [+1.8, +3.4] * | +1.2 [+0.2, +2.2] * | +3.7 [+2.3, +5.1] * | +3.5 [+2.0, +4.9] * |
| e4_late_fusion-vs-e3c_modality_dropout_val_drop_ir_ir | +4.2 [+2.7, +5.6] * | +0.9 [-0.6, +2.3] | +5.4 [+2.8, +8.0] * | +6.5 [+3.7, +9.2] * |
| e4_late_fusion-vs-e3c_modality_dropout_val_drop_rgb_ir | +1.0 [+0.1, +2.0] * | +1.6 [+0.3, +3.0] * | +2.0 [+0.5, +3.6] * | +0.5 [-1.0, +1.8] |
| e4b_late_fusion_illum-vs-e4_late_fusion_val_ir | +0.3 [+0.1, +0.5] * | +0.3 [-0.2, +0.8] | +0.1 [-0.1, +0.4] | +0.0 [+0.0, +0.0] |

## Veri setleri arası test (E7, M3OT)

> DroneVehicle'da eğitilen modeller M3OT test bölümünde, sınıfsız `vehicle` AP (spec §8.3): rgb RGB etiketiyle, diğer modlar IR etiketiyle. M3OT kutuları çok küçük (< 1024 px²): mutlak değer değil modlar arası sıra yorumlanır.

| Çalıştırma | Mod | Etiket seti | vehicle AP50 [%95 GA] | AP50-95 |
|---|---|---|---|---|
| 2026-10-11_e1-rgb_s_49da0cb | rgb | rgb | 85.2 [83.8, 86.6] | 32.8 |
| 2026-10-14_e2-ir_s_02c4098 | ir | ir | 98.1 [97.3, 98.5] | 57.4 |
| 2026-10-14_e3-early-fusion_s_67ace37 | early_fusion | ir | 97.1 [96.5, 98.0] | 50.1 |
| 2026-10-15_e3c-modality-dropout_s_d4e1280 | early_fusion | ir | 88.9 [87.7, 90.2] | 43.3 |
| 2026-10-16_e4-late-fusion_s_b346199 | late_fusion | ir | 97.1 [96.5, 97.7] | 57.1 |
