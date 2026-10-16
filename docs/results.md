# Sonuçlar (val, ir etiketleri)

> Bu dosya `baykus-report` ile `results/runs/*.json`'dan üretildi; elle düzenlenmez.
> Köşeli parantez: görüntü düzeyinde bootstrap %95 güven aralığı (FR-V8).

| Çalıştırma | Mod | Split | Etiket | mAP50 [%95 GA] | mAP50-95 | Gündüz | Gece | Karanlık gece |
|---|---|---|---|---|---|---|---|---|
| 2026-10-11_e1-rgb_s_49da0cb | rgb | val | ir | 76.1 [73.6, 78.9] | 48.3 | 81.8 | 76.0 | 70.8 |
| 2026-10-14_e2-ir_s_02c4098 | ir | val | ir | 83.9 [81.7, 85.9] | 62.2 | 82.4 | 80.3 | 87.1 |
| 2026-10-14_e3-early-fusion_s_67ace37 | early_fusion | val | ir | 85.7 [83.8, 87.6] | 64.5 | 86.7 | 79.2 | 87.3 |
| 2026-10-15_e3c-modality-dropout_s_d4e1280 | early_fusion | val | ir | 85.1 [83.1, 86.9] | 63.8 | 86.0 | 80.2 | 86.7 |

![Aydınlatma kırılımı](figures/results/illumination.png)

## Etiket seti karşılaştırması (E5)

> Aynı tahminler iki etiket setine karşı (FR-V3). Δ = RGB − IR, mAP50 puanı.

| Çalıştırma | Mod | mAP50 IR [%95 GA] | mAP50 RGB [%95 GA] | Δ (RGB − IR) | Δ gündüz | Δ gece | Δ karanlık gece |
|---|---|---|---|---|---|---|---|
| 2026-10-11_e1-rgb_s_49da0cb | rgb | 76.1 [73.6, 78.9] | 78.0 [75.5, 80.8] | +1.8 | +1.3 | +1.6 | +0.3 |
| 2026-10-14_e2-ir_s_02c4098 | ir | 83.9 [81.7, 85.9] | 72.9 [70.4, 76.0] | -11.0 | -2.3 | -2.7 | -19.8 |
| 2026-10-14_e3-early-fusion_s_67ace37 | early_fusion | 85.7 [83.8, 87.6] | 75.8 [73.6, 78.7] | -9.8 | -2.8 | -2.2 | -18.0 |
| 2026-10-15_e3c-modality-dropout_s_d4e1280 | early_fusion | 85.1 [83.1, 86.9] | 75.9 [73.6, 78.6] | -9.2 | -2.0 | -2.2 | -17.8 |

## Eşleştirilmiş farklar

> `baykus-eval compare`: aynı bootstrap örnekleriyle A − B, mAP50 puanı [%95 GA]; `*` aralık 0'ı içermiyor (anlamlı). Tek seed: seed varyansı dahil değil.

| Karşılaştırma (A − B) | Genel | Gündüz | Gece | Karanlık gece |
|---|---|---|---|---|
| e2_ir-vs-e1_rgb_val_ir | +7.8 [+5.4, +9.9] * | +0.6 [-1.6, +2.8] | +4.3 [+0.8, +8.5] * | +16.3 [+12.8, +20.3] * |
| e3_early_fusion-vs-e1_rgb_val_ir | +9.5 [+7.3, +11.7] * | +4.9 [+2.9, +7.0] * | +3.2 [-0.2, +7.9] | +16.5 [+13.2, +20.2] * |
| e3_early_fusion-vs-e2_ir_val_ir | +1.8 [+0.8, +2.8] * | +4.3 [+2.5, +6.3] * | -1.1 [-2.6, +0.8] | +0.1 [-1.3, +1.4] |
