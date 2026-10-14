# Sonuçlar (val, ir etiketleri)

> Bu dosya `baykus-report` ile `results/runs/*.json`'dan üretildi; elle düzenlenmez.
> Köşeli parantez: görüntü düzeyinde bootstrap %95 güven aralığı (FR-V8).

| Çalıştırma | Mod | Split | Etiket | mAP50 [%95 GA] | mAP50-95 | Gündüz | Gece | Karanlık gece |
|---|---|---|---|---|---|---|---|---|
| 2026-10-11_e1-rgb_s_49da0cb | rgb | val | ir | 76.1 [73.6, 78.9] | 48.3 | 81.8 | 76.0 | 70.8 |
| 2026-10-14_e2-ir_s_02c4098 | ir | val | ir | 83.9 [81.7, 85.9] | 62.2 | 82.4 | 80.3 | 87.1 |

![Aydınlatma kırılımı](figures/results/illumination.png)

## Etiket seti karşılaştırması (E5)

> Aynı tahminler iki etiket setine karşı (FR-V3). Δ = RGB − IR, mAP50 puanı.

| Çalıştırma | Mod | mAP50 IR [%95 GA] | mAP50 RGB [%95 GA] | Δ (RGB − IR) | Δ gündüz | Δ gece | Δ karanlık gece |
|---|---|---|---|---|---|---|---|
| 2026-10-11_e1-rgb_s_49da0cb | rgb | 76.1 [73.6, 78.9] | 78.0 [75.5, 80.8] | +1.8 | +1.3 | +1.6 | +0.3 |
| 2026-10-14_e2-ir_s_02c4098 | ir | 83.9 [81.7, 85.9] | 72.9 [70.4, 76.0] | -11.0 | -2.3 | -2.7 | -19.8 |
