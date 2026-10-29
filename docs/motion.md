# Faz 4: hareket ve ürün davranışı

> Bu dosya `baykus-motion report` ile `results/**/*.json`'dan üretildi; elle düzenlenmez. Kabul kanıtı yoksa satır "kanıt yok" ile düşer; eksik sayı `—`.

## Kabul

| Kabul | Durum | Kanıt |
|---|---|---|
| Kabul 1 | ✅ | uçtan uca eşlik (klip senaryoları): 8 senaryo; en büyük köşe farkı 0.000022 px (`results/engine/parity_ego.json`) |
| Kabul 2 | ✅ | hareket birim eşliği: CI koşusu `40893688435` `success` (`results/system/ci_f4.json`) |
| Kabul 3 | ✅ | sentetik doğruluk: CI koşusu `40893688435` `success` (`results/system/ci_f4.json`) |
| Kabul 4 | ✅ | kör insan örneklemi (val): karar `pass`; (a) uyum 72/73 (`results/motion/label_audit_val.json`) |
| Kabul 5 | ✅ | gecikme p95 ≤ 100 ms: motor p95 97.9 ms ≤ 100 ms (Docker, HTTP hariç) (`results/bench/e3c_onnx_fp32_engine_laptop-docker_f4.json`) |
| Kabul 6 | ✅ | ekran (elle, demo dizisi): 28/28 ekran maddesi; Wi-Fi kapalı demo kayıtlı; kapalı ağ otomatik kanıtı geçti (`results/system/manual.json`, `results/system/offline.json`) |
| Kabul 7 | ✅ | bölgeler ve olaylar (CI): CI koşusu `40893688435` `success` (`results/system/ci_f4.json`) |
| Kabul 8 | ✅ | gerileme ve API sağlamlığı (CI): CI koşusu `40893688435` `success` (`results/system/ci_f4.json`) |
| Kabul 9 | ✅ | oracle işlev kapısı (val): BA 0.949 ≥ 0,85; R_mov 0.985, R_sta 0.913 ≥ 0,80 (`results/motion/m3ot_dev_oracle.json`) |
| Kabul 10 (a, c) | ✅ | yeniden konumlama doğruluğu (CI): CI koşusu `40893688435` `success` (`results/system/ci_f4.json`) |
| Kabul 10 b | ❌ (ADR: docs/adr/0011-kabul-10b.md) | yeniden konumlanan dönüşüm (seek, long_gap): köşe farkı sınırı 2,0 px; ölçülen: seek 0.34 px, long_gap 5.01 px (`results/engine/parity_ego.json`) |
| Kabul 11 | ✅ | araç özeti ve günlük (CI): CI koşusu `40893688435` `success` (`results/system/ci_f4.json`) |

Kabul 1-11 çıkışı engeller (tasarım §6.6); Kabul 10 b ADR ile kapanabilir. Faz çıkışına **izin veriyor**.

## İddia kuralı

Hareket iddiası ("hareketli / duran ayrımı çalışıyor") ❌.
Tutmayan koşullar: test oracle BA — < 0.85; test motor BA — < 0.70; test motor P_mov 0.679 < 0.80.
README "hareketli / duran ayrımı çalışıyor" cümlesini kullanmaz; sayılar olduğu gibi yazılır.

Ürün iddiası (bölgeler, alarmlar, harita belleği, dışa aktarma) ✅.
Kabul 6, 7, 10 ve 11 tuttu; olayların gerçek dünyadaki doğruluğu iddia edilmez.

## Hareket: val

| Satır | Dizi | n | hareketli | duran | R_mov | R_sta | BA | kapsam | P_mov | P_sta |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 1_08 | 3997 | 2839 | 1158 | 0.988 | 0.927 | 0.958 | 0.986 | — | — |
| oracle | 2_08 | 5816 | 3611 | 2205 | 0.983 | 0.906 | 0.944 | 0.978 | — | — |
| oracle | birleşik | 9813 | 6450 | 3363 | 0.985 | 0.913 | 0.949 | 0.981 | 0.791 | 0.802 |
| motor | 1_08 | 3997 | 2839 | 1158 | 0.825 | 0.709 | 0.767 | 0.809 | — | — |
| motor | 2_08 | 5816 | 3611 | 2205 | 0.843 | 0.865 | 0.854 | 0.877 | — | — |
| motor | birleşik | 9813 | 6450 | 3363 | 0.835 | 0.811 | 0.823 | 0.849 | 0.717 | 0.724 |

## Hareket: test

| Satır | Dizi | n | hareketli | duran | R_mov | R_sta | BA | kapsam | P_mov | P_sta |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 1_03 | 731 | 731 | 0 | 0.997 | — | — | 0.997 | — | — |
| oracle | 2_03 | 594 | 594 | 0 | 1.000 | — | — | 1.000 | — | — |
| oracle | birleşik | 1325 | 1325 | 0 | 0.998 | — | — | 0.998 | 0.699 | — |
| motor | 1_03 | 731 | 731 | 0 | 0.962 | — | — | 0.962 | — | — |
| motor | 2_03 | 594 | 594 | 0 | 0.864 | — | — | 0.864 | — | — |
| motor | birleşik | 1325 | 1325 | 0 | 0.918 | — | — | 0.918 | 0.679 | — |

Bu bölümde duran araç yok: R_sta ve BA tanımsız.

## İnsan örneklemi

**val** karar `pass`; (a) uyum 72/73 (0.986) ✅.
**test** karar `pass`; (a) uyum 80/80 (1.000) ✅.

| Bölüm | Satır | İnsanla uyum | Oran |
|---|---|---|---|
| val | oracle | 106/120 | 0.883 |
| val | motor | 89/120 | 0.742 |
| test | oracle | 80/80 | 1.000 |
| test | motor | 72/80 | 0.900 |

## Harita kayması ve yeniden konumlama

| Bölüm | Dizi | ego ok | kayma 15 ortanca / p95 (px) | kayma 60 ortanca / p95 (px) | yeniden konumlama |
|---|---|---|---|---|---|
| val | 1_08 | 0.997 | 11.82 / 21.06 | 52.95 / 95.02 | 0.636 |
| val | 2_08 | 1.000 | 9.33 / 17.42 | 37.14 / 89.15 | 0.545 |
| test | 1_03 | 0.996 | 8.67 / 12.60 | 30.87 / 43.68 | 0.500 |
| test | 2_03 | 1.000 | 8.12 / 11.92 | 28.75 / 36.21 | 0.500 |

## Gecikme

Faz 4 bench: p50 79.35 ms, p95 97.91 ms ≤ 100 ms ✅ (Kabul 5).

| Varyant | p50 (ms) | p95 (ms) |
|---|---|---|
| f4 (tam) | 79.35 | 97.91 |
| ego kapalı | 71.99 | 79.45 |
| IR kaybı | 76.93 | 86.91 |
| atlama | 79.86 | 92.03 |

## Sınırlamalar

- **Az veri:** yalnız iki val ve iki test dizisi; güven aralığı yok, genelleme güvencesi değil.
- **Kare hızı:** 30 kare/sn varsayımı; hız ve bekleme süreleri bu varsayıma bağlı.
- **Harita göreli:** harita koordinatı ürünün kendi kestirimi; mutlak konum iddia edilmez.
- **Olay doğruluğu:** olayların gerçek dünyadaki doğruluğu ölçülmedi.
- **Kabul 10 b (ADR 0011):** uzun boşluktan sonra harita birkaç piksel kayabilir.
