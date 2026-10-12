# GPU bütçesi (Colab)

| Tarih | Deney | Smoke süresi | 1 epoch süresi | Planlanan epoch | Gerçek süre | Not |
|---|---|---|---|---|---|---|
| 2026-10-11 | E1 (rgb, s) | 1 dk 50 sn (eğitim 79 sn) | 7 dk 8 sn (eğitim 5 dk 29 sn + val 21 sn ≈ 5,8 dk/epoch) | 50 | — | 100 epoch tahmini 9,7 saat → karar: epoch 100 → 50, tahmini 4,9 saat (spec §10.2 adım 1); bütün deneyler 50 epoch |

## Notlar

- **2026-10-11 ölçümü** (`02_train`, Colab T4, Python 3.13.15, torch 2.11.0+cu130, Ultralytics 8.4.175, commit `1a92f50`): kurulum 8 sn, Kaggle'dan indirme + `illum-relabel` + `slices` + kutu kontrolü 3 dk 6 sn. Run kayıtları Drive'da: `2026-10-11_e1-rgb_s_1a92f50_smoke`, `2026-10-11_e1-rgb_s_1a92f50_ep1`.
- **Epoch kararı:** 5 tam eğitim (E1, E2, E3, E3b, E3c) × 100 epoch ≈ 49 saat; ücretsiz Colab'ın haftalık süresi kabaca 15-30 saat (spec §7.1). 50 epoch ile eğitim başına ≈ 4,9 saat, haftada en fazla 2 tam eğitim (spec §22). Karşılaştırma adil kalsın diye bütün config'ler aynı tarifle (`test_repo_configs_share_training_recipe`).
- **Darboğaz (hipotez, ölçülmedi):** T4 oturumunda 2 CPU çekirdeği var ve eğitim 3,4 it/s (batch 16). Veri yükleme sınırlıysa `n` modeline geçmek süreyi az kısaltır. Doğrulamak için eğitim sırasında `nvidia-smi` ile GPU kullanımı ölçülmeli.
- **2026-10-12 kesinti:** E1 tam eğitimi (`..._e1-rgb_s_49da0cb`, T4) 38/50 epoch'ta Colab GPU kotası bittiği için durdu. `last.pt` her epoch Drive'a yazılıyordu, ama eğitim kaydı sadece sonda yazıldığı ve devam yolu etiketleri kurmadığı için devam edilemiyordu; `abf259c` ile düzeltildi (kayıt başta yazılır, devamda etiketler kurulur, eksik kayıt yeniden oluşturulur). Devam eden kısmın donanımı kayıtta `hardware_resumed` alanında.
