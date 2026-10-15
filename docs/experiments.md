# Deney günlüğü

Her girdi: tarih · deney ID · hipotez · kurulum (config, commit, seed) · sonuç (JSON/MLflow bağlantısı) · yorum.

## 2026-10-11 · E0 · Veri hazırlama ve veri raporu
- **Soru:** Veri ne kadar temiz? İki etiket seti ne kadar farklı? Aydınlatma etiketi güvenilir mi?
- **Kurulum:** `notebooks/colab/01_data.ipynb` (Colab CPU) · `baykus-data report` (yerel, Kaggle `coco` + `meta`) · ham veri Kaggle `brendanalvey/visdrone-dronevehicle`
- **Sonuç:** `docs/data-report.md`, ADR 0004, ADR 0005, Spike B (`docs/spikes/02-dronevehicle-format.md`), Spike C (`docs/spikes/03-colab-env.md`)
- **Bulgular:**
  1. 28.439 çiftten 36'sı okunamıyor (hepsi train) → 28.403 çift.
  2. IR etiket seti RGB'den %9-10 daha fazla kutu içeriyor; ~33 bin IR nesnesinin RGB karşılığı yok. "Ana etiket seti IR" kararı (ADR 0003) ve E5'in önemi buradan.
  3. Aynı nesnenin iki setteki merkez kayması p50 0 px, p90 7,2 px, p99 16,6 px.
  4. Parlaklık + resmi oran eşiği aydınlık geceleri "gündüz" sayıyordu (160 örnekte %77,5). Gündüz eşiği elle etiketli örnekle 114,14'e çekildi (%95,0). Oturum klasörü (`<folder>`) etiket kaynağı olarak işe yaramadı (görüntülerin %23'ü tek tip klasörde).
  5. Colab'da veri hazırlığı ~1 saat → işlenmiş veri private Kaggle veri seti (`ouzhantarhan/baykus-dv-rgbt`, 9,84 GB); 4 kanallı TIFF (27 GB) yüklenmedi.
- **Açık kalan:** İşlenmiş görüntülerde kutuların gözle kontrolü (H1 Görev 10) → H2 Görev 9'un ilk Colab oturumunda.
  **Kapandı (2026-10-11):** `02_train` kutu kontrolünde 3 train görüntüsünde RGB kutuları RGB'de, IR kutuları IR'de araçlara oturuyor (kullanıcı kontrolü); kutu sayıları 12/12, 15/15, 14/14.

## 2026-10-14 · E1 · Sadece RGB (yolo26s)
- **Hipotez:** RGB gündüz iyi, gece ve karanlık gecede belirgin düşer.
- **Kurulum:** `configs/e1_rgb.yaml` (50 epoch, seed 42, `config_hash` sha256:749df56f…) · eğitim commit `49da0cb`: 1-38. epoch Colab T4 (kota bitti), 39-50. epoch Colab Pro A100 (`3b) devam`) · değerlendirme commit `d99411b`, val, `conf=0.001`, `max_det=1000`, 1000 bootstrap.
- **Sonuç:** `results/runs/2026-10-11_e1-rgb_s_49da0cb_val_ir.json`, `_val_rgb.json` · `results/train/2026-10-11_e1-rgb_s_49da0cb.json` · tablo `docs/results.md` · çapraz kontrol `docs/eval-crosscheck.md` · MLflow: devam kısmı `2544432b499f48209bacc2645700e0e4` (Colab konsolu; eğitim kaydı yeniden kurulduğu için kayıtta `null`, `4cc1215` ile düzeltildi), T4 kısmının kimliği kayıtta yok.
- **Bulgular (IR etiketleri, ana set):**
  1. mAP50 0,761 [0,736; 0,789], mAP50-95 0,483; precision 0,845, recall 0,869 (conf 0,25).
  2. Aydınlatma: gündüz 0,818 [0,792; 0,849] · gece 0,760 [0,695; 0,815] · karanlık gece 0,708 [0,662; 0,754]. Gündüz ve karanlık gece aralıkları örtüşmüyor; gece aralığı geniş, ikisiyle de örtüşüyor.
  3. RGB etiketleriyle mAP50 0,780 [0,755; 0,808] (gündüz 0,831, gece 0,777, karanlık gece 0,711): IR etiketlerinden yüksek. IR setinde RGB'de karşılığı olmayan nesneler (E0 bulgu 2) RGB modelini IR etiketlerinde cezalandırıyor.
  4. Sınıf: car 0,929, bus 0,936, truck 0,720, freight_car 0,620, van 0,602. Boyut: small 0,492, medium 0,765, large 0,884.
  5. FR-V6: Ultralytics 0,7639 vs pycocotools 0,7613, fark 0,0026 ≤ 0,01.
  6. Ultralytics val mAP50 39-50. epoch arasında 0,756-0,762 (düzlük; `gpu-budget.md`).
- **Yorum (taslak, kullanıcıyla netleşecek):** Düşüş var ama çöküş yok: gündüzden karanlık geceye ~11 puan; RGB karanlık gecede de 0,71 tutuyor. "Belirgin düşüş" hipotezi gündüz-karanlık gece için destekleniyor, gece için güven aralığı karar vermeye yetmiyor. E2 (IR) ve E3 (füzyon) aynı kırılımla karşılaştırılınca anlam kazanacak.

## 2026-10-14 · E2 · Sadece IR (yolo26s)
- **Hipotez:** IR gece ve karanlık gecede RGB'den iyi; gündüz farkı küçük.
- **Kurulum:** `configs/e2_ir.yaml` (50 epoch, seed 42, `config_hash` sha256:d7751ab4…) · eğitim ve değerlendirme commit `02c4098` (`main`), Colab Pro A100, Ultralytics 8.4.177 · val, `conf=0.001`, `max_det=1000`, 1000 bootstrap.
- **Sonuç:** `results/runs/2026-10-14_e2-ir_s_02c4098_val_ir.json`, `_val_rgb.json` · `results/train/2026-10-14_e2-ir_s_02c4098.json` · tablo `docs/results.md` · çapraz kontrol `docs/eval-crosscheck.md` · MLflow `d534385a65104b279e417653b6b907b0`.
- **Bulgular (IR etiketleri, ana set):**
  1. mAP50 0,839 [0,817; 0,859], mAP50-95 0,622; precision 0,894, recall 0,947 (conf 0,25).
  2. Aydınlatma: gündüz 0,824 [0,797; 0,855] · gece 0,803 [0,761; 0,843] · karanlık gece 0,872 [0,839; 0,904]. En yüksek karanlık gecede; üç aralık birbiriyle örtüşüyor.
  3. RGB etiketleriyle mAP50 0,729 [0,704; 0,760] (gündüz 0,802, gece 0,777, karanlık gece 0,673). Fark (RGB − IR) −11,0 puan, karanlık gecede −19,8. E1'de aynı fark +1,8'di (E5).
  4. Sınıf: car 0,979, bus 0,970, truck 0,792, freight_car 0,779, van 0,675. Boyut: small 0,578, medium 0,845, large 0,922.
  5. FR-V6: Ultralytics 0,8424 vs pycocotools 0,8390, fark 0,0034 ≤ 0,01.
  6. Eğitim: Ultralytics val mAP50 en yüksek 0,840 (37-38. epoch); mozaik kapandıktan sonra (41-50) 0,834'ten 0,821'e indi, `best.pt` 0,840 noktasını tutuyor (`gpu-budget.md`).
- **E1 ile (eşleştirilmemiş, aralıklara bakarak):** genel 0,761 [0,736; 0,789] vs 0,839 [0,817; 0,859], aralıklar örtüşmüyor. Karanlık gece 0,708 vs 0,872, gündüz 0,818 vs 0,824. Kesin karşılaştırma eşleştirilmiş bootstrap ile (`baykus-eval compare`, H3 Görev 10'da `COMPARE` adımı).
- **Başlatma:** E2 logunda "Remapped 3/5 cls head rows from pretrained weights by class name" (8.4.177). 8.4.175'te aynı kod varsayılan açık olduğu için E1 de aynı başlatmayı aldı (`gpu-budget.md`, 2026-10-15).
- **Yorum (taslak, kullanıcıyla netleşecek):** Hipotez gündüz (fark ~0,6 puan) ve karanlık gece (~16 puan) için sayılarla uyumlu; gece farkı (~4 puan) aralıkların içinde. RGB etiketleriyle karanlık gecedeki büyük düşüş, RGB etiket setinin karanlık görüntülerde IR'nin gördüğü nesnelerden farklı/eksik kutular içerdiğini düşündürüyor (E0 bulgu 2-3); E5 yorumunda E3 ile birlikte ele alınacak.

## 2026-10-14 · E3 · Erken füzyon, rgb_mean (yolo26s)
- **Hipotez:** Basit 4 kanal birleştirme hem E1'i hem E2'yi geçer; en büyük kazanç karanlık gecede.
- **Kurulum:** `configs/e3_early_fusion.yaml` (50 epoch, seed 42, `first_conv_init: rgb_mean`, `config_hash` sha256:d809f05b…) · eğitim ve değerlendirme commit `67ace37` (`h3`), Colab Pro A100, Ultralytics 8.4.177 · `rgbt4` oturumda üretildi (train + val, 3 dk 9 sn, ADR 0004) · val, `conf=0.001`, `max_det=1000`, 1000 bootstrap.
- **Sonuç:** `results/train/2026-10-14_e3-early-fusion_s_67ace37.json` · MLflow `0b721f9899424cf9b2d3303e71e8c56a` · değerlendirme `results/runs/2026-10-14_e3-early-fusion_s_67ace37_val_{ir,rgb}.json` (Drive'dan gelecek).
- **Bulgular (Colab konsolu, 2026-10-15; JSON gelince güven aralıkları ve kırılımlar eklenecek):**
  1. `baykus-eval` mAP50: IR etiketleri **0,8565**, RGB etiketleri **0,7583**; `dropped=0` (ikisi de).
  2. Ultralytics val (max_det 1000, `best.pt`): mAP50 0,860, mAP50-95 0,646; sınıf car 0,986, bus 0,972, truck 0,837, freight_car 0,804, van 0,702.
  3. Ultralytics'in kendi COCO değerlendirmesi (faster-coco-eval 1.8.0, `maxDets=100`): AP50 0,855, AP50-95 0,644; small 0,307, medium 0,647, large 0,822.
  4. Eğitim: Ultralytics val mAP50 en yüksek 0,861 (31. epoch); mozaik kapandıktan sonra 0,850'den 0,835'e (`gpu-budget.md`).
- **E1/E2 ile (ham sayılar, IR etiketleri):** E1 0,761 · E2 0,839 · E3 0,8565. RGB etiketleriyle E1 0,780 · E2 0,729 · E3 0,758. Eşleştirilmiş farklar `results/compare/` gelince.
- **Yorum:** sonuçlar tamamlanınca, kullanıcıyla.
