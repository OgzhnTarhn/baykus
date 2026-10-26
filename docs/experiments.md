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
- **Yorum (2026-10-16):** Düşüş var ama çöküş yok: gündüzden karanlık geceye ~11 puan; RGB karanlık gecede de 0,71 tutuyor. "Belirgin düşüş" hipotezi gündüz-karanlık gece için destekleniyor, gece için E1'in kendi aralıkları karar vermeye yetmiyor. E2 ile eşleştirilmiş karşılaştırma (E2 − E1) bedeli netleştirdi: RGB modeli gece 4,3, karanlık gecede 16,3 puan geride (ikisi de anlamlı), gündüz fark yok (+0,6, anlamsız).

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
- **E1 ile (eşleştirilmemiş, aralıklara bakarak):** genel 0,761 [0,736; 0,789] vs 0,839 [0,817; 0,859], aralıklar örtüşmüyor. Karanlık gece 0,708 vs 0,872, gündüz 0,818 vs 0,824. Eşleştirilmiş bootstrap (`results/compare/e2_ir-vs-e1_rgb_val_ir.json`, E2 − E1): genel +7,8 [+5,4; +9,9] *, gündüz +0,6 [−1,6; +2,8], gece +4,3 [+0,8; +8,5] *, karanlık gece +16,3 [+12,8; +20,3] * (`*` anlamlı).
- **Başlatma:** E2 logunda "Remapped 3/5 cls head rows from pretrained weights by class name" (8.4.177). 8.4.175'te aynı kod varsayılan açık olduğu için E1 de aynı başlatmayı aldı (`gpu-budget.md`, 2026-10-15).
- **Yorum (2026-10-16):** Hipotez eşleştirilmiş bootstrap ile tamamen destekleniyor: E2 − E1 gündüz +0,6 [−1,6; +2,8] (fark yok), gece +4,3 [+0,8; +8,5] ve karanlık gece +16,3 [+12,8; +20,3] (ikisi de anlamlı). E2'nin kendi kırılımında karanlık gece (0,871) gündüzden (0,824) yüksek görünüyor ama aralıklar örtüşüyor; "termal gece daha iyi görür" iddiası bu veriyle kurulamaz. RGB etiketleriyle karanlık gecedeki büyük düşüş E5'te açıklandı.

## 2026-10-14 · E3 · Erken füzyon, rgb_mean (yolo26s)
- **Hipotez:** Basit 4 kanal birleştirme hem E1'i hem E2'yi geçer; en büyük kazanç karanlık gecede.
- **Kurulum:** `configs/e3_early_fusion.yaml` (50 epoch, seed 42, `first_conv_init: rgb_mean`, `config_hash` sha256:d809f05b…) · eğitim ve değerlendirme commit `67ace37` (`h3`), Colab Pro A100, Ultralytics 8.4.177 · `rgbt4` oturumda üretildi (train + val, 3 dk 9 sn, ADR 0004) · val, `conf=0.001`, `max_det=1000`, 1000 bootstrap.
- **Sonuç:** `results/runs/2026-10-14_e3-early-fusion_s_67ace37_val_ir.json`, `_val_rgb.json` · `results/train/2026-10-14_e3-early-fusion_s_67ace37.json` · `results/compare/*.json` · tablo `docs/results.md` · çapraz kontrol `docs/eval-crosscheck.md` · MLflow `0b721f9899424cf9b2d3303e71e8c56a`.
- **Bulgular (IR etiketleri, ana set):**
  1. mAP50 0,857 [0,838; 0,876], mAP50-95 0,645; precision 0,893, recall 0,954 (conf 0,25).
  2. Aydınlatma: gündüz 0,867 [0,846; 0,892] · gece 0,793 [0,750; 0,834] · karanlık gece 0,873 [0,840; 0,902].
  3. Sınıf: car 0,981, bus 0,967, truck 0,833, freight_car 0,801, van 0,700. Boyut: small 0,583, medium 0,859, large 0,942.
  4. RGB etiketleriyle 0,758 [0,736; 0,787] (gündüz 0,840, gece 0,771, karanlık gece 0,693); fark (RGB − IR) −9,8, karanlık gecede −18,0 (E5).
  5. FR-V6: Ultralytics 0,8600 vs pycocotools 0,8565, fark 0,0035 ≤ 0,01.
  6. Eğitim: Ultralytics val mAP50 en yüksek 0,861 (31. epoch); mozaik kapandıktan sonra 0,850'den 0,835'e (`gpu-budget.md`).
- **Eşleştirilmiş farklar (A − B, mAP50 puanı, %95 GA, 1000 bootstrap; `*` anlamlı):**

  | | Genel | Gündüz | Gece | Karanlık gece |
  |---|---|---|---|---|
  | E3 − E1 | +9,5 [+7,3; +11,7] * | +4,9 [+2,9; +7,0] * | +3,2 [−0,2; +7,9] | +16,5 [+13,2; +20,2] * |
  | E3 − E2 | +1,8 [+0,8; +2,8] * | +4,3 [+2,5; +6,3] * | −1,1 [−2,6; +0,8] | +0,1 [−1,3; +1,4] |

- **Yorum (2026-10-16):** Hipotezin ilk yarısı tutuyor: E3 genel skorda hem E1'i (+9,5) hem E2'yi (+1,8) anlamlı geçiyor. İkinci yarısı tutmuyor: karanlık gecedeki kazanç E1'e karşı büyük (+16,5) ama bunu IR kanalı tek başına da sağlıyor (E2 − E1 +16,3; E3 − E2 +0,1, fark yok). Füzyonun gerçek katkısı **gündüzde**: E3 gündüz hem E1'i (+4,9) hem E2'yi (+4,3) anlamlı geçiyor, yani ışık varken iki kamera birbirini tamamlıyor ve tek başına en iyi kameradan daha iyi sonuç veriyor. Gece ve karanlık gecede E3 IR'nin önüne geçmiyor (−1,1 ve +0,1, ikisi de anlamsız). Kısaca: karanlıkta termal taşıyor, gündüzde iki kameranın birleşimi her birinden iyi.

## 2026-10-15 · E5 · Etiket seti farkı (E1-E3)
- **Soru:** Aynı tahminler IR ve RGB etiketlerine karşı ne kadar farklı skor alıyor?
- **Kurulum:** E1-E3 değerlendirmeleri (`baykus-eval run`, iki etiket seti, aynı tahmin dosyası), val, 1000 bootstrap.
- **Sonuç:** `docs/results.md` "Etiket seti karşılaştırması (E5)" bölümü · `results/runs/*_val_{ir,rgb}.json`.
- **Bulgular (mAP50, Δ = RGB − IR, puan):**

  | | IR | RGB | Δ genel | Δ gündüz | Δ gece | Δ karanlık gece |
  |---|---|---|---|---|---|---|
  | E1 (rgb) | 76,1 | 78,0 | +1,8 | +1,3 | +1,6 | +0,3 |
  | E2 (ir) | 83,9 | 72,9 | −11,0 | −2,3 | −2,7 | −19,8 |
  | E3 (early_fusion) | 85,7 | 75,8 | −9,8 | −2,8 | −2,2 | −18,0 |

  1. IR görebilen modellerin (E2, E3) kaybı karanlık gecede toplanıyor (≈ −18/−20 puan); gündüz ve gecede fark 2-3 puan.
  2. Karanlık gecede sıralama etiket setine göre değişiyor: RGB etiketleriyle E1 0,711 · E3 0,693 · E2 0,673 (RGB modeli en iyi görünüyor); IR etiketleriyle E1 0,708 · E2 0,871 · E3 0,873.
- **Yorum (2026-10-16):** IR'yi gören modellerin (E2, E3) RGB etiketlerinde precision'ı ≈ 10 puan düşüyor (0,894 → 0,797; 0,893 → 0,798), E1'inki 3 puan (0,845 → 0,811). Bu, E0 bulgu 2 ile uyumlu: IR setinde RGB karşılığı olmayan ≈ 33 bin nesne var; karanlıkta RGB görüntüde görünmeyen araçlar RGB setinde etiketlenmemiş, IR'yi gören modeller onları bulunca RGB etiketlerine karşı yanlış pozitif sayılıyor. Sonuç: RGB etiketleriyle karanlık gecede üç model ayırt edilemiyor (E1 0,711 [0,662; 0,764], E2 0,673 [0,628; 0,728], E3 0,693 [0,649; 0,744]; aralıklar örtüşüyor), IR etiketleriyle E2/E3 E1'den ≈ 16 puan önde ve aralıklar örtüşmüyor. Etiket seti seçimi karanlık gecedeki gerçek farkı gizleyebildiği için "ana set IR" kararı (ADR 0003) ölçümle destekleniyor. Gündüz ve gecedeki 2-3 puanlık kayıp için olası açıklama iki setin kutu kayması (E0: p90 7,2 px); doğrulanmadı.

## 2026-10-15 · E6 (E1-E3) · Sağlamlık paketi
- **Soru:** Hangi model hangi arızada dayanıyor?
- **Kurulum:** `03_robust.ipynb` → `baykus-robust run` (seed 42, val, IR etiketleri, 1000 bootstrap), commit `125fa2d` (`h3`), Colab Pro A100, 12 çekirdek. Oturum bir kez koptu, aynı komutla biten ölçümler atlanarak tamamlandı; toplam süre kaydedilmedi. Karartma piksel çarpanı, hizasızlık IR'nin yatay kaydırılması (etiketler yerinde).
- **Sonuç:** `results/robust/*.json` (32 dosya, `dropped_predictions = 0`) · `docs/robustness.md` · `docs/figures/results/robustness.png` · görüntü düzeyinde tahminler Drive `baykus/robust/*/predictions_coco.json.gz` (kanca K1).
- **Temiz koşul tutarlılık kontrolü:** bozulma hattından geçen temiz val ana değerlendirmeyle birebir aynı (E1 0,761264 · E2 0,839005 · E3 0,856506; fark 0,0).
- **Bulgular (mAP50, köşeli parantez %95 GA; aralıkların örtüşmesine bakılarak, eşleştirilmiş test yapılmadı):**
  1. **Kamera kaybı:** RGB kaybında E3 71,8 [68,8; 74,3] (temizden −13,9; E2'nin temiz 83,9'unun altında). IR kaybında E3 12,8 [11,6; 14,3]: çöküyor, E1'in temiz 76,1'inin çok altında. Erken füzyon IR kanalına bağımlı ve tek kamerayla tek modaliteli modelin yerini tutmuyor.
  2. **Hizasızlık (yalnız E3):** 5 px 83,4 (−2,3) · 10 px 69,4 (−16,3) · 20 px 25,8 (−59,9) · 40 px 8,4. E0'daki doğal kayma p50 0 px, p90 7,2 px, p99 16,6 px (`docs/data-report.md`): 5 px tipik kayma aralığında ve kayıp küçük; 10 px p90-p99 arasında ve kayıp büyük; 20 px p99'un üstünde (kalibrasyon bozulması) ve model çöküyor. Eşik 5-10 px arasında: veri setindeki nesnelerin ≈ %4,8'i zaten 10 px'ten fazla kaymış.
  3. **Karartma (RGB):** E3 her şiddette E1'in üstünde (×0,1: 73,8 [71,0; 76,2] vs 57,6 [54,6; 60,8]) ama ×0,25'ten itibaren E2'nin temiz skorunun altına iniyor (×0,1'de E2 temiz 83,9 [81,7; 85,9], aralıklar örtüşmüyor).
  4. **IR gürültüsü:** E3 E2'den dayanıklı (σ10: 75,9 vs 61,9; σ20: 39,9 vs 19,5; aralıklar örtüşmüyor); RGB kanalı kısmen taşıyor.
  5. **Bulanıklık (iki kamera birden):** E3 en hassas model: çekirdek 11'de 64,1 [61,7; 66,6], E2 78,7 [76,4; 81,0], E1 72,4 [69,8; 75,1]; çekirdek 3'te üçü de temize yakın.
- **Yorum (2026-10-16):** Erken füzyon temiz veride en iyi model, ama sağlamlıkta zayıf. IR kaybında (12,8) temiz E1'in, RGB kaybında (71,8) temiz E2'nin altına düşüyor: bir kamera giderse yerine o kameranın tek başına modelini koymak daha iyi. RGB ×0,25 karartmadan itibaren de E2'nin temiz skorunun altında. Hizasızlığa çok duyarlı: 5 px'te kayıp küçük, 10 px'te 16 puan (veri setinde nesnelerin ≈ %4,8'i bu kadar kaymış), 20 px'te çöküyor. Güçlü bulanıklıkta (çekirdek 11) üç modelin en düşüğü. Avantajı sadece IR gürültüsünde: RGB kanalı kısmen taşıyor ve E3, E2'den dayanıklı. "Termal kamera kapandığında hangi model dayanıyor?" sorusunun H3 cevabı: E3 dayanmıyor; RGB modeli zaten etkilenmiyor. Bu, H4'teki E3c (modality dropout) ve E4 (geç füzyon) deneylerinin gerekçesi: amaç temiz skoru koruyup kamera kaybında tek modaliteli modele yakın kalmak. Bulanıklıktaki hassasiyetin nedeni bilinmiyor (hipotez: model iki kanalın ince kenar uyumuna dayanıyor); H4'te E3c ile birlikte bakılabilir.

## 2026-10-16 · E3b · Erken füzyon, ul_default — yapılmadı
- **Karar (kullanıcı, H4 planı):** E3b yapılmadı: sadece literatürdeki rgb_mean seçimini kontrol ediyordu, son sistemi değiştirmez; bütçe E3c'ye ayrıldı (spec §22 kesme sırası: ek hedefler → E3b).
- **Not:** E3 `rgb_mean` ile eğitildi. `ul_default` kodda 4 kanallı config'in varsayılanı olarak kalır; `configs/e3b_ul_default.yaml` silindi.

## 2026-10-16 · E3c · Erken füzyon + modality dropout (yolo26s)
- **Hipotez:** Tam modalitede küçük bir bedel, kamera kaybında büyük kazanç (spec §13).
- **Kurulum:** `configs/e3c_modality_dropout.yaml` (E3 + `modality_dropout` p_rgb 0,15 / p_ir 0,15; 50 epoch, seed 42, `first_conv_init: rgb_mean`, `config_hash` sha256:b8c73fd3…) · eğitim ve değerlendirme commit `d4e1280` (`main`), Colab Pro A100, Ultralytics 8.4.177, 2,123 saat · val, `conf=0.001`, `max_det=1000`, 1000 bootstrap.
- **Sonuç:** `results/runs/2026-10-15_e3c-modality-dropout_s_d4e1280_val_ir.json`, `_val_rgb.json` · `results/train/2026-10-15_e3c-modality-dropout_s_d4e1280.json` · `results/robust/2026-10-15_e3c-modality-dropout_s_d4e1280_val_*_ir.json` (16) · MLflow `07b9556171ee4edd84078c4788e5699c`.
- **Bulgular (IR etiketleri):**
  1. mAP50 0,851 [0,831; 0,869], mAP50-95 0,638; precision 0,896, recall 0,954 (conf 0,25). E3: 0,857 [0,838; 0,876].
  2. Aydınlatma: gündüz 0,860 [0,837; 0,886] · gece 0,802 [0,759; 0,840] · karanlık gece 0,867 [0,834; 0,897].
  3. Sınıf: car 0,981, bus 0,969, truck 0,814, freight_car 0,781, van 0,708. Boyut: small 0,606, medium 0,852, large 0,938.
  4. RGB etiketleriyle 0,759 [0,737; 0,786]; fark (RGB − IR) −9,2 (E5).
  5. FR-V6: Ultralytics 0,8541 vs pycocotools 0,8506, fark 0,0035.
  6. Sağlamlık (E6, aynı val, mAP50 [%95 GA]; E3 ile yan yana):

     | Koşul | E3 | E3c |
     |---|---|---|
     | RGB kaybı | 0,718 [0,688; 0,743] | 0,829 [0,808; 0,848] |
     | IR kaybı | 0,128 [0,116; 0,143] | 0,720 [0,695; 0,748] |
     | Karartma ×0,1 | 0,738 [0,710; 0,762] | 0,832 [0,811; 0,851] |
     | IR gürültüsü σ20 | 0,399 [0,373; 0,429] | 0,535 [0,510; 0,567] |
     | Bulanıklık 11 | 0,641 [0,617; 0,666] | 0,652 [0,623; 0,684] |
     | Hizasızlık 10 px | 0,694 [0,661; 0,728] | 0,696 [0,662; 0,726] |

  7. Temiz koşul kontrolü: sağlamlık paketindeki temiz skor ana değerlendirmeyle aynı (0,8506).
- **Yorum (taslak, kullanıcıyla netleşecek):** Hipotez tutuyor görünüyor. Temiz veride E3c E3'ün 0,6 puan altında; güven aralıkları örtüşüyor (eşleştirilmiş fark Görev 16'da `baykus-eval compare` ile). Kamera kaybında fark çok büyük: IR kaybında E3 çöküyor (0,128), E3c 0,720'de kalıyor; RGB kaybında 0,718'e karşı 0,829. Karartma ve IR gürültüsünde de E3c önde; bulanıklık ve hizasızlıkta fark yok (aralıklar örtüşüyor). Yani dropout, modellerin bir kameraya tek başına yaslanmasını engelliyor ama iki kameranın hizasız olmasına karşı bir şey öğretmiyor. E4 ile seçim, sonuçlardan önce yazılan kurala göre (H4 planı, "E3c / geç füzyon seçim kuralı").

## 2026-10-16 · E3c (ek) · Eşleştirilmiş farklar
- **Sonuç:** `results/compare/e3c_modality_dropout-vs-e3_early_fusion_val_*.json` (1000 bootstrap, aynı örnekler).
- **Bulgular (E3c − E3, mAP50 puanı [%95 GA]):** temiz −0,6 [−1,6; +0,4] (anlamlı değil) · RGB kaybı +11,1 [+8,9; +13,5] ·
  IR kaybı +59,1 [+56,4; +61,9] · kamera kaybı puanı K +35,1 [+33,1; +37,1].
- **Yorum (taslak):** Hipotez tutuyor: temiz veride ölçülebilir bir bedel yok, kamera kaybında büyük kazanç var.

## 2026-10-16 · E4 · Geç füzyon, WBF (E1 + E2)
- **Hipotez:** İki ayrı modelin kutularını birleştirmek tek modelden iyi; bir kamera kaybolursa öbür model tek başına çalışmaya devam eder.
- **Kurulum:** `configs/fusion/e4_late_fusion.yaml` (`config_hash` sha256:361de02b…) · E1 `2026-10-11_e1-rgb_s_49da0cb` + E2 `2026-10-14_e2-ir_s_02c4098` val tahminleri · ensemble-boxes WBF, `conf_type=avg`, `skip_box_thr` 0,001 · ızgara 12 hücre (iou_thr 0,5 / 0,6 / 0,7 × ir_weight 0,5 / 1 / 2 / 3, w_rgb 1), ölçüt val mAP50 (IR etiketi), eşitlikte ızgara sırası · seçim 659 sn (A100 oturumu, 12 çekirdek) · commit `b346199`.
- **Sonuç:** `results/fusion/e4_late_fusion_val_selection.json` · `results/runs/2026-10-16_e4-late-fusion_s_b346199_val_{ir,rgb}.json`.
- **Izgara (val mAP50):**

  | iou_thr \ ir_weight | 0,5 | 1 | 2 | 3 |
  |---|---|---|---|---|
  | 0,5 | 0,8224 | 0,8323 | 0,8341 | 0,8334 |
  | 0,6 | 0,8259 | 0,8383 | 0,8426 | 0,8423 |
  | 0,7 | 0,8279 | 0,8405 | 0,8476 | **0,8478** |

- **Bulgular (IR etiketleri):**
  1. Seçilen iou_thr 0,7, ir_weight 3 (2 ile fark 0,0002). mAP50 0,848 [0,828; 0,867], mAP50-95 0,629; precision 0,905, recall 0,945.
  2. Aydınlatma: gündüz 0,853 · gece 0,811 · karanlık gece 0,865. Sınıf: car 0,976, bus 0,967, truck 0,813, freight_car 0,791, van 0,693.
  3. RGB etiketleriyle 0,799; fark (RGB − IR) −4,8 (E3: −9,8) → etiket seti farkına en az duyarlı mod.
  4. Eşleştirilmiş farklar (temiz): E4 − E2 +0,9 [+0,2; +1,6] (anlamlı) · E4 − E3 −0,9 [−1,8; +0,0] (anlamlı değil).
- **Not:** Parametre val'de seçildi ve val'de raporlandı → iyimser yanlı; tarafsız sayı final test ölçümünde.
- **Yorum (taslak):** Geç füzyon IR modelini az ama anlamlı geçiyor, erken füzyonla temiz veride ayırt edilemiyor.

## 2026-10-16 · E4b · Geç füzyon, aydınlatmaya göre ağırlık
- **Kurulum:** `configs/fusion/e4b_late_fusion_illum.yaml` · iou_thr E4'ten (0,7) · her aydınlatma diliminde ayrı ir_weight (0,5 / 1 / 2 / 3), val'de · seçim 282 sn.
- **Sonuç:** `results/fusion/e4b_late_fusion_illum_val_selection.json` · `results/runs/2026-10-16_e4b-late-fusion-illum_s_b346199_val_{ir,rgb}.json`.
- **Bulgular:**
  1. Seçilen ir_weight: gündüz 1, gece 2, karanlık gece 3 (karanlık arttıkça IR'ye daha çok ağırlık: beklenen yön).
  2. mAP50 0,850 [0,831; 0,869]; gündüz 0,857 · gece 0,812 · karanlık gece 0,865.
  3. E4b − E4 +0,3 [+0,1; +0,5] (anlamlı ama çok küçük); dilimlerde anlamlı değil.
- **Yorum (taslak):** Aydınlatmaya göre ağırlık, sabit ağırlığa pratikte bir şey katmıyor. Seçim kuralında aday değildi.

## 2026-10-16 · E6 (geç füzyon) · Sağlamlık paketi
- **Kurulum:** E1 ve E2'nin her koşuldaki tahminleri E4'ün parametreleriyle birleştirildi (`baykus-fusion robust`, GPU yok). Hizasızlıkta E2'nin kaydırılmış IR tahmini kullanıldı.
- **Sonuç:** `results/robust/2026-10-16_e4-late-fusion_s_b346199_val_*_ir.json` (16) · tablo `docs/robustness.md`.
- **Bulgular (mAP50; E3 / E3c / E4):**
  1. RGB kaybı 0,718 / 0,829 / 0,839 · IR kaybı 0,128 / 0,720 / 0,761.
  2. Hizasızlık 10 px 0,694 / 0,696 / 0,744 · 20 px 0,258 / 0,266 / 0,485 · 40 px 0,084 / 0,090 / 0,351.
  3. Bulanıklık 11: 0,641 / 0,652 / 0,809. Karartma ×0,1: 0,738 / 0,832 / 0,835. IR gürültüsü σ20: 0,399 / 0,535 / 0,513.
  4. Kamera kaybında E4'ün değeri kalan tek modelin temiz değerine eşit ya da çok yakın (RGB kaybı 0,8389, E2 temiz 0,8390; IR kaybı 0,7613, E1 temiz 0,7613): kaybolan modelin kutusu gelmiyor, kalan modelin kutuları sıra değiştirmeden ağırlık payıyla küçülüyor.
- **E4b:** yalnız 4 koşul (temiz, karartma 0,5 / 0,25 / 0,1: 0,850 / 0,850 / 0,846 / 0,835); geri kalan 12 koşul yapılmadı (seçim kuralında aday değil, temizde E4'ten farkı 0,3 puan; kullanıcı kararı, 2026-10-16).
- **Yorum (taslak):** Geç füzyon hizasızlık ve bulanıklığa erken füzyondan çok daha dayanıklı: iki model ayrı ayrı çalıştığı için IR'nin kayması RGB'nin kutularını bozmuyor. IR gürültüsünde E3c biraz önde.

## 2026-10-16 · E7 · M3OT veri setleri arası test
- **Kurulum:** M3OT test bölümü (`results/m3ot_stats.json`: 960 RGB-IR çifti, 2 dizi, kutu RGB 2473 / IR 2465, süzülen satır yok), sınıfsız `vehicle` AP, `single_cls` + `agnostic_nms`; rgb modu RGB etiketiyle, öbürleri IR etiketiyle; E4 val'de seçilen parametrelerle (`baykus-fusion apply`), E4b uygulanmadı (aydınlatma etiketi yok) · Colab A100, commit `47c42dc`.
- **Sonuç:** `results/m3ot/*.json` (10) · tablo `docs/results.md` ve `docs/data/m3ot.md`.
- **Bulgular (AP50 [%95 GA]):** E1 0,852 [0,838; 0,866] · E2 0,981 [0,973; 0,985] · E3 0,971 [0,965; 0,980] · E3c 0,889 [0,877; 0,902] · E4 0,971 [0,965; 0,977].
  Sıra: E2 > E3 ≈ E4 > E3c > E1. DroneVehicle val sırası (E3 > E3c > E4 > E2 > E1) ile aynı değil: M3OT'ta IR tek başına en iyi.
- **Etiket hizası:** RGB ve IR etiketleri aynı kare + iz numarasında ortanca IoU 0,35, düşey kayma ortanca 5,5 px (std 10,8); kutular ≈ 12,5 px (`docs/data/m3ot.md`). İki girdiyi birlikte kullanan modeller hizasız girdiyle çalışıyor.
- **Yorum (taslak):** E3c'nin M3OT'ta 8 puan geride kalması ve E4'ün E2'nin altında kalması bu hizasızlıkla uyumlu; ayrıca ölçülmedi. Kareler ardışık video kareleri: güven aralıkları iyimser.

## 2026-10-16 · E10 · Orta füzyon (YOLOv11-RGBT) — yapılmadı
- **Neden:** Ultralytics dışı çerçeve (export ve C++ motoru zorlaşır), Faz 1 bütçesi; spec kesme sırasında ilk madde. Faz 2 tasarımında yeniden değerlendirilir.

## 2026-10-16 · E3c / E4 seçimi (kural sonuçlardan önce yazıldı, H4 planı)
- **Temiz val mAP50 (IR):** E3 0,8565 (en yüksek) · E3c 0,8506 (−0,6 puan) · E4 0,8478 (−0,9 puan) → elenen yok (eşik 2,0 puan).
- **Kamera kaybı puanı K** (RGB kaybı ve IR kaybı ortalaması): E3c 0,7741 (0,8287 / 0,7195) · E4 0,8001 (0,8389 / 0,7613).
  K_E4 − K_E3c = +0,0260 [+0,0182; +0,0338] (`results/compare/e4_late_fusion-vs-e3c_modality_dropout_val_camloss_ir.json`, birleşik bootstrap, 1000 örnek). Aydınlatma dilimlerinin üçünde de anlamlı.
- **Seçim: E4** (aralık tamamen 0'ın üstünde).
- **H5'te kontrol:** E4 cihazda iki model çalıştırıyor; gecikme hedefi (ADR 0007) tutmazsa E3c (tek model) seçilir.

## 2026-10-17 · E9 · Edge: format × hassasiyet × donanım (laptop CPU)
- **Soru:** Sevk adayları (E4, E3c) ONNX'e çevrilip küçültülünce doğruluk ne kadar düşüyor, laptop işlemcisinde ne kadar hızlı?
- **Kurulum:** `configs/edge/e9_edge.yaml` (`config_hash` sha256:6ceeaee4…) · export Ultralytics 8.4.177, `nms=False` (uçtan uca `(1, 300, 6)`), 640×640, batch 1, opset 18 · FP16: CPU dönüşümü, giriş/çıkış FP32 · INT8: ORT `quantize_static`, QDQ, S8S8, kanal başına, MinMax, 500 train görüntüsü (seed 42); "tam" = Conv/Gemm/MatMul (Ultralytics kuralı), "head FP32" = ek olarak son `/model.N/` (tespit başlığı) FP32 · doğruluk `baykus-edge eval` (val 1469, IR etiketleri, `conf 0,001`, bootstrap yok; Colab A100 oturumunun 12 CPU çekirdeği, ORT 1.31.0 CPU EP) · hız `baykus-bench` (§14: 50 ısınma + 500 ölçüm, toplam = ön işleme + çıkarım + son işleme [+ WBF]; AMD Ryzen 7 8845HS, 8/16 çekirdek, prize takılı, ORT 1.30.0 CPU EP, varsayılan thread) · commit `7dc9f78`.
- **Sonuç:** `results/edge/*.json` (20), `results/bench/*_laptop-cpu.json` (8), `results/bench/parity_laptop-cpu_*.json` (4); tablo `docs/results.md` "Edge (E9)".
- **Bulgular (val mAP50 IR; Δ = aynı pipeline'ın ONNX FP32'sine göre puan; laptop p50 / p95 ms):**

  | Pipeline | Varyant | mAP50 | Δ | p50 | p95 | FPS | Boyut (MB) |
  |---|---|---|---|---|---|---|---|
  | E4 | PyTorch | 0,8478 | — | — | — | — | — |
  | E4 | onnx_fp32 | 0,8321 | 0 | 118,1 | 131,6 | 8,5 | 76,3 |
  | E4 | onnx_fp16 | 0,8322 | +0,0 | 121,4 | 133,0 | 8,3 | 38,3 |
  | E4 | onnx_int8 | 0,7661 | −6,6 | 216,1 | 241,5 | 4,6 | 20,6 |
  | E4 | onnx_int8_headfp32 | 0,8296 | −0,2 | 203,9 | 224,4 | 4,9 | 23,1 |
  | E3c | PyTorch | 0,8506 | — | — | — | — | — |
  | E3c | onnx_fp32 | 0,8411 | 0 | 46,4 | 59,9 | 21,0 | 38,2 |
  | E3c | onnx_fp16 | 0,8410 | −0,0 | 49,3 | 62,5 | 19,7 | 19,2 |
  | E3c | onnx_int8 | 0,8083 | −3,3 | 98,7 | 117,8 | 9,9 | 10,3 |
  | E3c | onnx_int8_headfp32 | 0,8337 | −0,7 | 90,1 | 108,1 | 10,9 | 11,6 |

  1. **Tam INT8 doğruluğu bozuyor** (−3,3 ile −7,7 puan; E1 −4,1, E2 −7,7): tespit başlığının INT8'e çevrilmesi kaybın çoğu; başlık FP32 kalınca kayıp −0,2 / −0,7 puan (spec §14'teki risk doğrulandı).
  2. **INT8 laptopta hızlanmadı, ≈ 2 kat yavaşladı** (E3c 46 → 90 ms), işlemci AVX-512 VNNI desteklediği hâlde. Olası neden (ölçülmedi): yalnız konvolüsyonlar nicemlendiği için her katman arasında QuantizeLinear/DequantizeLinear dönüşümü var. **FP16 da işlemcide hızlanmadı** (yalnız dosya yarıya iniyor). NFR-5'teki "INT8 ile daha hızlı" beklentisi bu nicemleme yapısıyla tutmadı.
  3. **E4 iki modeli sırayla çalıştırıyor:** çıkarım RGB ≈ 50 ms + IR ≈ 57 ms; ön işleme 8 ms, WBF ≈ 3 ms. Paralel çalıştırma ölçülmedi.
  4. **Eşlik (20 val görüntüsü, `conf ≥ 0,25`):** FP32 ve FP16'da laptop ile Colab birebir (IoU 1,0, skor farkı 0). INT8 iki varyantta da farklı (1-5 kutu eşleşmiyor, skor farkı 0,09-0,47): INT8 doğruluğu Colab'da ölçüldü ve laptopu tam temsil etmiyor; INT8 sevk edilmediği için laptopta yeniden ölçülmedi.
  5. **PyTorch → ONNX FP32 farkı** −0,9 (E1, E3c) ile −1,6 puan (E4; E2 −1,3). Tahmin (ölçülmedi): uçtan uca çıktının görüntü başına 300 kutu sınırı (PyTorch değerlendirmesi 1000) ve kare 640×640 girdi (PyTorch val dikdörtgen 640×512). Sevk kuralını etkilemiyor (Δ ONNX FP32'ye göre); açık iş.
  6. Tepe bellek (ölçüm süreci): E3c ≈ 345 MB, E4 ≈ 594 MB.

## 2026-10-17 · Sevk kararı (kural ölçümden önce, ADR 0007)
- **Kural:** laptop CPU, toplam p95 ≤ 100 ms; varyant ONNX FP32'den en fazla 1,0 puan düşük; aday sırası E4 → E3c.
- **İz:** E4 → uygun varyantlar onnx_fp32, onnx_fp16, onnx_int8_headfp32; en hızlı onnx_fp32 p95 **131,6 ms → geçmedi**. E3c → uygun varyantlar aynı üçü; en hızlı onnx_fp32 p95 **59,9 ms → geçti**.
- **Karar: E3c · ONNX FP32** (kullanıcı onayı, 2026-10-17). H4 seçim kuralının 5. maddesiyle de aynı ("E4 cihazda gecikme hedefini tutmazsa E3c").
- **Bedeli (H4 ölçümleri, val):** kamera kaybı puanı K E3c 0,774 / E4 0,800; hizasızlık 20 px E3c 0,266 / E4 0,485; bulanıklık 11 E3c 0,652 / E4 0,809; M3OT AP50 E3c 0,889 / E4 0,971. Temiz val ONNX FP32'de E3c 0,841 / E4 0,832.
- **E8 (n boyutu): yapılmadı** — `s` boyutundaki E3c hedefi tuttu (ADR 0007 madde 6 gerekmedi).
- **Sonra:** E4'ün hızlandırılması (iki modeli paralel çalıştırmak, RTX 4060 GPU) ve E3c'nin hizasızlığa dayanıklılığı Faz 2-3'e.

## 2026-10-17 · NPU denemesi (FR-X6, H5 Görev 18) — yapılmadı
- **Neden:** kullanıcı kararı (2026-10-17). Ryzen AI Software kurulumu uv ortamından ayrı ve uzun; sevk kararı laptop CPU'yla verildi (ADR 0007), NPU satırı S6 için zorunlu değil (spec FR-X6: "S6'yı engellemez"). Faz 3'te yeniden değerlendirilebilir.

## 2026-10-17 · Final test (test seti, bir kez)
- **Kurulum:** DroneVehicle test (8980 görüntü, 159.614 kutu), sevk kararından sonra bir kez (`07_final_test`, Colab A100, 12 çekirdek); E1-E3c PyTorch (`baykus-eval run --split test`), E4 / E4b val'de seçilen parametrelerle (`baykus-fusion apply`), sevk edilen E3c · ONNX FP32 (`baykus-edge eval`, ORT CPU); iki etiket seti, 1000 bootstrap. Testten sonra hiçbir seçim değişmedi.
- **Sonuç:** `results/runs/*_test_*.json` (12), `results/edge/*_test_onnx_fp32_*.json` (2); tablo `docs/results-test.md`.
- **Bulgular (test mAP50, IR etiketleri [%95 GA]; gündüz / gece / karanlık gece):**

  | Sistem | mAP50 | Gündüz | Gece | Karanlık gece | RGB etiketiyle |
  |---|---|---|---|---|---|
  | E1 RGB | 0,733 [0,721; 0,745] | 0,800 | 0,746 | 0,669 | 0,763 |
  | E2 IR | 0,810 [0,799; 0,821] | 0,798 | 0,785 | 0,825 | 0,699 |
  | E3 erken füzyon | 0,823 [0,812; 0,834] | 0,832 | 0,790 | 0,829 | 0,722 |
  | E3c (PyTorch) | 0,825 [0,814; 0,835] | 0,828 | 0,796 | 0,831 | 0,721 |
  | E4 geç füzyon | 0,820 [0,810; 0,830] | 0,828 | 0,801 | 0,820 | 0,778 |
  | E4b | 0,823 [0,812; 0,832] | 0,833 | 0,801 | 0,820 | 0,781 |
  | **E3c · ONNX FP32 (sevk)** | **0,814 [0,804; 0,825]** | 0,820 | 0,789 | 0,819 | 0,711 |

  1. Test değerleri val'den 2-3 puan düşük (E3c 0,851 → 0,825); sıralama RGB < IR < füzyon aynı.
  2. Füzyon modelleri (E3, E3c, E4, E4b) testte birbirinden ayırt edilemiyor (aralıklar örtüşüyor; eşleştirilmiş farklar val'de, testte hesaplanmadı). Füzyon E1'den ≈ 9 puan, E2'den ≈ 1 puan önde; karanlık gecede E1 0,669, füzyon 0,82-0,83.
  3. Sevk edilen ONNX FP32 sürümü PyTorch E3c'den 1,1 puan düşük (val'de 1,0): E9 bulgusu 5'teki fark, testte de aynı.
- **Yorum (taslak):** Seçim val'de yapıldı, test bunu bozmadı: E3c hızlı (laptop CPU p95 60 ms) ve testte en iyi füzyon modelleriyle aynı düzeyde.

## 2026-10-17 · E9 (ek) · TensorRT, Colab T4
- **Kurulum:** Ultralytics `format="engine"` (TensorRT 11.4.0.106), FP32 / FP16 (ModelOpt AutoCast; `/model.23/` başlık düğümleri FP32 kaldı) / INT8 (ModelOpt, 500 train kalibrasyon görüntüsü; Conv'un yanında Add, Mul, Resize, MaxPool, TopK da nicemlendi, kalanı FP16) · doğruluk `baykus-edge eval` (TensorRT, Ultralytics AutoBackend) · bench `baykus-bench --hardware colab-t4` (Tesla T4, oturumda 2 CPU çekirdeği; ön işleme CPU'da).
- **Sonuç:** `results/edge/*_val_trt_*.json` (12), `results/bench/*_colab-t4.json` (6); tablo `docs/results.md` "Edge (E9)".
- **Bulgular (val mAP50 IR, Δ ONNX FP32'ye göre puan; T4 p50 / p95 ms):**

  | Pipeline | Varyant | mAP50 | Δ | p50 | p95 | FPS | Motor (MB) |
  |---|---|---|---|---|---|---|---|
  | E3c | trt_fp32 | 0,8411 | −0,0 | 13,1 | 14,1 | 75,8 | 121,5 |
  | E3c | trt_fp16 | 0,8409 | −0,0 | 9,9 | 11,4 | 99,3 | 92,9 |
  | E3c | trt_int8 | 0,8258 | −1,5 | 9,9 | 11,7 | 99,5 | 34,3 |
  | E4 | trt_fp32 | 0,8321 | −0,0 | 32,9 | 44,0 | 28,7 | 243,1 |
  | E4 | trt_fp16 | 0,8323 | +0,0 | 29,0 | 40,9 | 32,3 | 189,0 |
  | E4 | trt_int8 | 0,8322 | +0,0 | 25,9 | 36,6 | 36,9 | 94,3 |

  1. **GPU'da iki sistem de çok hızlı:** E3c FP16 p95 11 ms (≈ 100 FPS), E4 INT8 p95 37 ms (≈ 37 FPS). ADR 0007'nin 100 ms hedefi GPU'da E4 için de rahatça tutuluyor; GPU'lu bir cihazda daha sağlam E4 seçilebilir (sevk kararı CPU için, değişmedi).
  2. **T4'te çıkarım artık darboğaz değil:** E3c FP16 / INT8'de çıkarım ≈ 4,8 ms, ön işleme ≈ 4,8 ms (CPU, 2 çekirdek); E4'te ön işleme 10-11 ms + WBF 3-5 ms. INT8 FP16'ya göre E3c'de hız kazandırmadı.
  3. **TensorRT INT8 doğruluğu ORT INT8'den çok daha iyi:** E3c −1,5 puan (ORT tam INT8 −3,3), E1 −2,1, E2 −1,2. E4 INT8'de birleştirme kaybı kapatıyor (−0,0). TensorRT FP32 doğruluğu ONNX FP32 ile aynı.
  4. Motor dosyaları ONNX'ten büyük (FP32 121 MB / ONNX 38 MB): TensorRT motoru T4'e özgü derlenmiş çekirdekleri de içeriyor.

## 2026-10-17 · Açık iş: PyTorch → ONNX FP32 farkının nedeni (analiz 1)
- **Soru:** ONNX FP32 neden PyTorch değerlendirmesinden ≈ 1 puan düşük (E9 bulgusu 5)? Hipotezler: (a) uçtan uca çıktının 300 kutu sınırı, (b) kare 640×640 dolgulu girdi (PyTorch val dikdörtgen).
- **Kurulum:** laptopta, Drive'dan indirilen val tahminleri (E3c PyTorch + ONNX FP32; E2 PyTorch + TensorRT FP32 — TRT FP32 doğruluğu ONNX FP32 ile aynı, 0,8262) ve `data/dv_rgbt/coco/val_ir.json`; `build_metrics` (bootstrap yok).
- **Bulgular (val mAP50, IR):**

  | Model | Tahmin | mAP50 | small | medium | large | Görüntü başına > 300 kutu |
  |---|---|---|---|---|---|---|
  | E3c | PyTorch | 0,8506 | 0,606 | 0,852 | 0,938 | 4 görüntü |
  | E3c | PyTorch, ilk 300 kutu | 0,8498 | 0,603 | 0,851 | 0,938 | — |
  | E3c | ONNX FP32 (640×640) | 0,8411 | **0,509** | 0,849 | 0,921 | 7 görüntüde 300'e dayanıyor |
  | E2 | PyTorch | 0,8390 | 0,578 | 0,845 | 0,922 | 5 görüntü |
  | E2 | PyTorch, ilk 300 kutu | 0,8365 | 0,564 | 0,844 | 0,922 | — |
  | E2 | TensorRT FP32 (640×640) | 0,8262 | **0,456** | 0,837 | 0,912 | 9 görüntüde 300'e dayanıyor |

  1. **300 kutu sınırı farkın küçük bir kısmı:** E3c 0,1 / 0,95 puan, E2 0,25 / 1,3 puan.
  2. **Kayıp küçük nesnelerde:** small AP E3c −9,7, E2 −12,2 puan; medium / large neredeyse aynı. Export'lu model toplamda daha fazla kutu üretiyor (E3c 72.201 / 62.860).
  3. Kalan hipotez (b): ölçek aynı (640×512 görüntü iki yolda da büyütülmüyor), değişen yalnız dolgu ve ızgara boyutu. Doğrulamak için PyTorch modelini Ultralytics val'de `rect=False` (kare girdi) ile ölçmek gerekiyor (Colab, kısa T4 oturumu).
- **Karar (kullanıcı, 2026-10-17):** Colab doğrulaması yapılmadı; sevk kararını etkilemiyor. Dikdörtgen (640×512) export Faz 3'te C++ motoruyla birlikte denenecek (hem küçük nesne doğruluğu hem ≈ %20 daha az piksel).

## 2026-10-17 · F2 · Sensör sağlığı kapısı (08_health, tek Colab oturumu) — ham sonuçlar
- **Kurulum:** `notebooks/colab/08_health.ipynb`, Colab A100-SXM4-40GB, 12 CPU çekirdeği, Python 3.13.15, Ultralytics 8.4.177, `BRANCH=f2` (commit `4fb259e`). Model E3c `2026-10-15_e3c-modality-dropout_s_d4e1280` (PyTorch, Faz 1 sağlamlık yolu; plan F2 Spec yorumları 1). Faz 1 val taban tahminleri 16 koşulun hepsinde Drive'da (eksik yok). Eşikler train'den 500 görüntü (seed 42) × 16 koşul, Youden J. Yeni değerlendirme kayıtları `n_boot = 0`; aralıklar `baykus-health compare` (1000 eşleştirilmiş bootstrap, seed 42, IR etiketleri).
- **Süreler:** kurulum 6,9 sn · veri 9 dk 30 sn · eşikler 19,4 sn · val kapılı paket 21 dk 44 sn · test taban (3 koşul) 17 dk 48 sn · test kapılı 36 dk 55 sn · karşılaştırma (val + test) 8 dk 7 sn.
- **Eşikler (train; `results/health/thresholds.json`):** rgb_lost std ≤ 0,086 (TPR 1,000 / FPR 0,000) · ir_lost std ≤ 6,564 (1,000 / 0,000) · ir_noisy gürültü ≥ 15,035 (1,000 / 0,000) · misaligned kayma ≥ 16,158 px (0,979 / 0,067) · rgb_blurry (bilgi) ≤ 32,574 (0,968 / 0,116) · ir_blurry (bilgi) ≤ 64,683 (0,955 / 0,089). Train temiz karelerde kapı oranı **%5,00**.
- **Val (mAP50, IR; E3c → E3c + kapı; kapının IR'yi kapattığı kare / 1469):**

  | Koşul | E3c | + kapı | Δ puan | Kapı |
  |---|---|---|---|---|
  | clean | 0,8506 | 0,8197 | −3,09 | 60 |
  | darken-0.5 / 0.25 / 0.1 | 0,8411 / 0,8360 / 0,8317 | 0,8102 / 0,7910 / 0,7669 | −3,09 / −4,50 / −6,48 | 67 / 68 / 76 |
  | noise-5 / 10 | 0,8413 / 0,7704 | 0,8106 / 0,7413 | −3,07 / −2,91 | 63 / 69 |
  | **noise-20** | 0,5355 | 0,7195 | **+18,40** | 1469 |
  | blur-3 / 7 / 11 | 0,8307 / 0,7440 / 0,6523 | 0,8078 / 0,7232 / 0,6328 | −2,29 / −2,08 / −1,95 | 61 / 61 / 65 |
  | misalign-5 / 10 | 0,8337 / 0,6955 | 0,8015 / 0,6613 | −3,23 / −3,42 | 88 / 162 |
  | **misalign-20 / 40** | 0,2656 / 0,0900 | 0,6820 / 0,7112 | **+41,63 / +62,13** | 1396 / 1464 |
  | drop_rgb / drop_ir | 0,8287 / 0,7195 | aynı (piksel değişmedi) | 0 / 0 | rgb_off 1469 / ir_off 1469 |

  Z birleşik (noise-20, misalign-20, misalign-40): **+40,7 puan [+38,3; +43,1]**.
- **Kabul ölçütleri (val, ölçümden önce sabit):** Kabul 1 ❌ temiz fark −3,09 puan (sınır ≥ −0,5) · Kabul 2 ✅ Z aralığı tamamen 0'ın üstünde · Kabul 3 ❌ temiz yanlış alarm %4,08 (sınır ≤ %2). Kabul 4 (kalibrasyon) laptopta ölçülecek.
- **Test kontrolü (tek sefer, 8980 görüntü):** clean 0,8250 → 0,7919 (−3,32; kapı 445 kare, %4,96) · noise-20 0,5401 → 0,6904 (+15,03; 8980) · misalign-20 0,2959 → 0,6604 (+36,46; 8539, %95,1). Z (noise-20, misalign-20): **+25,7 puan [+24,6; +26,9]**. Testteki kapısız temiz değer Faz 1 final testiyle aynı (0,8250): ölçüm yolu tekrarlanabilir.
- **İlk okuma:**
  1. Kapı bozuk IR'de büyük kazanç sağlıyor (gürültüde kapalı IR = IR kaybı 0,7195; hizasızlıkta tespit %95-99,7).
  2. Ama her koşulda karelerin ≈ %4-5'inde yanlış tetikleniyor ve bu kareler pahalı: temiz val'de %4 kare 3 puan kaybettiriyor; karartma arttıkça kayıp büyüyor (−3,1 → −6,5). Dal incelemesinin öngördüğü gibi kaynağın karanlık karelerdeki hizasızlık kestirimi olması muhtemel; sağlık kaydıyla doğrulanacak.
  3. Tasarımın anti-hedefi gerçekleşti: kapı temiz veride sistemi kötüleştiriyor. Önceden kaydedilen kurala göre Faz 2 kapısı **olumsuz sonuç**.
- **Sonuç dosyaları:** Drive `results/health/` (41 KB), `results/gate/` (55 KB), `results/robust/*_test_*` (3); repoya indirilmeyi bekliyor.

## 2026-10-17 · F2 · Kalibrasyon ve kontroller (laptop)
- **Kalibrasyon (Görev 11; `results/health/calibration_e3c_onnx_fp32.json`):** E3c · ONNX FP32, tablo val'in bütün tespitlerinden (`conf ≥ 0,001`) izotonik regresyonla (80 blok), test ECE (15 aralık, ham skor ≥ 0,25, 174.487 tespit) **0,0802 → 0,0071**. Test mAP50 0,8144 → 0,8133: **|Δ| 0,112 puan, Kabul 4 ❌** (sınır ≤ 0,1; önceden sabit). `map50_before` sevk edilen modelin test kaydıyla aynı (0,8144).
- **Kontroller** (`scripts/analysis/f2_checks.py`; dal incelemesinin önerileri, yeniden Colab yok):
  1. **Yanlış alarmların kaynağı:** temiz val'deki 60 alarmın **60'ı "IR hizasız"**; **48'i karanlık gece** karesi (val'in %28'i karanlık gece; karanlık gecede alarm oranı 48 / 415 = %11,6), 4 gece, 8 gündüz. Alarm karelerinde faz korelasyonu yanıtının medyanı **0,051** (alarm olmayanlarda 0,559), ölçülen kayma medyanı 57,6 px: karanlık RGB'de kenar haritası eşleşmiyor, kestirim anlamsız bir tepe buluyor. Dal incelemesinin öngördüğü neden doğrulandı.
  2. **Tam sayı Youden J** (`81e3ccb`) ile `train_features.csv`'den yeniden hesaplanan altı eşik Colab'dakilerle **birebir aynı**: kayan nokta eşitlik hatası bu ölçümü etkilemedi.
  3. **Oturumlar arası sapma görülmedi:** temiz val'de kapının dokunmadığı 1409 karenin **1409'unda** kapılı koşunun tahminleri Faz 1'in **ana değerlendirme** tahminleriyle (`baykus-eval`, `data/gap/<run>_val`) aynı (kutu 2, skor 4 ondalık). `compare`'in eşleştirdiği sağlamlık tabanı (Drive `robust/<run>_val_clean`) laptopta olmadığı için onunla ayrıca karşılaştırılmadı; iki Faz 1 kaydının mAP50'si aynı (0,8506). Eşleştirmenin geçerliliği için güçlü ama dolaylı kanıt.
  4. **Kabul 4'ün kaybı skor eşitliklerinden:** kalibrasyon skorları 80 değere indiriyor (val'de 67.455 farklı ham skordan). **Val'de** (tablo val'den, val'e uygulanmış) mAP50 0,841056 → 0,841138; eşitlik ham skorla bozulunca **0,841056 = kalibrasyon öncesi**: sıralama birebir korunuyor (matematiksel özellik, veriyle seçilmiş bir şey değil). Kabul 4'ün sonucu değişmez (❌); Faz 3 motoru için kural "kalibre skorla sırala, eşitlikte ham skor" önerilir. *Not:* bu analiz ilk önce test setinde yapılmıştı (aynı sonuç: 0,814391 = önce); CLAUDE.md'nin "test yalnız final ölçümde" kuralına aykırı olduğu için val'de yeniden yapıldı; test sayısı hiçbir seçimde kullanılmadı.
- **Faz 2 kabul özeti:** Kabul 1 ❌ (−3,09 puan), Kabul 2 ✅ (+40,7 [+38,3; +43,1]), Kabul 3 ❌ (%4,08), Kabul 4 ❌ (0,112 puan). Rapor `docs/health.md`.

## 2026-10-17 · F2 · Karar: kapı düzeltmesi Faz 3'e
- **Karar (kullanıcı, 2026-10-17):** Faz 2 olumsuz sonuçla kapanır. Hizasızlık yanlış alarmının düzeltmesi bu fazda denenmez: kural val sonuçlarına bakılarak bulundu (val'de ölçmek iyimser olurdu) ve test bir kez kullanıldı (yeniden doğrulanamaz).
- **Faz 3 tasarımına önceden yazılacak girdiler:** (1) faz korelasyonu yanıtı düşükse (ör. eşik train'den) ya da RGB `dark` ise `misaligned` kararı verilmez; ölçüt yine "temiz veride zarar yok" önce, sonra fayda; (2) ardışık karelerle donmuş kare tespiti; (3) kalibre skorla sıralamada eşitlik ham skorla bozulur (Kabul 4'teki kayıp böylece sıfır, `scripts/analysis/f2_checks.py`); (4) dedektör hızı: laptop CPU'da `check_frame` medyan 11,3 ms (20 val çifti, 640×512; tasarım hedefi < 5 ms tutmadı), C++'ta yeniden ölçülür.

## 2026-10-18 · F3 · Tasarım kararları (tasarım incelemesi sonrası)
- **Bağlam:** `docs/design/faz3.md` taslağı eleştirel incelemeden geçti (eşlik ölçütleri, takipçi girdisi, OpenCV sürümü, kapalı ağ kanıtı, `r_min` ölçütü, M3OT sağlık kontrolü). Bulgular tasarımın v2'sine işlendi.
- **Karar (kullanıcı, 2026-10-18; önerilen seçenekler kabul edildi):**
  1. Hizasızlık yanlış alarmı iki katmanlı kuralla düzeltilir: **kare kuralı** (yanıt eşiği `r_min`; train temiz karelerde kapı alarmı ≤ %2 olan en küçük değer; Youden J değil, "önce temiz veride zarar yok") ve videoda **zaman kuralı** (son 5 karenin en az 3'ünde kayma vektörü ±3 px içinde tutarlı). Modelin yeniden eğitimi ve görüntüye bakan sinir ağı yok: sorun modelde değil kuralda. Kullanıcı önce bir karar modeli (karar ağacı) önerdi; train keşfinde ağaç elendi (aşağıdaki girdi), zaman kuralını kullanıcı onayladı ("video kısmından ilerleyelim"), ayrıntılar Claude'a bırakıldı.
  2. **Karar değişikliği:** dikdörtgen (640×512) export Faz 3'te değil Faz 5'te (2026-10-17 kararı "Faz 3'te C++ motoruyla birlikte" idi). Neden: sevk edilen modeli değiştirir, Colab'da val ölçümü ve ADR 0007 sevk kuralının yeniden uygulanmasını ister.
  3. M3OT test dizilerindeki sağlık ölçümü yalnız raporlanır, hiçbir eşik ona bakılarak değişmez; demo ölçütü tutmazsa sınırlama notu ve demo için test dışı bir M3OT dizisi.
  4. Faz 3'te (S) görev yok: bütün kodu Claude yazar, kullanıcı inceler.
  5. Ek ölçümler girer: motor tespitleriyle M3OT AP50 + ECE, kapının M3OT'ta sentetik bozulmayla denenmesi, internetsiz kurulum paketi (`docker save`). Donmuş kare girmez.
  6. Takip eşliğinde belgelenmiş kayan nokta eşitliği istisnası kabul (kural ölçümden önce, tasarım §6.1 Kabul 4).

## 2026-10-18 · F3 · Hizasızlık alarmı: train keşfi (karar ağacı elendi)
- **Soru:** Faz 2'nin temiz karedeki hizasızlık yanlış alarmları (val %4,1) bir karar modeliyle (karar ağacı) mı, yanıt eşiğiyle mi düzelir?
- **Kurulum:** yalnız train (`data/health/train_features.csv`, 500 görüntü × 16 koşul, Faz 2 özellikleri); geçici betik (repoda değil), laptop CPU, saniyeler. Val ve test kullanılmadı. Sayılar keşif; kayıtlı sonuç Faz 3'te `results/health/gate_model.json`.
- **Bulgular:**
  1. Train'in 500 temiz karesinin 25'inde (%5) hizasızlık alarmı; iki grup: **(a)** 12 kare, neredeyse hepsi karanlık, yanıt ≤ 0,08, kayma çoğu 38-248 px (anlamsız kestirim); **(b)** 13 kare, çoğu aydınlık, yanıt 0,16-0,69, kayma 16-45 px — büyük olasılıkla veri setinin gerçek RGB-IR kayması (E0: merkez kayması p99 16,6 px; Faz 2 eşiği 16,16 px; sentetik şiddet 20 px).
  2. Yanıt eşiği (bütün train, çapraz doğrulamasız): r 0,10 → temiz alarm %2,6, misalign-20/40 yakalama %94,5 (karanlıkta %89) · r 0,20 → %2,2, %89,0 (%74) · r 0,30 → %1,6, %79,1 (%47).
  3. Karar ağacı (Faz 2'nin 9 özelliği, derinlik 2-3, yaprakta ≥ 20, görüntüye göre gruplu 5 katlı çapraz doğrulama, ≤ %2 temiz alarm ölçütü içte seçilerek): katlar dışı yakalama derinlik 2-3'te ≈ %0, `misalign_px` çıkarılınca ≈ %18 (temiz alarm %1,0). Üst dallar RGB / IR gürültü kestirimi: ağaç kaymanın güvenilirliğini değil sentetik **bozulma türünü** öğreniyor.
- **Yorum:** (a) grubunu yanıt eşiği temizler; (b) grubu tek kareye bakan hiçbir ölçüyle sentetik 20 px kaymadan ayrılamaz (bilgi eksik, model değil). Videoda gerçek kayma kareden kareye sabit, anlamsız kestirim zıplıyor → zaman tutarlılığı kuralı (tasarım §4.2). Karar ağacı tasarımdan çıkarıldı.

## 2026-10-18 · F3 · Tasarımın ikinci bağımsız incelemesi (Codex)
- **Kaynak:** kullanıcının Codex'e yaptırdığı salt okuma inceleme (`docs/design/faz3.md`, commit `c700b77`); 2 kritik, 7 önemli, 2 küçük bulgu. Hepsi kabul edildi, tasarımın v3'üne işlendi.
- **Değişenler:** zaman kuralı için uçtan uca eşlik testi ve 3 px sınırına uzaklık kuralı; Kabul 1-8 için "çıkışı engeller / yalnız raporlanır" tablosu; takip istisnasının mekanik kanıtları ve "ilk farkta dur, ≥ %90 kare" kuralı; `r_min` ve `gate_model.json` fikstür üretiminden ve Release'ten önce; **M3OT test yalnız son ölçüm** — fikstürler, takip eşliği, demo dizisi ve demo ölçütü M3OT val'den (kullanıcı kararı 3 daraltıldı); kare başına alarm sözleşmesi; M3OT sağlık protokolü (Faz 2 `shift(ir, 20, 0)` yatay sabit, gecikme tanımı, dizi başına ölçüt); `/api/detect`'in iki istek biçimi; çıktı katmanları (değerlendirme 0,001 / takip 0,1 / gösterim conf); Playwright ile tarayıcı isteklerinin kaydı; Kabul 7'de FR listesi; spec §8.4 istisnası §10'a.
- **Ek bulgu (Claude):** Spike 04'e göre bazı M3OT dizilerinde kare numaraları ardışık değil; ardışık olmayan dizi demo, klip ve takip fikstürü adayı olamaz.

## 2026-10-18 · F3 · Plan incelemesi (Codex) ve Kabul 1 sınır kuralları
- **Kaynak:** kullanıcının Codex'e yaptırdığı salt okuma plan incelemesi (`docs/plans/F3.md` ilk taslağı); 3 kritik, 10 önemli, 5 küçük bulgu; hepsi kabul, plan yeniden yazıldı (37 görev, sıralı numara).
- **Tasarım değişikliği (ölçümden önce, açık karar):** ilk taslak Kabul 1'i sessizce daraltmıştı (değerlendirme katmanı yerine yalnız takip ve gösterim; satır dışlama; ham satır yer değiştirmesi). Tasarım §6.1 Kabul 1'e yazılan kural: (a) ham çıktı satır satır ≤ 1e-4, yalnız ardışık skorları `1e-5` içindeki Python çiftinde IoU eşlemesi; (b) değerlendirme, takip ve gösterim katmanlarında pipeline'ın kendi sıralı çıktısı; sınıra `1e-4`'ten yakın satır serbest, eşit kalibreli ve ham farkı `1e-4` içindeki komşular yer değiştirebilir, her katmanda karşılaştırılan oran ≥ %99; (c) kalibrasyon double hassasiyette (float32'de tablo değerleri değişiyordu), blok sınırlarında birebir. §4.6 fikstür kuralında ham skorlar için kare eleme kaldırıldı (80 blok sınırı nedeniyle uygulanamazdı); `response` uzaklığının koşullu olmasının gerekçesi yazıldı.
- **Plandaki diğer düzeltmeler:** CI'da eşlik testleri atlanamaz (`BAYKUS_REQUIRE_FIXTURES=1`, testler `docker run` ile model ve fikstür bağlanarak); test bölümü Colab'da dönüştürülür ama ayrı klasörde kalır ve bütün CLI'lar `--i-know-this-is-test` ister; geliştirme bölümü ve demo dizisi `configs/m3ot_dev.json`'da; Faz 2'de değerlendirilmeyen koşullarda taban tahmini; `reason_code` / `reason` ayrımı; modül ve pipeline fikstürleri ayrı, fikstür manifestosu; ortak `Runtime` yolu (takip ve `track` süresi dahil); ölçüm adaptörü son ölçümden önce geliştirme bölümünde sınanır; elle kabul sistem dondurulmadan önce; taşınabilir paket boş klasörde sınanır; süre tablosu ve 5 günlük tampon (≈ 7-8 hafta).
- **Doğrulama (Claude, kullanıcı isteğiyle; Codex'in sözüne dayanılmadı):** 18 bulgunun hepsi ilk taslak metne ve koda karşı tek tek denetlendi; hepsi doğru çıktı. Kodla doğrulananlar: kalibrasyon tablosunun 80 `y` değerinin 75'i float32'de değişiyor ve 80 `x`'in 80'inde double `nextafter` komşuları float32'de aynı değere düşüyor (ilk taslaktaki sınır testi sınırı hiç sınamazdı); `test_health_gate.py:77` `pair` sözlüğünü bütünüyle eşitliyor; `suite.py` kapı piksel değiştirmediyse kapılı değerlendirme yapmıyor; `REASONS` kodu metne çeviriyor; `ci.yml` yalnız `main` push ve PR'da; spec atıfları (FR-V9, NFR-1, §8.4, §20.4 DoD, test kuralı) yerinde. Kabul 1'de Codex'in iki seçeneğinden "açık karar" seçildi (tam geri dönüş farklı işlemcide sınır skorlarında kararsız olurdu). **Ek bulgu (Claude):** Python `darken` `np.rint` (yarıda çifte) kullanıyor, JS `Math.round` yarıda yukarı yuvarlıyor; UI karartması `.5` piksellerde Python'dan ayrılırdı → planda float32 aritmetiği ve yarıda çifte yuvarlama + test.

## 2026-10-18 · F3 · M3OT dönüşümü (val, train, test) ve geliştirme bölümü
- **Kurulum:** Colab CPU, `notebooks/colab/09_m3ot_val.ipynb` (dal `f3`). Kaynak: notebook hücre 3-5 çıktıları (kullanıcı yapıştırdı).
- **İndirme:** `M3OT.zip` 10,74 GB, 6 dk 49 sn (hücre 3, tarayıcıdan kopyalanan cURL). İlk denemelerde figshare adresleri Colab'a dosya yerine bot doğrulama sayfası döndürdü (`figshare.com/ndownloader` HTTP 202 ve 403, `ndownloader.figshare.com` 202, `api.figshare.com` 202); yönlendirilen S3 adresi 10 sn geçerli (`X-Amz-Expires=10`). Hücreye `-L` (yönlendirmeyi izle) eklendi (`3011189`).
- **Dönüştürme** (`baykus-data m3ot --splits val train test`, 9 dk 14 sn): 10.790 RGB-IR çifti, 18 dizi, eşleşmeyen dizi yok; yalnız RGB'de olan 1 kare, yalnız IR'de olan 0; kutu RGB 127.932, IR 126.832; süzülen satır 0. Diskte: ham 12 GB, `rgb` 1,3 GB, `ir` 2,0 GB, `coco` 35 MB, etiketler 43 MB × 2, `meta` 23 MB.
- **Diziler** (`baykus-data m3ot-seqs`; dizi adlarında `-` → `_`): val `1_08`, `2_08` (600'er kare); train 14 dizi (`1_01`/`2_01` 1365, `1_02`/`2_02` 700, `1_04`/`2_04` 300, `1_05`/`2_05` 300, `1_06`/`2_06` 600, `1_07`/`2_07` 450, `1_09`/`2_09` 600 kare); test `1_03`, `2_03` (480'er kare). **18 dizinin hepsinde kare numaraları ardışık** (dosya adındaki numarayla, iki modalitede de olan kareler). Spike 04'teki "kare numaraları ardışık değil" gözlemi bu dönüşümde görülmedi.
- **Karar (tasarım §4.6, ölçümden önceki kural):** val'de ardışık ve iki modaliteli dizi var → geliştirme bölümü `val`, diziler `1_08`, `2_08` (`configs/m3ot_dev.json`, `demo: null`; demo dizisi Görev 12'de demo ölçütüyle). Test dizileri `1_03`, `2_03` yalnız paketlenir.
- **Düzeltme:** notebook hücre 7 test dizilerini `1-03 2-03` diye yazıyordu; dönüştürücü adları `1_03 2_03`.
- **Paketler (hücre 7, 22 sn):** `demo-pack` geliştirme `1_08`, `2_08` → `dev_pack` 344 MB (zip 339 MB); test `1_03`, `2_03` (`--allow-test`) → `test_pack` 308 MB (zip 305 MB). Drive `MyDrive/baykus/m3ot/{dev_pack,test_pack}.zip`.
- **Laptop:** `data/m3ot_dev/` (val 1200 kare, 13.474 IR kutusu), `data/m3ot_test/` (960 kare, 2465 IR kutusu: E7'deki sayıyla aynı), `demo-data/sequences/` = `1_08`, `2_08` (test dizileri demo-data'ya girmedi). Her dizide kareler 1..N, 640×512, `index.json` fps 30. Hiçbiri git'te değil.

## 2026-10-18 · F3 · Kare kuralı, val keşfi (PyTorch E3c)
- **Etiket:** val'de keşif. Kural (`r_min`) train'de seçildi ama tasarım val sonuçlarına bakılarak kuruldu; kapılı koşular Faz 2'deki **PyTorch** E3c (sevk edilen ONNX değil). Test kullanılmadı. Kaynak `results/health/gate_model_val.json`.
- **Kurulum:** `r_min` = 0,2450 (`results/health/gate_model.json`, train: temiz alarm %2,0, misalign-20/40 yakalama %85,5 · karanlık %61,7 · aydınlık %95,7). Faz 2 dosyaları (Drive `robust/`, 46 dosya, 69,6 MB) kare kuralıyla birleştirildi: `reason == "IR hizasız"` ve `response < r_min` olan karelerin tahmini taban koşudan, diğerleri kapılı koşudan. Ön koşul (kapının dokunmadığı karelerde kapılı = taban tahmin) **16 koşulun 16'sında tuttu**. Eşleştirilmiş bootstrap 1000, seed 42; Faz 2'nin Kabul 1-3 tanımları. Laptop, `baykus-health guard-val`.
- **Sonuç (Faz 2 kapısı → kare kuralıyla kapı):**

| | Faz 2 | Kare kuralı |
|---|---|---|
| Kabul 1: temiz fark (≥ −0,5 puan) | −3,09 ❌ | **−0,22** ✅ (GA −0,50 / −0,01) |
| Kabul 2: Z birleşik fark, GA alt sınır > 0 | +40,7 [38,3; 43,1] ✅ | **+33,8 [31,3; 36,7]** ✅ |
| Kabul 3: temiz yanlış alarm (≤ %2) | %4,08 ❌ | **%0,68** ✅ |
| misalign-20 kapı oranı / fark | %95,0 / +41,6 | %83,6 / +35,7 |
| misalign-40 kapı oranı / fark | %99,7 / +62,1 | %86,7 / +47,4 |
| noise-20 fark | +18,4 | +18,4 (gürültü kuralı değişmedi) |

  - Kaldırılan alarm: temiz 50 (60 alarmın 50'si), karartma 57-68, bulanık 52-60, misalign-20 168, misalign-40 191 kare.
  - misalign-20 yakalama: tümü %83,6, **karanlık %58,8**, aydınlık %93,4 (misalign-40: %86,7 · %61,4 · %96,6). Tasarımda yazılan bilinen risk ölçüldü: gece gerçek kaymada da yanıt düşük, kare kuralı o alarmı da iptal ediyor.
- **Yorum:** kare kuralı Faz 2'nin ana sorununu (temiz val'de %4 yanlış alarm, −3,1 puan) val'de gideriyor; bedeli karanlıkta gerçek kaymanın ≈ %40'ını kaçırmak. Zaman kuralı tek karelik DroneVehicle'da ölçülemez (M3OT videolarında, Görev 12). Ürün varsayılanı değişmez: `gate=advise` (tasarım §4.2; testte doğrulanmadı).

## 2026-10-18 · F3 · M3OT sağlık kontrolü (V, geliştirme bölümü) ve demo dizisi
- **Kurulum:** tasarım §4.7 protokolü, `baykus-health m3ot --dev --pick-demo`, laptop CPU, Python referans pipeline'ı (ONNX FP32, `models/`, manifest denetimli). Val `1_08`, `2_08` (600'er kare); koşullar `clean`, `misalign-20`, `misalign-20@K` (K = 300), `noise-20`, `drop_ir`; üç kural; AP50 IR etiketiyle, sınıfsız, ham ≥ 0,001. Kaynak `results/health/m3ot_dev_health.json`. **Yalnız raporlanır.**
- **Sonuç (birleşik, 1200 kare):**

| Kural | temiz hizasız alarm | misalign-20 alarm | gecikme (@K) | temiz AP50 auto / off | noise-20 AP50 auto / off |
|---|---|---|---|---|---|
| Faz 2 | %39,7 | %74,8 | 1 / 0 kare (1_08 / 2_08; 2_08'de K−1'de alarm zaten açık) | 0,148 / 0,396 | 0,001 / 0,162 |
| kare (`r_min`) | %0 | %0 | yok | 0,396 / 0,396 | 0,001 / 0,162 |
| kare + zaman | %0 | %0 | yok | 0,396 / 0,396 | 0,001 / 0,162 |

  - Kayıp ve gürültü kuralları M3OT'ta da çalışıyor: `drop_ir` IR kayıp alarmı %99,7, `noise-20` IR gürültülü alarmı %99,7 (pencerenin ilk iki karesi hariç).
  - `misalign-20` E3c'yi M3OT'ta çökertiyor (AP50 0,396 → 0,001); kapı kararı yok, AP aynı.
- **Neden (kare örnekleri, laptop):** M3OT'ta temiz karelerde ölçülen RGB-IR kayması büyük ve değişken (1_08'den 5 kare: 4-44 px), faz korelasyonu yanıtı hep düşük (0,03-0,17; DroneVehicle'da temiz ortanca 0,56), kareler aydınlık (parlaklık ≈ 107). `r_min` = 0,245 bu yüzden bütün hizasızlık alarmlarını kaldırıyor; sentetik kayma vektörde doğru görünüyor (dx her karede +20) ama alarm üretmiyor. Faz 2 kuralı ise temiz karelerin %40'ında yanlış alarm veriyor.
- **`gate=auto` AP'si:** kapı IR'yi kapatınca AP50 ≈ 0 (temizde Faz 2 kuralıyla 0,396 → 0,148, noise-20'de 0,162 → 0,001). IR etiketiyle ölçüldüğü ve M3OT'ta RGB-IR etiketleri ≈ 12 px'lik kutular ölçeğinde hizasız olduğu için (`docs/data/m3ot.md`) RGB'ye kalan tespitler IR etiketiyle eşleşmez; bu sütun kapının kalitesini değil etiket hizasını ölçer.
- **Demo ölçütü** (kare + zaman, ölçümden önce sabit): iki dizide de temiz ≤ %2 ✅, misalign-20 ≥ %90 ❌, gecikme ≤ 10 ❌. Kural (tasarım §4.6): hiçbiri tutmazsa ad sırasıyla ilk dizi + sınırlama notu → **demo `1_08`** (`configs/m3ot_dev.json`). Sınırlama: M3OT demosunda hizasızlık alarmı (IR kaydırma arıza simülasyonu) görünmez; kayıp ve gürültü alarmları görünür. README ve `docs/system.md`'de böyle yazılacak.
- **Yorum:** eşikler DroneVehicle'da seçildi; M3OT'ta hizasızlık kestirimi (kenar faz korelasyonu) bu iki dizide güvenilir değil. Tasarım bunu öngörmüştü ("demo hikâyesi ölçülmeden yazılmaz"); eşik ya da kural M3OT'a göre değiştirilmez (test'e sızma olmasın, (T) yalnız raporlanır).

## 2026-10-18 · F3 · Motor gecikmesi, laptop Docker (Kabul 5)
- **Kurulum:** `baykus_bench` (plan F3 Görev 27), runtime imajı `baykus-engine`, bellek içi kip (`Runtime`, HTTP hariç) ve `--http` kipi (compose `back` iç ağı, kapalı ağ katmanı). Girdi `data/edge/bench_inputs` (H5'teki 20 DroneVehicle val çifti, döngüde), `gate=advise`, takip açık, tek oturum, conf 0,25; 50 ısınma + 500 ölçüm; ORT iş parçacığı varsayılan. Laptop prizde (Win32_Battery durumu 2), Docker Desktop 4.62.0 (219486), `.wslconfig` yok (WSL varsayılanı: Docker sanal makinesi 16 çekirdek, 7,96 GB). Kaynak `results/bench/e3c_onnx_fp32_engine_laptop-docker.json` ve `..._laptop-docker-http.json`.
- **Sonuç (ms, p50 / p95):**

| Aşama | Bellek içi | HTTP kipi (sunucudaki süreler) |
|---|---|---|
| decode | 6,51 / 7,68 | 6,71 / 7,60 |
| health | 8,72 / 10,77 | 7,56 / 9,43 |
| pre | 3,18 / 4,39 | 3,25 / 4,44 |
| infer | 45,88 / 62,11 | 45,72 / 62,06 |
| post | 0,00 / 0,01 | 0,00 / 0,01 |
| track | 0,13 / 0,20 | 0,12 / 0,18 |
| **total** | **64,81 / 82,19** | 63,31 / 80,80 |
| HTTP dahil gidiş-dönüş | — | 65,84 / 84,68 |

  - Bellek içi: 14,8 FPS (1000 / ortalama total), tepe RSS 287 MB.
- **Kabul 5 (motor p95 ≤ 100 ms, Docker): ✅ 82,2 ms.**
- **Yorum:** H5'te aynı model Python ORT ile laptop CPU'da (Docker'sız) pre + infer + post p95 59,9 ms'ti; burada infer p95 62,1 ms (Docker / WSL katmanı ve kuyruk), üstüne Faz 3'te eklenen decode (≈ 7,7) ve sağlık (≈ 10,8) geliyor. HTTP katmanı p95'e ≈ 4 ms ekliyor (eşik yok). Ek ölçüm yapılmadı (yalnız sevk edilen yol).

## 2026-10-18 · F3 · Kapalı ağ kanıtları 1-4, laptop (Görev 33)
- **Kurulum:** `scripts/offline_proof.sh` (CI `compose` işiyle aynı adımlar), imajlar `baykus-engine` (runtime) ve `baykus-ui` (Next 16.4.2 standalone, node:22-bookworm-slim), `/data` = `fixtures/m3ot/clip` (15 kare), Docker Desktop 4.62.0, Docker Compose v5.0.2. Kanıt 3 yerelde sistem Chrome'uyla (`PW_CHANNEL=chrome`), CI'da Playwright Chromium'u. Kaynak `results/system/offline.json`.
- **Sonuç (kapalı katman, `compose.offline.yml`):** kanıt 1 `offline_check` geçti (klip kare 1, `/api/detect` 200, 10 tespit, dört model sha256'sı `models.json` ile aynı); kanıt 2 ui ve motor konteynerinde 1.1.1.1:443 → `ENETUNREACH`, example.com → `EAI_AGAIN` (engelli). **Kanıt 3** (normal compose): 34 tespit isteği (3'ü IR 20 px kaydırma arızasıyla multipart), 2 tur başı oturum sıfırlaması, 127.0.0.1:3000 dışına istek 0. **Kanıt 4:** imajdaki `.next` (148 dosya) taramasında izin listesi dışı adres 0.
- **Ölçümün gerçekten ağı ölçtüğünün denetimi (bir kez, elle):** normal compose'da ui konteynerinden `egress_check` **düştü** (1.1.1.1:443 bağlantı kuruldu, example.com çözüldü: "dış adrese ulaşıldı", çıkış 1); motor normal compose'da da engelli kaldı (yalnız `internal` `back` ağında). Kanıt 3'ün sahte kopyası tarayıcıdan tek bir `https://example.com/x` isteği eklenince düştü.
- **Kabul 6 (otomatik kısım):** yerelde ✅; CI yeşili push sonrası. Elle Wi-Fi kapalı demo Görev 35'te.

## 2026-10-19 · F3 · Elle kanıtlar: Wi-Fi kapalı demo ve ekran (Görev 35)
- **Kurulum:** normal compose (`docker compose up`, imajlar 2430d60 sonrası), `127.0.0.1:3000`, demo dizisi M3OT val `1_08` (600 kare). Kaynak `results/system/manual.json`, görüntüler `docs/figures/system/`.
- **Kabul 7 (ekran):** 20 madde (FR-U1 oynatıcı, FR-U2 görünümler, FR-U3 conf / takip / kapı / gecikme grafiği, FR-U4 beş arıza, sağlık paneli, alarm başlama / temizlenme / sıfırlama, kapı rozeti) Claude tarafından Chrome eklentisiyle sınandı; hepsi `ok`. IR 20 px kaydırmada "IR hizasız" alarmı M3OT'ta çıkmıyor (eşleşme yanıtı `r_min` altında; Görev 12'deki bilinen sınırlama).
- **Kabul 6 (elle):** kullanıcı Wi-Fi'yi kapatıp sayfayı yeniledi; oynatma ve tespit çalıştı (`01-wifi-kapali.jpg`: kare 306, conf 0,10, 10 tespit, motor p95 96,4 ms son 100 karede, arıza kapalı). `wifi_off: true`.
- **Kullanıcı gözlemi (ölçüm değil):** bazı araçlar kutusuz kalıyor; motosikletler tespit edilmiyor. Model DroneVehicle'ın 5 sınıfıyla (car, truck, bus, van, freight_car) eğitildi, motosiklet sınıfı yok; M3OT'ta tespit gücü val'de AP50 0,396 (`results/engine/m3ot_dev_detect.json`). Sınırlama olarak raporlanır (Görev 37); Faz 3'te model değişmez.

## 2026-10-19 · F3 · Son ölçüm: M3OT test (Görev 36, bir kez)
- **Kurulum:** sistem donduruldu (kullanıcı onayı; etiket `f3-freeze` = `b25b175`; donmadan sonra yalnız ölçüm aracı eklendi: `baykus-track oracle` ve `scripts/m3ot_measure.sh`, ikisi de önce val'de sınandı). Test dizileri `1_03`, `2_03` (2 × 480 kare, 2465 IR kutusu); sınıfsız `vehicle`, IR etiketleri. Tespit C++ `baykus_detect` (motor imajı, ham ≥ 0,001), AP50 pycocotools, ECE 15 aralık ham ≥ 0,25; takip C++ `baykus_track` (takip katmanı ≥ 0,1, `trackers` 2.6.1 varsayılanları), `trackers eval`; sağlık `baykus-health m3ot --test` (Python referansı, laptop CPU). Kaynaklar `results/engine/m3ot_test_detect.json`, `results/track/m3ot_test_{engine,oracle}.json`, `results/health/m3ot_test_health.json`; val provası `m3ot_dev_*`.
- **Tespit (birleşik, 960 kare):** AP50 **0,860** (`1_03` 0,896 · `2_03` 0,820); ECE ham 0,361 → kalibre 0,347; çift kutu oranı 0,012 (takip girdisi, IoU ≥ 0,7 farklı sınıf). Faz 1 E3c M3OT AP50 0,889 [0,877; 0,902] (PyTorch + `agnostic_nms`, E7) yanında: fark −2,9 puan; tasarım §4.3'e göre tek bir nedene bağlanmaz (ONNX, 300 kutu sınırı, sınıflar arası çift kutu, NMS yok). Val provasında aynı komutlar AP50 0,396 (kare başına ≈ 11,2 kutu; test ≈ 2,6).
- **Takip (birleşik):** motor HOTA **0,489** · MOTA 0,547 · IDF1 0,677; oracle HOTA **0,949** · MOTA 0,981 · IDF1 0,941. Takipçi doğru kutularla neredeyse kusursuz; motor satırındaki düşüş tespitten (ADR 0008).
- **Sağlık (T, birleşik, 960 kare; yalnız raporlanır):**

| Kural | temiz hizasız alarm | misalign-20 alarm | gecikme (@K = 240) | temiz AP50 auto / off | noise-20 AP50 auto / off |
|---|---|---|---|---|---|
| Faz 2 | %35,1 | %83,0 | 0 / 0 kare | 0,569 / 0,860 | 0,033 / 0,818 |
| kare (`r_min`) | %0 | %0,4 | — | 0,860 / 0,860 | 0,033 / 0,818 |
| kare + zaman | %0 | %0 | yok | 0,860 / 0,860 | 0,033 / 0,818 |

  - `drop_ir` IR kayıp alarmı %99,6, `noise-20` IR gürültülü alarmı %99,6 (pencerenin ilk iki karesi hariç); yanlış kayıp / gürültü alarmı temizde %0.
  - `misalign-20` E3c'yi testte de çökertiyor (AP50 0,860 → 0,006); ürün kuralı (kare + zaman) bu kaymayı yakalamıyor — val'deki bulgu testte tekrarlandı. Demo ölçütü iki test dizisinde de tutmadı (temiz ≤ %2 ✅, misalign-20 ≥ %90 ❌, gecikme ≤ 10 ❌).
  - Python sağlık pipeline'ının temiz `off` AP50'si (0,8605) C++ `baykus_detect` ölçümüyle aynı.
- **Yorum:** Faz 3 motoru M3OT test'te Faz 1 ölçümüne yakın tespit veriyor; takipçi sağlam, sınır dedektörde. Sağlık kayıp ve gürültüde güvenilir; hizasızlık kestirimi M3OT'ta güvenilir değil (Faz 2 kuralı temizde %35 yanlış alarm, ürün kuralı gerçek kaymayı kaçırıyor). Kullanıcı gözlemi (motosiklet sınıfı yok, val sahnelerinde kaçan araçlar) sınırlama olarak `docs/system.md`'ye. Hiçbir eşik ya da kural bu sonuçlara göre değiştirilmedi.

## 2026-10-19 · F3 · Faz çıkışı (Görev 37)
- **Rapor:** `baykus-system report` → `docs/system.md` (mimari, API, Kabul 1-8 kanıt dosyalarından, gecikme, M3OT tespit ve takip, val keşfi, M3OT sağlık, sınırlamalar); kanıt dosyası yoksa satır düşer.
- **Kabul 1-8:** hepsi ✅ (1-2 `parity_detect`, 3 `parity_health`, 4 `parity_track` 600/600 × 2, 5 p95 82,2 ms, 6 kanıt 1-4 + Wi-Fi kapalı demo, 7 20/20 ekran maddesi, 8 15/15 hata durumu + 1000 rastgele istek). Tasarım §6.7'ye göre faz çıkışına izin veriyor → spec S7, S8 işaretlendi.
- **README:** Faz 3 bölümü (ekran görüntüsü, sayılar, sınırlamalar, internetsiz kurulum üç komut, "H5 60 ms Windows yerel; motor Docker'da; karşılaştırılamaz", M3OT atfı).
- **Sürüm:** `main` ← `f3`, etiket `v0.2.0`.

## 2026-10-20 · F4 · Val keşfi: kamera hareketi ve hareketli/duran ayrımı
- **Kurulum:** yalnız M3OT val (`1_08`, `2_08`, IR, GT izleri); geçici betik (repoda değil), laptop CPU. Test kullanılmadı. Sayılar keşif; kayıtlı sonuç Faz 4 Görev 1'de `results/motion/explore_val.json`.
- **Bulgular:** kamera hareket ediyor (öteleme ortanca 1,50 / 1,06 px/kare, dönme ≈ 0,02°/kare, ölçek ≈ %0,2/kare yaklaşma); ardışık kare zinciri 600 karede ölçeği 4,3 / 3,0 kat şişiriyor, GT kutu boyu yalnız ≈ 1,3-1,6 kat büyüyor → anahtar kareye kayıt şart. Telafi edilmiş 15 karelik hız / kutu boyu iki tepeli: `≤ 0,03` 1309 / 2381, bant 412 (%8,3) / 848 (%11,7), `≥ 0,10` 3253 / 4043 (`1_08` / `2_08`).
- **Sonuç:** eşikler 0,03 / 0,10 ve `K = 15` bu keşiften, ölçümden önce sabit (tasarım §4, §5.2); val sonuçları "val'de keşif".

## 2026-10-20 · F4 · Tasarım incelemesi (Codex) ve kararlar
- **Kaynak:** `docs/design/faz4.md` v1'in Codex incelemesi (12 bulgu + §10 kararlarına tutum). Her bulgu kodla ve belgeyle doğrulandı; tablo tasarım §12'de.
- **Kabul edilenler (v2):** harita nesli sözleşmesi (kanal değişimi / uzun kayıpta yeni nesil, kısa kayıpta yeniden bağlama); süzgeçsiz `tracks[]` (bölge olayları güven kaydırıcısından bağımsız; `runtime.cpp` kimlikleri yalnız süzülmüş listeye yazıyordu); etiketleyiciden bağımsız seçilen kör insan örneklemi (val kapı, test rapor); uçtan uca klip eşliği (6 senaryo, Release `fixtures-f4`); sabit paydalar, geri çağırma ve kesinlik; oracle işlev kapısı (Kabul 9); iddia kuralı; harita kayması referans kayda karşı; 550 ardışık kare bench; referans hız formülünde payda `t + K`; S6 ve K7 Görev 0'a.
- **Kabul edilmeyenler:** CI takip kapısının "gerçek kapı olmadığı" (kapı `ctest` `Tracker.FixtureParity`, `REQ=1`; `compare` yalnız kanıt dosyası — yine de düzeltilecek); mevcut test bölümünün hareket görevi için kullanılamayacağı (v1'deki tek test atfı val sayısıyla değiştirildi; görev testte hiç ölçülmedi); motor satırına çıkış kapısı (tespit Faz 4'te değişmiyor; olumsuz sonuç raporlanır).
- **Kullanıcı kararı (2026-10-20):** §10/1 = **A** — test yalnız `1_03`, `2_03`; M3OT train'den ek dokunulmamış bölüm yok, Colab yok.

## 2026-10-20 · F4 · Tasarım v3: ürün paketleri
- **Kullanıcı kararı:** "planda basit bir çalışır gibi değil, gerçekten ürün özellikleri göstermeli"; dört paketin hepsi: (A) harita etkileşimi + araç kartı, (B) bölge kuralları (yasak bölge, bekleme sınırı, yönlü sayım hattı) + alarm onayı, (C) olay günlüğü, zaman çizelgesi, anlık görüntü, JSON / CSV / PNG dışa aktarma, (D) bölgeler, alarmlar, günlük ve araç özeti motorda (API). "Zaman yeter, hepsinin olmasını istiyorum" → **kesme sırası yok**.
- **Claude'un eklediği (gerekçe: bölgeler motorda ve oturum boyunca kalıcı olacaksa):** harita belleği ve yeniden konumlama (anahtar kare kütüphanesi, ORB, geçici nesil, birleşme dönüşümü; doğrulanmayan bağlamada bölgeler askıda); atlama `scope=tracks`; API kimlik tabanı. Bölge / özet / günlük için Python referansı yok (veri setinde karşılığı yok, ölçülmez) → C++ senaryo testleri + kullanıcıyla gözden geçirilen klip regresyonu (Kabul 7, 11); yeniden konumlama doğruluğu Kabul 10.
- **Etkisi:** tasarım `docs/design/faz4.md` v3, plan `docs/plans/F4.md` 23 görev, ≈ 9-10 hafta (spec penceresi 8 hafta; takvimin ≈ 3 ay önünde). v3 bağımsız incelemeden geçmedi.

## 2026-10-20 · F4 · Kör insan örneklemi, 1. tur (val) — Kabul 4 ❌, sunum düzeltildi
- **Sonuç** (`results/motion/label_audit_val_r1.json`, 80 düzgün + 40 hedefli, seed 42): (a) uyum 65/69 = %94,2 (Wilson %95 GA 0,860-0,977) < %95 → ❌; (b) emin değilim 0/80 ✅; (c) insan-kararlı örneklerin etiketli nüfustaki oranı 69/80 = %86,3 ✅.
- **Uyumsuzluklar:** 4 düzgün (+ 2 hedefli) örneğin hepsi insan "hareketli", etiketleyici "duran", `v_ref` 0,021-0,029 (eşik 0,03). Hedefli bant örneklerinde (0,042-0,095) insan 15 / 16 "hareketli".
- **Kullanıcı geri bildirimi:** "RGB'de kutu arabanın üstünden kayıyor"; bir aracı "en sonda hareket ediyordu, o yüzden hareketli dedim; hareketsiz sayalım". Cevap sonuç görüldükten sonra **değiştirilmedi** (körlük bozulurdu); kural belirsizliği 2. tur sayfasına yazıldı.
- **İnceleme:** kırpma penceresi görüntüde sabitti; kamera bir saniyede onlarca piksel kaydığı için park sırasındaki araçlar da kutusuyla birlikte kayıyor görünüyordu. RGB'deki kutu IR etiketinden (RGB etiketleri laptopta yok) ve doğal RGB-IR kayması yüzünden araçtan kayıyordu.
- **Düzeltme (tasarım §5.3'ün tek düzeltme hakkı, sunuma):** pencere her karede aracın GT kutusuna ortalanır (kestirici yok), dışı siyah; RGB'de kutu yerine köşe işaretleri; sayfaya çoğunluk kuralı. Etiketleyici ve eşikler değişmedi. 2. tur: seed 43, 1. turun örnekleri hariç (`data/motion/audit_r2/`).
- **Not (karar değil):** 2. turda da 0,02-0,03 aralığındaki araçlara insan "hareketli" derse, sorun sunum değil tanımdır (`lo = 0,03` yavaş akan trafiği "duran" sayıyor); o zaman B planı ya da tanım değişikliği kullanıcı kararıdır.

## 2026-10-20 · F4 · Bulgu: M3OT val'de RGB ve IR kareleri hizalı değil
- **Nasıl görüldü:** kullanıcı kör örneklemin RGB panelinde kutunun aracın üstünden kaydığını gösterdi. Aynı karenin RGB ve IR'si yan yana açıldı (val `2_08` 434, `1_08` 100 ve 450): görüş alanı, ölçek ve konum belirgin farklı (yol IR'de ortada, RGB'de sağda). Kesitler arası SIFT ve kenar şablonu eşleştirmesi güvenilir bir dönüşüm bulamadı (skor 0,23-0,34). Ölçüm değil, gözlem.
- **Etkisi:** (1) örneklem sayfasının RGB paneli başka bir yeri gösteriyordu → kaldırıldı, örneklem yalnız IR; tarayıcı kaydı anahtarı tur başına benzersiz yapıldı (2. tur sayfası 1. turun cevaplarını açıyordu; kullanıcı 2. turu cevaplamamıştı, sayfa yeniden üretildi). (2) Faz 4 ego'da RGB yedeği kaldırıldı (tasarım §5.1). (3) Faz 3'teki M3OT gözlemleriyle (temiz karede "IR hizasız" alarmları, val AP50 0,40'a karşı test 0,86) uyumlu; nedensellik ölçülmedi.

## 2026-10-21 · F4 · Kör insan örneklemi, 2. tur (val) — Kabul 4 ✅
- **Sayfa:** yalnız IR, pencere araca ortalı (dışı siyah), çoğunluk kuralı yazılı, seed 43, 1. turun 120 örneği hariç; tarayıcı kaydı tura özgü.
- **Sonuç** (`results/motion/label_audit_val.json`): (a) uyum 72/73 = %98,6 (Wilson %95 GA 0,926-0,998) ✅; (b) emin değilim 0/80 ✅; (c) kapsam 73/80 = %91,3 ✅. Hedefli 40: insan "duran" 14 → etiketleyici duran 12, bant 2; insan "hareketli" 26 → hareketli 11, bant 14, duran 1. Bant örneklerinin çoğunu insan "hareketli" görüyor (bant ölçümden dışlandığı için Kabul'ü etkilemez; yavaş araçlar raporda not edilir).
- **Dondurma:** etiketleyici `motion-labels-v1` (`2811f1c`); val etiketleri bu commit'ten yeniden üretildi, sha256'ları örneklemdekiyle aynı (`data/motion/labels_val_*.json`, `git_dirty: false`).

## 2026-10-21 · F4 · Oracle satırı (val) — Kabul 9 ✅ ve ego raporu
- **Oracle** (`results/motion/m3ot_dev_oracle.json`, `a677048`; GT kutuları + GT kimlikleri, ürünün ego + yeniden konumlama + hareket modülü, IR): etiketli 9813 örnek; **BA 0,949** (iz bootstrap %95 GA 0,925-0,967), **R_mov 0,985**, **R_sta 0,913**, kapsam 0,981, BA_karar 0,975; kararlı bölüm BA 0,967; dizi başına BA 0,958 (`1_08`) / 0,944 (`2_08`). Karar gecikmesi: 33 geçişin hepsi, ortanca 5, p90 13,4 kare. **Kabul 9 ✅** (≥ 0,85 / 0,80 / 0,80). GA yalnız dizi içi örnekleme belirsizliği; iki diziye karşı genelleme güvencesi değil.
- **Kesinlik (tasarım tanımı, payda bütün çıktılar):** P_mov 0,791 (8032 `moving` çıktısı: 6354 doğru, 115 duran GT'de, 1563 dışlanan GT'de — bant / kenar; eşleşmeyen yok), P_sta 0,802. Dışlanan nüfusta ürün: bant 1239 → 572 `moving` / 437 `stationary` / 230 `unknown`; kenar 2422 → 991 / 230 / 1201. Not: demo ölçütü (motor satırı `P_mov ≥ 0,80`, Görev 16) bu tanımla; dışlanan GT'lerle eşleşen çıktılar paydada kalıyor.
- **İnsanla doğrudan** (2. tur kör örneklemi, 120 örnek, insan hepsine karar verdi): 106/120 = 0,883 (Wilson 0,814-0,929); ürün 5'inde `unknown`; ürünün karar verdiklerinde 106/115 = 0,922.
- **Ego raporu** (`results/motion/m3ot_dev_ego.json`, `16b177d`; oracle koşusu): `ok` oranı 0,997 / 1,000; anahtar kare 58 / 48; tek nesil (`1_08`'de 1 birleşme); iç nokta ortanca 151 / 166.
  - **Harita kayması** (ürün `M_{t+Δ}⁻¹·M_t` ile doğrudan SIFT + homografi): Δ15 dört köşe + merkezin en büyüğü ortanca 11,8 / 9,3 px (p95 21,1 / 17,4), **merkez** 1,3 / 1,4 px (p95 6,2 / 4,5); Δ60 köşe 53,0 / 37,1, merkez 9,0 / 7,5; Δ150'de referans kaydı çoğunlukla başarısız (0 / 2 çift). Köşe farkı merkezinkinin ≈ 5-9 katı: köşeler benzerlik modelinin perspektif hatasını taşıyor (§3 homografi satırı), merkez çoğunlukla birikimli kaymayı. Merkez sütunu bu yüzden eklendi (`16b177d`; tanım değişmedi, ek sütun).
  - **Duran araç kararlılığı:** 28 bölüm; `map_xy` std / harita kutu boyu ortanca 0,36, p95 0,89.
  - **Yeniden konumlama** (100 karede bir ±50 kare benzetilmiş atlama): 13/22 başarı (0,59), deneme ortanca 1, en çok 6; kesintisiz koşuyla karşılaştırılabilen 12'de köşe farkı ortanca 0,79 px, p95 2,89 px; **yanlış bağlama (> 10 px) 0**.
  - **Donmuş kare** (temiz val): IR'de donmuş kare oranı 0,015 / 0,013 ve alarm oranı aynı (M3OT IR akışında art arda tekrarlanan kareler var); RGB 0; `freeze@300` alarm gecikmesi 3 kare. Eşik 0,5 ölçümden önce sabit, ayarlanmaz: ürünün oturumunda bu kareler IR donmuş alarmı → ego kaybı (sınırlama).
- **Bilinen davranış (Görev 6, tasarım §5.2 harfiyen):** tek bir ego kaybı `K + 1` kare `v` hesaplatmaz; tutma `K` kare olduğundan karar tam 1 kare `unknown`'a düşer.

## 2026-10-23 · F4 · M3OT araç kapsamı tanısı ve takipçi eşikleri (val) — Görev 15A
- **Neden:** Görev 15 klip gözden geçirmesinde kullanıcı karede 30'dan fazla araç görülürken motorun 0-5 iz verdiğini gördü (tasarım `docs/design/faz4-kapsama.md`).
- **Kapsam** (`results/engine/m3ot_dev_coverage_e3c.json`, `baykus-engine coverage`, E3c ONNX FP32, Python referansı, val `1_08` + `2_08`, IR etiketi, sınıfsız; karede 11,2 etiketli araç, ortanca kutu 13 px):
  | Koşul | AP50 | Bulma (IoU ≥ 0,5) ham ≥ 0,05 / 0,1 / 0,3 / 0,5 / 0,7 | Tespit / kare ham ≥ 0,1 |
  |---|---|---|---|
  | temiz | 0,396 | 0,402 / 0,348 / 0,230 / 0,146 / 0,064 | 5,8 |
  | RGB kapalı | 0,388 | 0,382 / 0,325 / 0,215 / 0,152 / 0,081 | 5,5 |
  | RGB yerine IR | 0,308 | 0,296 / 0,236 / 0,137 / 0,079 / 0,033 | 4,3 |
- **Yorum:** model en düşük eşikte bile araçların %40'ını buluyor; RGB'yi kapatmak sonucu değiştirmiyor, yani val'deki RGB-IR hizasızlığı bu kaybın nedeni değil (küçük araç / alan farkıyla uyumlu; ayrıca ölçülmedi). Geçici tanı betiğiyle 4 parçalı (2 × 2) çıkarım AP50 0,50 vermişti; resmî ölçüm Görev 15D'de (`tiles`).
- **Takipçi eşikleri** (`results/track/m3ot_dev_sweep_e3c.json`, `baykus-track sweep`, 30 çift, kural ölçümden önce tasarımda sabit): bugünkü 0,7 / 0,6 HOTA 0,220, MOTA 0,152, iz/kare 2,46; seçilen **etkinleşme 0,3 / yüksek 0,3**: HOTA 0,292, MOTA 0,207, IDF1 0,436, iz/kare 4,99. En yüksek HOTA 0,294 (0,2 / 0,2) tolerans içinde → kural yüksek eşiği seçti. Izgarada sonuç fiilen iki eşiğin büyüğüyle belirleniyor (ör. etkinleşme 0,7'de yüksek eşik 0,6-0,2 hep 0,220; yüksek 0,6'da etkinleşme 0,6-0,2 hep 0,268).
- **Motor (C++) ile** (`results/track/m3ot_dev_engine.json`, `scripts/m3ot_measure.sh dev`): HOTA 0,220 → **0,292**, MOTA 0,152 → 0,207, IDF1 0,296 → 0,436 (Python ızgarasıyla birebir); oracle değişmedi (0,974). Klip (Release `fixtures-f4`, 8 senaryo) iz × kare 1116 → 5218; Kabul 1 ✅ (hata 0, köşe ≤ 1,8e-5 px), Kabul 4 ✅ (600 / 600 kare), Kabul 10 b ✅.
- **Bölge klibi:** olaylar clean 12 → 24, long_gap 12 → 26 (karede iz 0-5 → 10-15); Claude'un bağımsız kural yeniden hesabı aynı olayları veriyor; kullanıcının görsel gözden geçirmesi model değişiminden sonra (Görev 15C).

## 2026-10-24 · F4 · M3OT öğretmen etiketleri, 1. koşu geçersiz — Görev 15B
- **Koşu** (Colab defter 10 hücre 4, `baykus-data m3ot-teacher`, E3c Release `models-v1`, kod `031063a`, `teacher.json` `created_at` 2026-10-23T18:09:50Z): train 14 dizi, 647 iz; **oy alan iz 2** (`2_04:6` 1 oy, `2_04:7` 2 oy, hepsi `car`), 645 iz oysuz → kural gereği `car`; sınıf sayımı 647 car / 0 diğer.
- **Neden:** hata. Tespit kırpılmış 160 × 128 pencerede yapılıyor, kutular zaten pencere koordinatında; kod pencere köşesini ikinci kez çıkarıp kutuları iz kutusundan kaydırıyordu. Birim testi 64 × 64 karelerle yazılmıştı (pencere köşesi hep (0, 0)), hatayı göremedi. Düzeltme ve 640 × 512 regresyon testi `98d537a`.
- **Sonuç:** bu `teacher.json` kullanılmaz, kör denetim yapılmadı. Hücre 4-5 düzeltilmiş kodla yeniden çalışacak.

## 2026-10-26 · F4 · M3OT öğretmen etiketleri, 2. koşu — Görev 15B
- **Koşu** (Colab defter 10 hücre 3-5, düzeltilmiş kod, f4 dalı `d9066ad`; E3c Release `models-v1`): hücre 3 dönüştürülmüş train'i Drive önbelleğinden açtı (8630 çift, 14 dizi, RGB 111 678 / IR 110 893 kutu, eşsiz dizi yok, yalnız RGB kare 1); hücre 4 27 dk 44 sn, 647 iz: car 640, truck 5, bus 2, van 0, freight_car 0; hücre 5 kör denetim sayfası 100 örnek (`teacher_audit.zip` 8,6 MB, `teacher.json` 463 KB; 1. koşunun dosyası 140 KB — oylar dosyada tutulduğu için büyüme oyların geldiğini gösteriyor, oy alan iz sayısı dosya gelince ölçülecek).
- **Sonraki adım:** kör denetim (kullanıcı, laptop) → `baykus-data m3ot-audit-score` → karar (eşik 0,80: altındaysa hepsi `car`).
