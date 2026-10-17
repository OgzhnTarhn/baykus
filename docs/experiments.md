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
