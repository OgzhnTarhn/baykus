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
- **E1 ile (eşleştirilmemiş, aralıklara bakarak):** genel 0,761 [0,736; 0,789] vs 0,839 [0,817; 0,859], aralıklar örtüşmüyor. Karanlık gece 0,708 vs 0,872, gündüz 0,818 vs 0,824. Eşleştirilmiş bootstrap (`results/compare/e2_ir-vs-e1_rgb_val_ir.json`, E2 − E1): genel +7,8 [+5,4; +9,9] *, gündüz +0,6 [−1,6; +2,8], gece +4,3 [+0,8; +8,5] *, karanlık gece +16,3 [+12,8; +20,3] * (`*` anlamlı).
- **Başlatma:** E2 logunda "Remapped 3/5 cls head rows from pretrained weights by class name" (8.4.177). 8.4.175'te aynı kod varsayılan açık olduğu için E1 de aynı başlatmayı aldı (`gpu-budget.md`, 2026-10-15).
- **Yorum (taslak, kullanıcıyla netleşecek):** Hipotez gündüz (fark ~0,6 puan) ve karanlık gece (~16 puan) için sayılarla uyumlu; gece farkı (~4 puan) aralıkların içinde. RGB etiketleriyle karanlık gecedeki büyük düşüş, RGB etiket setinin karanlık görüntülerde IR'nin gördüğü nesnelerden farklı/eksik kutular içerdiğini düşündürüyor (E0 bulgu 2-3); E5 yorumunda E3 ile birlikte ele alınacak.

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

- **Yorum (taslak, kullanıcıyla netleşecek):** Hipotezin ilk yarısı tutuyor: E3 genel skorda hem E1'i hem E2'yi anlamlı geçiyor. İkinci yarısı tutmuyor: karanlık gecedeki kazanç E1'e karşı büyük (+16,5) ama bunu IR kanalı tek başına da sağlıyor (E2 − E1 +16,3; E3 − E2 +0,1, fark yok). Füzyonun E2'ye kattığı fark **gündüzde** (+4,3): ışık varken RGB'nin renk ve doku bilgisi işe yarıyor. Gece E3 ile E2 arasında fark yok (aralık 0'ı içeriyor). Kısaca: karanlıkta termal taşıyor, gündüzde RGB katkı veriyor; erken füzyon ikisinden de yararlanıyor ama ikisinin en iyisinden fazlasını vermiyor.

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
- **Yorum (taslak, kullanıcıyla netleşecek):** E0 bulgu 2: IR setinde RGB karşılığı olmayan ≈ 33 bin nesne var; karanlıkta RGB görüntüde görünmeyen araçlar RGB setinde etiketlenmemiş. IR'yi gören modeller bu araçları buluyor ve RGB etiketlerine karşı yanlış pozitif sayılıyor. Etiket seti seçimi karanlık gece sonucunu tersine çevirebildiği için "ana set IR" kararı (ADR 0003) ölçümle destekleniyor; RGB etiketleriyle raporlanan karanlık gece skorları termal modelleri haksız biçimde cezalandırır.

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
- **Yorum (taslak, kullanıcıyla netleşecek):** Erken füzyon temiz veride en iyi model, ama sağlamlıkta zayıf: IR kaybında, büyük hizasızlıkta ve güçlü bulanıklıkta tek modaliteli modellerin altına düşüyor. "Termal kamera kapandığında hangi model dayanıyor?" sorusunun H3 cevabı: E3 dayanmıyor (12,8), RGB modeli zaten etkilenmiyor. Bu, H4'teki E3c (modality dropout) ve E4 (geç füzyon) deneylerinin gerekçesi: amaç temiz skoru koruyup kamera kaybında tek modaliteli modele yakın kalmak. Bulanıklıktaki hassasiyetin nedeni bilinmiyor (hipotez: model iki kanalın ince kenar uyumuna dayanıyor); H4'te E3c ile birlikte bakılabilir.
