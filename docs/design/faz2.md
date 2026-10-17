# Faz 2 Tasarımı — Güvenilirlik Farkındalığı (yalın)

> **Tarih:** 2026-10-17 · **Durum:** onaylandı (2026-10-17) · **Dal:** `f2`
> **Dayanak:** spec §2.3 S10, §6.1 kancalar K1-K3, §22 Faz 2, ADR 0006; Faz 1 sonuçları (`docs/experiments.md`,
> `docs/robustness.md`, ADR 0007).
> **Kullanıcı yönergesi (2026-10-17):** "Boş işleri çıkar, iş yoğunluğunu arttırmadan tasarla."

## 1. Neden ve ne

Faz 1 şunu gösterdi: sevk edilen **E3c** (4 kanallı, modality dropout'la eğitilmiş tek model) bir kamera
tamamen kapanınca ayakta kalıyor, ama **bir kamera bozuk veri verirken** çöküyor:

| Koşul (val, mAP50) | E3c | E3c'de IR kanalı sıfır (= "IR kaybı") |
|---|---|---|
| Temiz | 0,851 | 0,720 |
| IR gürültüsü σ 20 | **0,535** | 0,720 |
| IR 20 piksel kayık | **0,266** | 0,720 |
| IR 40 piksel kayık | **0,090** | 0,720 |

Yani bozuk IR'yi **hiç vermemek**, bozuk IR'yi vermekten çok daha iyi. Model zaten kanal kapatmaya göre
eğitildi; eksik olan, **hangi karede hangi kameranın bozuk olduğunu bilmek**.

Faz 2'nin tek cümlesi: **Sistem her karede iki kameranın sağlığını ölçer, bozuk olanı modele vermez,
operatöre nedenini söyler ve skorlarının gerçek doğruluğa denk gelmesini sağlar.**

Mülakat hikâyesi: "IR kamera kaydığında modelim %27'ye düşüyordu. Yeniden eğitmeden, kameranın bozuk olduğunu
fark edip o kanalı kapatan bir kapı ekledim; aynı koşulda %72'ye çıktı ve operatör 'IR hizasız' uyarısını
gördü." (Sayılar Faz 1 ölçümlerinden; Faz 2 bunu kapıyla gerçekten ölçecek.)

## 2. Hedefler

1. **Sensör sağlık dedektörü (A):** Her kare için RGB ve IR'nin durumu: `tamam`, `kayıp`, `karanlık`, `gürültülü`,
   `bulanık`, `hizasız`, ve kapı kararı. Görüntü istatistiklerinden, kural tabanlı; GPU ve eğitim yok.
2. **Arızaya duyarlı kanal kapısı (B) = Faz 2'nin uyarlanabilir füzyonu:** Dedektör bir kameranın modele
   zarar vereceğine karar verirse o kanal sıfırlanır; E3c değişmeden kullanılır.
3. **Güven kalibrasyonu (C):** Sevk edilen E3c · ONNX FP32'nin skorları val'de kalibre edilir, testte ECE
   önce / sonra raporlanır.
4. **S10'un ölçülmesi** (§6): kapı, sabit E3c'ye karşı sağlamlık paketinde güven aralığıyla; arıza tespitinin
   doğruluğu; kalibrasyonun ECE'si.

## 3. Hedef dışı (bilerek kesilenler — "boş işler")

| Kesilen | Neden |
|---|---|
| Güvenilirlik tahmin eden **öğrenen ağ** (CNN, yeni eğitim) | GPU ve haftalar ister; sonuç belirsiz. Kural tabanlı dedektör aynı soruyu ucuz cevaplıyor ve operatöre **neden** söyleyebiliyor. |
| **Geç füzyonda (E4) kare başına WBF ağırlığı** (kanca K4) | E4 CPU'da sevk edilmedi; E4b (aydınlatmaya göre ağırlık) temiz veride yalnız +0,3 puan getirdi. |
| E3c'yi yeniden eğitmek, yeni model | Kapı mevcut modelle çalışıyor ("train only what ships"). |
| Bozulma başına ayrı kalibrasyon | Tek kalibrasyon yeterli; kapı zaten bozuk kanalı kesiyor. |
| Test setinde tüm sağlamlık paketi | 8980 görüntü × 16 koşul çok pahalı; testte yalnız 3 kritik koşul (§6.4). |
| Operatör ekranı, uyarı arayüzü | Faz 3 (sensör sağlık paneli). Faz 2 kare başına sağlık JSON'unu üretir, Faz 3 gösterir. |

**Anti-hedef:** Kapı temiz veride sistemi kötüleştirirse (yanlış alarmla iyi kamerayı kapatırsa) başarısızdır,
bozulmada ne kazandırırsa kazandırsın. Ayrıca: Faz 1'in sayıları değişmez; sevk edilen model aynı kalır.

## 4. Yöntem

### 4.1 (A) Sensör sağlık dedektörü

Kare başına, iki görüntüden hızlı istatistikler (CPU, hedef < 5 ms / kare):

| Durum | Kamera | Özellik (öneri) | Kapı kararına girer mi |
|---|---|---|---|
| `kayıp` | RGB, IR | piksel std ≈ 0 (boş / donmuş kare) | evet (kanal kapanır) |
| `karanlık` | RGB | ortalama parlaklık (ADR 0005'teki ölçü) | **hayır** (E3c karartmada zaten ayakta: ×0,1'de 0,832) — yalnız bilgi |
| `gürültülü` | IR (RGB de hesaplanır) | yüksek frekans gürültü kestirimi (ör. Laplace yanıtının MAD'ı) | evet, yalnız **zararlı** düzeyde |
| `bulanık` | RGB, IR | Laplace varyansı | hayır (iki kamera birlikte bulanıksa kapı çözmez) — bilgi |
| `hizasız` | çift | RGB ve IR kenar haritaları arasında faz korelasyonu kayması (px) | evet, yalnız **zararlı** kaymada |

**Kapı kuralı:** Zarar eşiği Faz 1 ölçümünden: bir koşulda E3c, ilgili kanal kapalıyken alacağından daha düşük
alıyorsa o koşul "zararlı". Val tablosuna göre zararlılar: IR gürültüsü σ 20, hizasızlık ≥ 20 px (10 px
sınırda: 0,696 / 0,720), kamera kaybı. σ 10 zararlı **değil** (0,770 > 0,720): kapı orada kapatmamalı.
Hizasız ve gürültülü durumda kapatılan kanal **IR** (etiketler IR seti, ADR 0003; sentetik kaymada kayan IR).

**Eşiklerin seçimi (val'e dokunmadan):** Eşikler **train bölümünden** seçilmiş 500 görüntüde (seed 42),
sağlamlık paketinin bozulmaları uygulanarak (kanca K2) hesaplanan özelliklerden seçilir: her durum için
"zararlı şiddetler" ile "temiz + zararsız şiddetler" arasını en iyi ayıran eşik (Youden J). Böylece val,
değerlendirme için temiz kalır.

**Çıktı:** Kare başına `health` sözlüğü (`rgb`, `ir`: durum + özellik değerleri; `gate`: `none` / `ir_off` /
`rgb_off`; `reason`). Faz 3'teki sağlık paneli ve alarmlar bunu okur.

### 4.2 (B) Arızaya duyarlı kanal kapısı

Ön işlemeden önce, kapı kararına göre ilgili kameranın pikselleri sıfırlanır (Faz 1'deki "kamera kaybı"
koşuluyla aynı işlem; ölçülmüş davranış). Model, export ve C++ planı değişmez; Faz 3'te motorun ön işlemesine
tek bir adım eklenir.

### 4.3 (C) Güven kalibrasyonu

Sevk edilen E3c · ONNX FP32'nin val tahminleri (kanca K3, `conf ≥ 0,001`) GT'yle eşleştirilir (IoU ≥ 0,5,
mevcut `collect_matches`); skor → "doğru tespit olasılığı" eşlemesi **izotonik regresyonla** (havuzlu bitişik
ihlaller, PAV; bağımlılık eklemeden) öğrenilir. Testte ECE (15 eşit aralık, skor ≥ 0,25, operatör eşiği)
önce / sonra ve güvenilirlik diyagramı. Eşleme monoton olduğu için sıralama, dolayısıyla mAP neredeyse aynı
kalmalı (kontrol edilir). Tamamen laptopta: val tahminleri zaten indirildi, testinkiler Drive'da (birkaç MB).

## 5. Neden bu yaklaşım (literatürle fark)

Spec'te anılan çalışmalar (ör. LER-YOLO, 2026) güvenilirliği ağın içinde **öğrenilen** bir haritayla
modelliyor ve yeni eğitim gerektiriyor. Bu tasarımın farkı: (1) **eğitimsiz** — modality dropout'la eğitilmiş
mevcut modelin "eksik kamerayla çalışabilme" yeteneğini kare başına bir kapıyla kullanıyor; (2) **açıklanabilir** —
her kapı kararının adı konmuş bir nedeni var (operatör için); (3) **ölçülmüş** — kazanç ve zarar güven aralığıyla,
olumsuz sonuç da raporlanır. (LER-YOLO'nun ayrıntıları bu tasarımda yeniden okunmadı; README'de iddia
yazılmadan önce kontrol edilir.)

## 6. Değerlendirme ve kabul ölçütleri (ölçümden önce sabit)

Hepsi IR etiketleri, 1000 bootstrap, seed 42. Kural değişirse eski hâli silinmez, gerekçesiyle eklenir.

### 6.1 Kapı (S10: uyarlanabilir füzyon vs sabit füzyon) — val, sağlamlık paketi (16 koşul + temiz)
- Karşılaştırma: **E3c + kapı − E3c**, koşul başına eşleştirilmiş fark [%95 GA].
- **Kabul 1 (zarar yok):** temiz val'de fark ≥ −0,5 puan (nokta).
- **Kabul 2 (fayda var):** zararlı koşullar kümesi Z = {IR gürültüsü σ 20, hizasızlık 20 px, hizasızlık 40 px}
  için ortalama farkın birleşik bootstrap aralığı tamamen 0'ın üstünde.
- Kamera kaybı koşullarında (zaten sıfır) fark ≈ 0 beklenir; kontrol amaçlı raporlanır.

### 6.2 Arıza tespiti (S10: tespit doğruluğu) — val, sağlamlık paketi
- Durum başına: zararlı şiddetlerde tespit oranı, temiz val'de yanlış alarm oranı.
- **Kabul 3:** temiz val'de kapıyı tetikleyen yanlış alarm ≤ %2 görüntü.
- Bilgi durumları (`karanlık`, `bulanık`) için yalnız oranlar raporlanır, kabul ölçütü yok.

### 6.3 Kalibrasyon (S10: ECE önce / sonra) — test
- ECE önce / sonra, güvenilirlik diyagramı; **Kabul 4:** mAP50 değişimi |Δ| ≤ 0,1 puan.

### 6.4 Test kontrolü (tek sefer)
Testte yalnız temiz, IR gürültüsü σ 20 ve hizasızlık 20 px: E3c ve E3c + kapı (6 değerlendirme). Seçim
yapılmaz; val sonucunun testte de tuttuğunu gösterir.

**Dürüstlük notu:** Zarar düzeyleri (hangi şiddette kapanılacağı) Faz 1'in val ölçümlerinden geliyor; eşikler
train'den seçiliyor. Val değerlendirmesi bu yüzden hafif iyimser olabilir; test kontrolü bunun için.

## 7. İş yükü ve takvim

| | Kim | Tahmin |
|---|---|---|
| Kod: dedektör, kapı, eşik seçimi, kalibrasyon, rapor (TDD) | Claude | 2-3 gün |
| **Colab, tek oturum (A100):** train'de özellik çıkarımı + eşik seçimi, val'de kapılı sağlamlık paketi, testte 3 koşul | Kullanıcı | ≈ 1 saat (tahmin; Faz 1'de E6 koşul başına ≈ 1 dk) |
| Drive'dan E3c ONNX test tahminleri (birkaç MB) | Kullanıcı | 2 dk |
| Belgeler, README bölümü, spec v4.4, `v0.2.0`? (§9) | Claude | 1 gün |

**Toplam:** ≈ 1-1,5 hafta (spec'teki 4 hafta yerine). Yeni eğitim yok, laptopa büyük indirme yok, tek Colab
oturumu.

## 8. Riskler

| Risk | Önlem |
|---|---|
| Doğal kayma (train'de p90 7,2 px, %4,8'i > 10 px) hizasızlık dedektörünü yanlış tetikler | Kabul 3 (≤ %2 yanlış alarm); eşik 20 px zararlılığına göre seçiliyor, 10 px değil. |
| Gece RGB'si "karanlık" ya da "gürültülü" görünür | `karanlık` kapıya girmez; RGB gürültüsü kapıya girmez (zararlı bulunmadı). |
| Sentetik bozulmada iyi çalışan dedektör gerçek arızada çalışmaz | Açıkça sınırlama olarak yazılır; Faz 3'te gerçek dizilerde (M3OT) niteliksel kontrol. |
| Kapı hiçbir şey kazandırmaz | Olumsuz sonuç raporlanır (spec §23); dedektör ve kalibrasyon yine operatör değeri taşır. |

## 9. Builder'a bırakılanlar (uygulama sırasında kararlaştırılır, plana yazılır)

- Özelliklerin kesin formülleri (gürültü kestirimi, kenar haritası, faz korelasyonu ön işlemesi) ve 4 kanalda
  hesap maliyeti.
- Kapı kararında birden fazla arıza aynı anda varsa öncelik (öneri: `kayıp` > `hizasız` > `gürültülü`).
- Kalibrasyon tablosunun dosya biçimi (Faz 3 C++ motoru okuyacak: düz JSON, eşik/değer çiftleri).
- Sürüm: Faz 2 sonunda `v0.1.1` mi `v0.2.0` mı (spec `v0.2.0`'ı Faz 3'e ayırmış; öneri `v0.1.1`).

Uygulama kararları: `docs/plans/F2.md` Spec yorumları 3-11.

## 10. Spec'e yansıyacaklar (onaydan sonra, v4.4)

§22 Faz 2 satırı ve takvim (≈ 1,5 hafta), S10'un ölçütleri (§6), §6.1'de K4'ün Faz 2'de kullanılmadığı notu,
§23 risk satırı.
