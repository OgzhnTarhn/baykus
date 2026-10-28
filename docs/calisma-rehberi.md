# Baykuş — Çalışma Rehberi

> **Tarih:** 2026-10-26 · **Dal:** `f4` · **Kimin için:** projeyi baştan sona anlamak, deftere not almak ve
> mülakatta anlatabilecek düzeyde teknik konulara çalışmak isteyen sen.
>
> Buradaki her sayı başka bir belgeden ya da `results/**/*.json` dosyasından alındı; kaynağı yanında yazıyor.
> Yeni ölçüm yok. `docs/ozet.md` 18 Ekim'de yazıldı ve Faz 3-4'ü içermiyor; güncel özet bu belge.

---

## 0. Bu rehber nasıl kullanılır?

**Defter düzeni önerisi:**
- Sol sayfa **kavram** (tanım, formül, küçük çizim). Sağ sayfa **projede nerede** (dosya, sayı, karar).
- Her bölümün sonundaki **📝 Deftere** kutusunu aynen yaz: en önemli 3-6 satır orada.
- **❓ Kendini sına** sorularını deftere yazma; kapatıp sesli cevapla. Takıldığın soru, tekrar okuyacağın yer.
- Sayıları ezberlemeye çalışma; §15'teki **sayı kartını** ayrı bir sayfaya yaz, yeter.

**Okuma sırası:** §1-3 büyük resim (1 oturum) → §4-9 Faz 1, ML temelleri (3-4 oturum) → §10 Faz 2 (1 oturum) →
§11 Faz 3, sistem (2 oturum) → §12 Faz 4 (2-3 oturum) → §13-17 pratikler, kod haritası, mülakat (1-2 oturum).
§18'de gün gün plan var.

---

## 1. Tek paragrafta proje

Drone'dan çekilmiş **normal (RGB)** ve **termal (IR)** görüntüleri birleştirip yerdeki araçları (car, truck,
bus, van, freight_car) gece-gündüz bulan bir sistem. Amaç yalnız iyi bir model değil: **bir kamera bozulunca
da çalışan**, **laptop işlemcisinde gerçek zamana yakın** koşan, **C++ motoru + web ekranı + internetsiz
kurulumu** olan, araçların **hareketli mi duran mı** olduğunu ayıran ve **her sayısı ölçümle kanıtlı** bir proje.
Hedef: Haziran 2027 mezuniyetinde savunma sanayiinde ML / görüntü işleme mühendisliği başvurusu
(`docs/spec.md` §2.1).

**Ne değil:** anti-drone sistemi değil (drone yalnız kameranın taşıyıcısı). Hedefleme / silah entegrasyonu
kapsam dışı; kullanım çerçevesi gözlem (arama-kurtarma, sınır ve tesis güvenliği).

**Projenin iddiası "ilk ben yaptım" değil** (alan çok çalışılmış). İddia (spec §3.2):
1. **Dürüst ölçüm:** kurallar ölçümden önce yazılır, seçim val'de, test bir kez, olumsuz sonuç da raporlanır.
2. **Operasyonel sağlamlık:** "bir kamera bozulunca ne oluyor?" sistemin içinde ölçülüyor.
3. **Uçtan uca sistem:** model + C++ motor + takip + ekran + kapalı ağ kurulumu bir arada.
4. **Güvenilirlik farkındalığı** (Faz 2) ve **hareketli hedef** (Faz 4).

📝 **Deftere:**
- Baykuş = RGB + IR füzyonlu araç tespiti + takip + hareketli/duran + C++ motor + internetsiz ekran.
- İddia: yenilik değil, **dürüst ölçüm + uçtan uca sistem + sağlamlık**.
- Sevk edilen model: **E3c · ONNX FP32** (erken füzyon + modality dropout).

---

## 2. Büyük resim: veri nereden girer, nereden çıkar?

```
  EĞİTİM TARAFI (Colab GPU, sen çalıştırırsın)
  ───────────────────────────────────────────
  DroneVehicle (Kaggle, ham)          M3OT (figshare, video dizileri)
        │ baykus-data                       │ baykus-data m3ot
        ▼                                    ▼
  kırpma, OBB→HBB, 4 kanallı TIFF,     MOT etiketleri → YOLO/COCO, iz tablosu
  COCO, aydınlatma etiketi
        │ (private Kaggle veri seti: ouzhantarhan/baykus-dv-rgbt)
        ▼
  baykus-train (Ultralytics YOLO26s) → best.pt
        │
        ├─► baykus-eval   (pycocotools mAP50, bootstrap GA, kırılımlar)
        ├─► baykus-robust (16 bozulma koşulu)
        ├─► baykus-fusion (geç füzyon, WBF)
        ├─► baykus-edge   (ONNX FP32/FP16/INT8, TensorRT) → .onnx
        └─► baykus-bench  (p50 / p95 gecikme)
                │
                ▼  her sayı → results/**/*.json → baykus-report → docs/*.md

  ÜRÜN TARAFI (laptop, Docker, internetsiz)
  ──────────────────────────────────────────
  tarayıcı ──► ui (Next.js, 127.0.0.1:3000) ──► engine (C++17, iç ağ)
                                                 │
     decode → sağlık(+donmuş) → kapı → letterbox → ONNX Runtime → çözümleme
     → ego-hareket → ByteTrack → hareket durumu → bölgeler/alarmlar → özet → günlük → JSON
```

**Önemli iş bölümü:** her C++ adımının bir **Python referansı** var (`python/src/baykus/`). C++ çıktısı Python
çıktısıyla karşılaştırılır (**eşlik / parity testi**), bu CI'da her commit'te çalışır.

📝 **Deftere:** Eğitim Python + Colab; ürün C++ + Docker. İkisini birbirine bağlayan şey: **ONNX dosyası** ve
**eşlik testleri**.

---

## 3. Fazlar ve bugünkü durum

| Faz | İçerik | Durum | Sürüm |
|---|---|---|---|
| 1. Çekirdek | Veri, modeller E1-E4b, sağlamlık, M3OT testi, edge ölçümü | ✅ Bitti | `v0.1.0` |
| 2. Güvenilirlik | Kare başına kamera sağlığı, kapı, güven kalibrasyonu | ✅ Bitti, **olumsuz sonuç** | `v0.1.1` |
| 3. Sistem | C++ motor, ByteTrack, operatör ekranı, internetsiz Docker | ✅ Bitti, Kabul 1-8 hepsi ✅ | `v0.2.0` |
| 4. Hareketli hedef | Ego-hareket, hareketli/duran, harita, bölgeler, olaylar, M3OT kapsamı | 🔧 Sürüyor (`f4` dalı) | `v0.3.0` hedef |
| 5. Sunum | README, demo videosu, teknik yazı | Bekliyor | — |

**Faz 4'te nerede kaldık (2026-10-26):**
- Bitti: Python referansları, C++ ego / yeniden konumlama / hareket / bölgeler / günlük / API, ekran I-IV
  (harita, araç kartı, bölge editörü, alarm paneli, zaman çizelgesi, dışa aktarma), CI ve kapalı ağ kanıtı,
  takipçi eşikleri (Görev 15A), parçalı çıkarım kodu (15D), öğretmen etiketi kör denetimi (15B: doğruluk
  0,92 → karar `teacher`).
- Sırada (çoğu senin adımın): Colab defter 10 hücre 6-10 (E3c-M ince ayarı) → 15C sevk kuralı → 15D ölçüm →
  15E belgeler → 16 motor satırı → 17 bench (laptop prizde) → 22 elle ekran kontrolü (≈ 1 saat) → 23 dondur,
  test bir kez, rapor, `v0.3.0`.

Kaynak: `docs/plans/F4.md`, `docs/experiments.md` (son girdiler), ADR 0006.

📝 **Deftere:** 5 faz. Faz 2 olumsuz sonuç ama **dürüstçe raporlandı** → mülakat hikâyesi. Faz 4'te model
M3OT'la ince ayar bekliyor (E3c-M).

❓ **Kendini sına:** Faz 2 neden "başarısız" sayıldı ve bu neden projeye zarar değil? (Cevap §10.)

---

## 4. Nesne tespiti temelleri (ML 101)

### 4.1 Tespit ne döndürür?
Her nesne için **kutu** `(x1, y1, x2, y2)`, **sınıf** ve **güven skoru** (0-1). Sınıflandırma "resimde araba
var" der; tespit "şu 12 yerde araba var" der.

### 4.2 IoU (Intersection over Union)
`IoU = kesişim alanı / birleşim alanı` (0-1). Tahmin ile gerçek kutu ne kadar örtüşüyor. Projede her yerde:
eşleştirme (IoU ≥ 0,5 doğru), takip, WBF kümeleme, eşlik testleri.

### 4.3 Precision, recall, AP, mAP
- **Precision:** bulduklarımın ne kadarı doğru? `TP / (TP + FP)`
- **Recall:** gerçekte olanların ne kadarını buldum? `TP / (TP + FN)`
- Skor eşiğini yukarıdan aşağı indirdikçe bir **precision-recall eğrisi** çıkar. Eğrinin altındaki alan **AP**.
- **mAP50:** IoU ≥ 0,5'te AP, 5 sınıfın ortalaması. **Projenin ana metriği.** (mAP50-95: IoU 0,5→0,95
  ortalaması, kutu hassasiyetini daha sıkı ölçer.)
- AP hesaplanırken eşik çok düşük tutulur (projede `conf=0.001`); yoksa eğrinin kuyruğu kırpılır.
- Proje kuralı: pycocotools'ta `maxDets ≥ 1000` (bir karede çok araç olabilir; varsayılan 100 sonucu düşürür).

### 4.4 NMS
Aynı nesne için çıkan üst üste kutulardan en iyisini tutup diğerlerini atmak. **YOLO26 NMS'siz (uçtan uca):**
çıktısı doğrudan `[1, 300, 6]` (300 kutu × `x1, y1, x2, y2, skor, sınıf`). Bunun bedeli: görüntü başına en
fazla 300 kutu.

### 4.5 YOLO
"You Only Look Once": görüntüye tek geçişte bakıp bütün kutuları tahmin eden gerçek zamanlı model ailesi.
```
Görüntü → Backbone (özellik çıkarır) → Neck (ölçekleri birleştirir) → Head (kutu + sınıf + skor)
```
Boyutlar `n, s, m, l, x`. Projede **`s`** (small). Önceden COCO'da eğitilmiş ağırlıklardan **ince ayar**
(transfer öğrenme).

### 4.6 Eğitimle ilgili kavramlar
| Kavram | Anlamı | Projede |
|---|---|---|
| Epoch | Eğitim verisinin bir kez tamamen geçilmesi | 100, `patience 20` (erken durdurma) |
| Batch | Bir adımda işlenen görüntü sayısı | 16 (T4) |
| imgsz | Model girdisi | 640 |
| Seed | Rastgeleliği sabitleyen sayı | 42; **tek seed** (sınırlama) |
| Augmentation | Eğitimde görüntüyü çevirme, mozaik vb. | Ultralytics otomatik; 4 kanalda renk (HSV) kapalı |
| `close_mosaic` | Son N epoch mozaiksiz | 10 |
| Letterbox | Oranı bozmadan 640×640'a sığdırıp kenarı gri (114) doldurmak | 640×512 → üst/alt 64 px gri |

### 4.7 train / val / test
- **train:** model öğrenir. **val:** her seçim (eşik, parametre, hangi model) burada yapılır.
  **test:** yalnız **en sonda, bir kez**. Teste bakıp seçim yaparsan test sayısı iyimser olur (sızıntı).
- DroneVehicle: **17.954 / 1.469 / 8.980** görüntü çifti (`docs/data-report.md`).
- Projede test okuyan her komut `--i-know-this-is-test` bayrağı ister: yanlışlıkla teste bakılmasın diye.

📝 **Deftere:**
- `IoU = kesişim / birleşim`; mAP50 = IoU ≥ 0,5'te AP'lerin sınıf ortalaması.
- YOLO26 NMS'siz, çıktı `[1, 300, 6]`.
- Seçim val'de, test **bir kez**. Neden: test sayısının tarafsız kalması.

❓ **Kendini sına:** Precision yüksek recall düşük bir model nasıl davranır? AP hesaplarken skor eşiği neden
0,001? Letterbox 640×512 görüntüye ne yapar?

---

## 5. Veri

### 5.1 DroneVehicle
- RGB-IR görüntü çiftleri, 5 araç sınıfı, 3 aydınlatma (gündüz / gece / karanlık gece).
- Etiketler **döndürülmüş kutu (OBB)**; biz **yatay kutuya (HBB)** çevirdik: 4 köşeyi saran en küçük eksen
  hizalı dikdörtgen (ADR 0002). Neden: takipçiler, pycocotools, TrackEval, C++ tarafı hep HBB.
- Görüntülerin çevresinde 100 px beyaz kenarlık var → kırpıldı, **640×512**.
- **İki ayrı etiket seti** (RGB ve IR ayrı etiketlenmiş). IR'de daha çok nesne var (car: RGB 389.779, IR
  428.086). **Ana tablo IR etiketleriyle** (ADR 0003): gece RGB'de görünmeyen araçlar IR setinde var.
  RGB etiketleriyle sonuç ayrıca raporlanır (E5).
- **Aydınlatma etiketi** resmi dosyada yoktu. RGB'nin HSV `V` ortalamasından türetildi (ADR 0005):
  karanlık eşiği 48,66 (resmi orana uydurma), gündüz eşiği 114,14 (160 elle etiketli zor örnekte %95
  doğruluk). "Türetilmiş etiket, resmi değil" diye raporlanır.

### 5.2 4 kanallı TIFF tuzağı (ADR 0004) — güzel bir mülakat hikâyesi
Erken füzyonda R, G, B, T tek görüntü. Tek sayfalı 4 kanallı TIFF'te **OpenCV 4. kanalı "alfa" sanıyor**,
RGB'yi T ile çarpıyor ve sırayı BGR yapıyor (yazılan 10,20,30,40 → okunan 5,3,2,40). **Hata vermiyor, sessizce
bozuyor.** Çözüm: her kanal **ayrı TIFF sayfası**, sıra R,G,B,T. Gidiş-dönüş testi bu bozulmayı yakalıyor.
Ders: veri hattına "çalışıyor" demeden önce piksel düzeyinde test.

### 5.3 Depolama
4 kanallı TIFF'ler 27 GB tuttu. İşlenmiş veri bir kez Colab'da üretilip **private Kaggle veri setine**
yüklendi (`ouzhantarhan/baykus-dv-rgbt`, 9,84 GB). TIFF'ler yüklenmedi; gerekince `rgb` + `ir`'den yeniden
üretiliyor (A100'de 188 sn).

### 5.4 M3OT
- İkinci veri seti (Scientific Data 2025, **CC BY 4.0**): drone videoları (100-120 m), RGB-IR, **MOT
  biçimi** (`frame, track_id, x, y, w, h, flag, class, visibility`), tek sınıf `vehicle`.
- Süzgeç: `flag == 1`, `class == 1`, `visibility > 0,25`.
- Bölümler: val `1_08`, `2_08` (2 × 600 kare, geliştirme), test `1_03`, `2_03` (2 × 480 kare), train 14 dizi.
- Kullanımı: (1) **veri setleri arası test** (hiç görülmemiş veride ne oluyor), (2) Faz 3-4'ün **video, takip
  ve demo** verisi, (3) Faz 4'te E3c-M ince ayarı için train.
- **Bulgu:** M3OT'ta RGB ve IR kareleri **aynı hizada değil** (görüş alanı, ölçek, konum farklı; etiket IoU
  ortancası 0,35). Bu yüzden Faz 4'te ego-hareket **yalnız IR**.

📝 **Deftere:**
- OBB → HBB; 100 px kenarlık kırpma → 640×512.
- Ana etiket seti **IR** (gece görünmeyen araçlar yüzünden).
- 4 kanal = 4 ayrı TIFF sayfası (OpenCV alfa tuzağı).
- M3OT: CC BY 4.0, MOT biçimi, RGB-IR hizasız.

❓ **Kendini sına:** Neden RGB etiketleriyle değil IR etiketleriyle ölçüyoruz? Aydınlatma etiketinin
zayıflığı ne? Sessiz veri bozulmasını nasıl yakalarsın?

---

## 6. Modeller ve füzyon (Faz 1)

Hepsi **YOLO26s**, Ultralytics (ADR 0001), Colab'da, tek seed.

| ID | Ne | Fikir |
|---|---|---|
| E1 | Sadece RGB | Gece çöker mi? |
| E2 | Sadece IR | Termal tek başına ne kadar iyi? |
| E3 | **Erken füzyon**: R,G,B,T → 4 kanal, tek model | En basit, en hızlı füzyon |
| **E3c** | E3 + **modality dropout** | Bir kamera kaybına dayanıklılık |
| E4 | **Geç füzyon**: E1 + E2 iki model, kutular **WBF** ile | Sağlam ama 2 kat hesap |
| E4b | E4 + aydınlatmaya göre WBF ağırlığı | Gece RGB'ye daha az güven |

### 6.1 Füzyon türleri
- **Erken (early):** kanalları girişte birleştir. Tek model, hızlı. Hizasızlığa duyarlı; bir kamera giderse
  çöker (E3, IR kapanınca 12,8).
- **Orta (mid):** iki omurga, özellik düzeyinde birleşim. Genelde en doğru ama karmaşık, export zor (E10,
  yapılmadı).
- **Geç (late):** iki ayrı model, çıktı kutuları birleştirilir. Bir kamera giderse diğeri çalışır.

### 6.2 İlk katman başlatma
COCO ağırlıkları 3 kanallı. 4 kanallı modelde Ultralytics kanal 0-2'ye COCO'nun RGB ağırlıklarını kopyalar,
kanal 3'ü (T) rastgele bırakır. E3'te T kanalı **RGB ağırlıklarının ortalamasıyla** başlatıldı (`rgb_mean`,
eğitim başında bir callback ile).

### 6.3 Modality dropout (E3c)
Eğitimde her örnekte **%15 olasılıkla RGB kanalları**, **%15 olasılıkla T kanalı** sıfırlanır (ikisi birden
asla). Model tek kameraya yaslanmayı bırakır. Sonuç: IR kapanınca E3 **12,8** → E3c **72,0** (val), temiz
veride bedeli yok denecek kadar az (85,7 → 85,1). Benzetme: bir gözünü kapatıp da yürümeyi öğrenmek.

### 6.4 WBF (Weighted Boxes Fusion)
İki modelin kutularını birleştirme yöntemi (NMS gibi atmak yerine **ortalar**):
1. Bütün kutular skora göre sıralanır.
2. IoU'su eşikten (`iou_thr`) büyük kutular aynı **kümeye** girer.
3. Kümenin kutusu = koordinatların **skorla ağırlıklı ortalaması**; skor = ortalama skor (az modelin
   desteklediği kümede düşürülür).
4. Model ağırlıkları: RGB 1, IR `ir_weight`.

Val'de seçilen (E4): `iou_thr 0,7`, `ir_weight 3,0`. E4b: aydınlatmaya göre `ir_weight` gündüz 1, gece 2,
karanlık gece 3 (`results/fusion/*_val_selection.json`). Araç: `ensemble-boxes`.

📝 **Deftere:**
- Erken füzyon = 4 kanal tek model; geç füzyon = 2 model + WBF.
- Modality dropout: eğitimde rastgele kamera karartma (%15 / %15) → kamera kaybında 12,8 → 72,0.
- WBF kutuları atmaz, **ağırlıklı ortalar**.

❓ **Kendini sına:** Erken füzyon neden hizasızlığa duyarlı? Modality dropout neden temiz veride az
kaybettirdi? WBF ile NMS'in farkı ne?

---

## 7. Değerlendirme: dürüst ölçüm nasıl yapılır?

### 7.1 Araçlar
- **pycocotools** (COCO standardı) ile mAP; Ultralytics `val` ile çapraz kontrol (`docs/eval-crosscheck.md`).
- **Kırılımlar:** aydınlatma (gündüz / gece / karanlık gece), boyut (COCO küçük < 32², orta < 96², büyük),
  sınıf.

### 7.2 Bootstrap güven aralığı (GA)
"Bu sayı ne kadar oynayabilir?" Test görüntülerinden **yerine koyarak** aynı sayıda örnek çek, mAP'yi yeniden
hesapla; 1000 kez (seed 42). Sonuçların %2,5 ve %97,5 yüzdelikleri = %95 GA. Tablolarda `[alt; üst]`.
- **Eşleştirilmiş bootstrap:** iki modeli **aynı** örneklemde ölç, farkın dağılımına bak. Farkın GA'sı 0'ı
  içermiyorsa "anlamlı", içeriyorsa "fark ölçüm belirsizliği içinde".
- **Sınırlama:** tek seed → seed varyansı GA'ya dahil değil. M3OT ardışık video kareleri → GA'lar iyimser
  (kareler bağımsız değil).

### 7.3 Projenin dürüstlük kuralları (ezberle, mülakatta söyle)
1. **Kabul ölçütü ölçümden önce yazılır** (sonuca göre eşik kaydırılmaz).
2. **Seçimler val'de, test bir kez** ve en sonda.
3. **Kanıtsız sayı yok:** her sayı `results/**/*.json`'da; JSON'da config, git commit, seed, girdi sha256.
   Ölçülmemiş değer `null`.
4. **Olumsuz sonuç da raporlanır** (Faz 2).
5. Belgeler (`docs/results*.md`, `system.md`, `health.md`) **JSON'lardan otomatik üretilir**, elle
   düzenlenmez.
6. **Anti-hedefler:** "çalışsa bile başarısız sayılır" listesi (ör. ürünün kendi tahminiyle üretilmiş etiketle
   ürünü ölçmek = döngüsel ölçüm).

📝 **Deftere:** Bootstrap = yerine koyarak yeniden örnekleme, 1000 kez, %2,5-%97,5. Eşleştirilmiş bootstrap
fark için. Kurallar: önce ölçüt, val'de seçim, test bir kez, kanıtsız sayı yok.

❓ **Kendini sına:** İki modelin GA'ları örtüşüyorsa ne dersin? Neden "en iyi"yi val'de seçip testte
raporluyoruz? Ardışık video karelerinde bootstrap neden iyimser?

---

## 8. Sağlamlık paketi (E6): kamera bozulunca ne oluyor?

Görüntüler bilerek bozulup doğruluk ölçülür (`baykus-robust`, 16 koşul):

| Bozulma | Şiddetler | Kamera |
|---|---|---|
| Karartma | ×0,5, ×0,25, ×0,1 | RGB |
| Gauss gürültüsü | σ 5, 10, 20 | IR |
| Bulanıklık | çekirdek 3, 7, 11 | ikisi |
| Hizasızlık (IR kaydırma) | 5, 10, 20, 40 px | IR |
| Kamera kaybı | RGB = 0 ya da IR = 0 | biri |

**Sonuç (val, mAP50; README):**

| Koşul | E3 | E3c | E4 |
|---|---|---|---|
| Temiz | 85,7 | 85,1 | 84,8 |
| RGB kapalı | 71,8 | 82,9 | 83,9 |
| IR kapalı | **12,8** | 72,0 | 76,1 |
| Bulanıklık 11 | 64,1 | 65,2 | **80,9** |
| IR 20 px kayık | 25,8 | 26,6 | **48,5** |

**Yorum:** temiz veride füzyon modelleri ayırt edilemiyor; fark **bozulmada** çıkıyor. E3c kamera kaybını
çözüyor ama **kaymaya ve bulanıklığa** karşı E4 kadar dayanıklı değil → Faz 2'nin çıkış noktası.

📝 **Deftere:** 5 bozulma türü. E3c'nin zayıf noktası **IR kayması** (20 px'de 26,6). E4 sağlam ama yavaş.

---

## 9. Edge: modeli hızlandırmak ve sevk kararı

### 9.1 Kavramlar
- **ONNX:** modelin PyTorch'tan bağımsız dosya biçimi. **ONNX Runtime (ORT)** ile her yerde (C++ dahil) koşar.
- **FP32 / FP16 / INT8:** sayı hassasiyeti. Küçüldükçe dosya küçülür, *genelde* hız artar, doğruluk düşebilir.
- **INT8 nicemleme:** ağırlık ve aktivasyonlar 8 bit tam sayıya. **Statik nicemleme** = eğitim verisinden
  birkaç yüz görüntüyle aktivasyon aralıkları **kalibre** edilir (ORT `quantize_static`, QDQ düğümleri).
- **Head FP32:** tespit başlığı nicemlenirse doğruluk 3-8 puan düşüyor; başlık FP32 kalınca < 1 puan.
- **TensorRT:** NVIDIA GPU hızlandırıcısı (Colab T4'te ölçüldü; sevk edilmedi).
- **p50 / p95:** gecikmenin ortancası / karelerin %95'inin altında kaldığı süre. **p95 takılmaları gösterir;**
  ortalama FPS gizler.
- **Bench protokolü:** batch 1, 640×640, **50 ısınma + 500 ölçüm**; süre = ön işleme + çıkarım + son işleme;
  laptop prizde; CPU, ORT sürümü, güç durumu JSON'a yazılır.

### 9.2 Sevk kuralı (ADR 0007) — ölçümden **önce** yazıldı
Laptop CPU (Ryzen 7 8845HS), **p95 ≤ 100 ms** ve val mAP50 kaybı **≤ 1,0 puan**. Aday sırası **E4 → E3c**.

| Aday | En hızlı uygun varyant | p95 | Geçti mi |
|---|---|---|---|
| E4 | ONNX FP32 | 131,6 ms | hayır |
| **E3c** | **ONNX FP32** | **59,9 ms** | **evet → sevk** |

**E3c · ONNX FP32:** p50 46 ms, p95 60 ms, 21 FPS, 38 MB, test mAP50 81,4 [80,4; 82,5].

### 9.3 Şaşırtıcı bulgular (mülakat malzemesi)
- **INT8 bu işlemcide yavaşlattı** (E3c p95 59,9 → 117,8 ms), VNNI desteğine rağmen. Olası neden (ölçülmedi,
  tahmin): yalnız konvolüsyonlar nicemlendiği için katmanlar arasında sürekli INT8 ↔ FP32 dönüşümü.
- **ONNX FP32, PyTorch'tan ≈ 1 puan düşük** (test 81,4 / 82,5). Olası nedenler (tam ölçülmedi): 300 kutu
  sınırı, kare 640×640 girdi. Açık iş.
- GPU'da (T4, TensorRT) ikisi de gerçek zamanlı: E3c FP16 ≈ 99 FPS, E4 INT8 ≈ 37 FPS.

📝 **Deftere:** Hedef p95 ≤ 100 ms, ≤ 1 puan kayıp, ölçümden önce. E4 elendi (131,6 ms), **E3c ONNX FP32
sevk** (59,9 ms). INT8 CPU'da hızlandırmadı.

❓ **Kendini sına:** Neden ortalama değil p95? Statik nicemlemede kalibrasyon verisi ne işe yarar? Kuralı
ölçümden önce yazmanın faydası ne?

---

## 10. Faz 2: kamera bozulunca fark etmek (olumsuz sonuç)

### 10.1 Soru
E3c, IR **bozuk veri** verirken çöküyor (IR 40 px kayınca val 9,0), ama IR'yi **hiç vermemek** 72,0 alıyor.
Yeniden eğitmeden, her karede kameraların sağlığını ölçüp bozuk olanı modele **vermeyen** bir **kapı** bu
farkı kapatır mı?

### 10.2 Sağlık ölçüleri (kural tabanlı, eğitim yok; `baykus.health.features`)
| Durum | Ölçü | Sezgi |
|---|---|---|
| Kamera kayıp | piksel std | Siyah ekranın std'si ≈ 0 |
| IR gürültülü | **Immerkær gürültü kestirimi** | 3×3 Laplace benzeri çekirdekle konvolüsyon; görüntü yapısını iptal edip gürültüyü bırakır |
| Bulanık | **Laplace varyansı** | Keskin kenar → yüksek varyans; bulanık → düşük |
| Karanlık | HSV `V` ortalaması | ADR 0005 eşiği |
| IR hizasız | **Faz korelasyonu** (kenar haritalarında) | FFT ile iki görüntü arasındaki kaymayı bulur; tepe yüksekliği = **yanıt** (güven) |

**Eşik seçimi:** train'den 500 görüntü × bütün bozulma koşulları, **Youden J** (`TPR − FPR` en büyük olan
eşik).

**Kapı:** bozuk bulunan kanal ön işlemeden önce **sıfırlanır** (model E3c olduğu için kamera kaybına zaten
dayanıklı). Öncelik: kayıp > gürültülü > hizasız; iki kanal birden kapatılmaz.

### 10.3 Sonuç
| mAP50 | E3c | E3c + kapı | Fark [%95 GA] |
|---|---|---|---|
| Bozuk IR, val | 29,7 | 70,4 | **+40,7** [+38,3; +43,1] |
| Temiz val | 85,1 | 82,0 | **−3,1** [−4,4; −1,8] |

Kapı bozukta çok işe yarıyor ama **temiz karelerin %4,1'inde yanlış alarm** veriyor → temiz veride 3 puan
kayıp → önceden yazılan kurala göre **başarısız**.

**Neden bulundu:** yanlış alarmların hepsi "IR hizasız", %80'i **karanlık gece**. RGB karanlıkken kenar yok,
faz korelasyonunun yanıtı çok düşük (0,05; normalde 0,56), kayma kestirimi anlamsız bir sayı buluyor.

### 10.4 Kalibrasyon
- **Kalibrasyon:** model "%80 eminim" dediğinde gerçekten %80 mi doğru? **ECE** (Expected Calibration Error)
  = skorları 15 kutuya böl, her kutuda `|doğruluk − ortalama skor|`, kutu büyüklüğüyle ağırlıklı ortala.
- **İzotonik regresyon:** skoru monoton (sırayı bozmayan) bir basamak fonksiyonuyla düzeltir.
- Sonuç: ECE **0,080 → 0,007**. Ama kalibre skorlar 80 farklı değere indiği için **eşit skorlar** sıralamayı
  bozup mAP'yi 0,11 puan düşürdü (sınır 0,1) → ❌. Çözüm (Faz 3): eşitlikte ham skorla sırala → kayıp 0.

### 10.5 Faz 3'teki düzeltme
- **Kare kuralı:** hizasızlık alarmı için faz korelasyonu **yanıtı ≥ `r_min`** şartı (train temiz karelerinde
  alarm ≤ %2 olan en küçük değer).
- **Zaman kuralı (videoda):** gerçek kayma kareden kareye aynı yönde kalır, anlamsız kestirim zıplar. Son 5
  karede ≥ 3 hizasız kare **ve** bunların ≥ 3'ünün kayma vektörü ortancaya ≤ 3 px yakın → alarm.
- Sonuç (val, keşif): temiz yanlış alarm %0,7, temiz fark −0,2 puan, misalign-20'de +35,7 puan. Ürün
  varsayılanı yine de **`gate=advise`** (yalnız uyarır, piksel kesmez); `auto` isteğe bağlı.
- **M3OT'ta sınır:** iki kamera arasında doğal kayma var; ürün kuralı temizde yanlış alarm vermiyor ama gerçek
  20 px kaymayı da yakalamıyor (`docs/system.md` Sınırlamalar).

📝 **Deftere:**
- Kapı fikri: bozuk kamerayı sıfırla, E3c zaten tek kamerayla çalışabiliyor.
- Başarısızlık nedeni: **karanlıkta faz korelasyonu güvenilmez** → yanlış "hizasız" alarmı.
- Düzeltme: **yanıt eşiği + zaman kuralı**. Varsayılan `advise`.
- ECE 0,080 → 0,007; eşit skor tuzağı.

❓ **Kendini sına:** Youden J nedir? Neden yeniden eğitim bu sorunu çözmezdi? Kalibrasyon mAP'yi neden
değiştirdi? Olumsuz sonucu mülakatta nasıl anlatırsın?

---

## 11. Faz 3: sistem — C++ motor, takip, ekran, kapalı ağ

### 11.1 C++ motor (`engine/`, C++17)
Kütüphaneler: **ONNX Runtime 1.30.0** (çıkarım), **OpenCV 5.0.0** (görüntü; kaynaktan derlenmiş), **Eigen**
(Kalman), **cpp-httplib** (HTTP), **nlohmann/json**, **GoogleTest**. Derleme **yalnız Docker'da** (Windows'a
C++ araç zinciri kurulmadı).

İstek başına aşamalar (`engine/src/runtime.cpp`) ve Faz 3 gecikmesi (p50 / p95 ms, `docs/system.md`):

| Aşama | Ne yapar | p50 / p95 |
|---|---|---|
| decode | JPG/PNG çözme | 6,5 / 7,7 |
| health | sağlık ölçüleri, kapı kararı | 8,7 / 10,8 |
| pre | letterbox, normalize, HWC → CHW | 3,2 / 4,4 |
| infer | ONNX Runtime | 45,9 / 62,1 |
| post | çıktı çözümleme, kalibre skor | 0,0 / 0,0 |
| track | ByteTrack | 0,1 / 0,2 |
| **total** | | **64,8 / 82,2** (eşik 100) ✅ |

Ders: zamanın %70'i çıkarımda; takip neredeyse bedava.

Önemli modüller (`engine/include/baykus/`): `preprocess`, `detector`, `health`, `temporal` (zaman kuralı),
`calibration`, `kalman`, `lsap` (Macar algoritması), `tracker`, `runtime`, `server`, `sessions`; Faz 4'te
`ego`, `reloc`, `motion`, `frozen`, `zones`, `polygon`, `event_log`, `track_summary`, `tiles`.

### 11.2 Eşlik (parity) testi — projenin omurgası
1. Python, fikstür görüntülerde aynı ONNX modeliyle beklenen çıktıyı üretir (`fixtures/expected/*.json`).
2. C++ aynı girdilerle çalışır, karşılaştırılır.
3. Faz 3 sonucu: 17 fikstürde ham ORT çıktısında en büyük fark **0**; letterbox tensörlerinin sha256'ları
   **aynı**; sağlık kararları aynı; takip **2 × 600 karede fark 0**.
4. CI'da her commit'te; kabul testleri **atlanamaz** (`BAYKUS_REQUIRE_FIXTURES=1`).
5. **Sınır kuralı:** fikstürdeki ölçüler eşiklere çok yakın olmamalı (yoksa küçük kayan nokta farkı kararı
   değiştirir, test rastgele kırılır).

Neden önemli: "C++'a çevirdim, aynı çalışıyor" iddiasının kanıtı. Savunma sanayiinde (C++ + Python birlikte)
tam aranan şey.

### 11.3 Takip: ByteTrack
- **Kalman filtresi:** her izin durumu `(x1, y1, x2, y2, vx1, vy1, vx2, vy2)`, **sabit hız modeli**. Her
  karede **tahmin** (predict: bir sonraki konum), tespit gelince **güncelleme** (update: tahmin ile ölçümü
  belirsizliklerine göre karıştır).
- **Macar algoritması** (doğrusal atama, LSAP): izler × tespitler maliyet matrisinde (1 − IoU) toplam
  maliyeti en küçük eşleşme. SciPy'ın `linear_sum_assignment`'ının C++ uyarlaması.
- **İki aşamalı eşleşme** (ByteTrack'in fikri): önce **yüksek skorlu** tespitler izlerle eşleşir; artan
  izler **düşük skorlu** tespitlerle de denenir (kısmen örtülmüş araçlar kaybolmasın).
- İz yaşam döngüsü: yeni iz `track_activation_threshold` üstündeki eşleşmemiş tespitten, ardışık 2 karede
  doğrulanınca açılır; eşleşmeyen iz **30 kare** bekler (`lost_track_buffer`), sonra silinir.
- Referans: `trackers==2.6.1` (roboflow). Parametreler `configs/engine/tracker.json`. Faz 4'te val'de
  seçildi: etkinleşme **0,3**, yüksek eşik **0,3** (eskiden 0,7 / 0,6).

### 11.4 Takip metrikleri
| Metrik | Ne ölçer |
|---|---|
| **MOTA** | `1 − (FN + FP + kimlik değişimi) / GT`; tespit ağırlıklı |
| **IDF1** | Kimliklerin ne kadar doğru korunduğu (kimlik düzeyinde F1) |
| **HOTA** | Tespit doğruluğu (DetA) ile ilişkilendirme doğruluğunun (AssA) geometrik ortalaması, birçok IoU eşiğinde; ikisini dengeli ölçer |

**Oracle satırı (ADR 0008):** takipçiye modelin değil **gerçek (GT) kutular** verilir. Böylece takipçi
hatası ile dedektör hatası **ayrılır**.

| M3OT test | HOTA | MOTA | IDF1 |
|---|---|---|---|
| oracle | 0,949 | 0,981 | 0,941 |
| motor | 0,489 | 0,547 | 0,677 |

Yorum: takipçi doğru kutularla neredeyse kusursuz; motordaki kayıp **tespitten** geliyor. Val (yoğun trafik,
kare başına ≈ 11 araç, ortanca kutu 13 px) çok daha zor: motor HOTA 0,220.

### 11.5 HTTP API (özet)
`GET /health`, `GET /v1/model`, `POST /v1/detect` (multipart RGB + IR; sorgu `conf`, `track`, `session`,
`gate=off|advise|auto`), `POST /v1/session/reset`. Faz 4 ekledi: `/v1/session/zones`, `/events`,
`/alarms/ack`, `/track`. Hatalar: 400 (eksik parça, bozuk görüntü), 404, 413 (çok büyük), 500; **motor tek
hatalı istekle çökmez** (1000 rastgele istekle sınandı).

### 11.6 Operatör ekranı (`ui/`, Next.js 16 + TypeScript)
- Oynatıcı (RGB / IR / yan yana), kutular ve iz numaraları, sağlık paneli, alarmlar (son 5 karenin 3'ü),
  **arıza simülasyonu** (karart, gürültü, kaydır, bulanıklaştır, kamera kapat), kapı rozeti, gecikme grafiği.
- **Tarayıcı motoru hiç görmez:** istekler Next sunucusunun API rotalarından motora iletilir.
- Font / CDN yok, her şey yerel. Testler: **Vitest** (saf fonksiyonlar `ui/lib/`), **Playwright** (uçtan uca).

### 11.7 Kapalı ağ ve Docker
- `docker compose up`: iki servis. Motor yalnız iç ağda (`internal: true`), dışarı çıkamaz; ekran
  `127.0.0.1:3000`.
- `compose.offline.yml` bütün ağları `internal` yapar (kanıt katmanı).
- `scripts/prepare.sh --bundle out/`: imajlar + modeller + demo verisi → internetsiz bilgisayara taşı,
  `docker load` + `docker compose up`.
- Kanıt: tarayıcıdan dış istek 0, konteynerlerden dış bağlantı başarısız, Wi-Fi kapalı elle demo
  (`results/system/offline.json`).

📝 **Deftere:**
- Motor aşamaları: decode → health → gate → pre → infer → post → track (Faz 4: + ego, motion, zones...).
- Eşlik = C++ çıktısı Python referansıyla birebir; CI'da zorunlu.
- ByteTrack = Kalman (sabit hız) + Macar + iki aşamalı eşleşme.
- **Oracle satırı** takipçi ile dedektörü ayırır (test HOTA 0,949 / 0,489).

❓ **Kendini sına:** Kalman filtresinde predict ve update ne yapar? ByteTrack düşük skorlu tespitleri neden
kullanır? HOTA'yı MOTA'dan ayıran ne? Tarayıcının motora doğrudan bağlanmaması neden iyi?

---

## 12. Faz 4: hareketli hedef ve operasyonel ürün

### 12.1 Sorun
Drone hareket ettiği için görüntüde **park etmiş araç da kayıyor**. Faz 3 her şeyi görüntü koordinatında
yapıyordu; "hangisi gerçekten hareket ediyor, kim kavşağa girdi" sorusunu cevaplayamıyordu.

### 12.2 Ego-hareket: kameranın kendi hareketini kestirmek
Fikir: araçlar dışındaki **arka plan** sabit; arka planın görüntüde nasıl kaydığı = kameranın hareketi.
Tarif BoT-SORT'un kamera hareketi telafisindeki (GMC) yaygın "seyrek optik akış + benzerlik dönüşümü +
RANSAC" yöntemi; ayırt edici olan **dürüst ölçüm**.

Adımlar (yarı çözünürlük gri IR, 320×256; parametreler `configs/engine/motion.json`):
1. **Maske:** tespit kutuları (araçlar hareket edebilir) ve kenar bandı dışarıda.
2. **Köşe bul** (`goodFeaturesToTrack`, en çok 300): takip etmesi kolay noktalar.
3. **Optik akış** (`calcOpticalFlowPyrLK`, Lucas-Kanade, piramitli): her köşe yeni karede nereye gitti?
   **İleri-geri denetimi:** noktayı geri takip et, ≤ 1 px'te başlangıca dönmüyorsa at.
4. **RANSAC + benzerlik dönüşümü** (`estimateAffinePartial2D`, 4 serbestlik: öteleme x/y, dönme, tek ölçek):
   rastgele küçük nokta kümeleriyle model kur, en çok **iç noktayı** (tutarlı nokta) açıklayanı seç, aykırıları
   (hareketli araçlar, hatalı eşleşmeler) at. `ok` = iç nokta ≥ 30 ve oranı ≥ 0,4.
5. **Anahtar kare:** her kare **ardışık kareye değil anahtar kareye** kaydedilir. Neden: ardışık dönüşümleri
   çarpmak hatayı biriktirir (**zincir kayması**): val keşfinde 600 karede ölçek **4,3×** şişti, gerçek
   büyüme ≈ 1,3-1,6×. Yeni anahtar kare: iç nokta azalınca, 30 kare geçince ya da kayma > 40 px olunca.
6. **Harita dönüşümü:** `M_t = M_k · A_full⁻¹` (kare → harita). Yarı çözünürlükten tam çözünürlüğe
   `A_full = S·A·S⁻¹`, `S = [[2,0,0,5],[0,2,0,5]]` (piksel merkezi kaymasını doğru taşımak için).

Neden homografi değil? Dönme ≈ 0,02°/kare, ölçek ≈ %0,2/kare; perspektif küçük. Homografi (8 serbestlik)
daha kırılgan. Modelin hatası ayrıca **ölçüldü** (harita kayması).

### 12.3 Harita nesli ve yeniden konumlama (harita belleği)
- **Nesil (`map_gen`):** aynı harita koordinatında geçen kareler. Kısa kayıpta (≤ 10 kare) son anahtar kareye
  yeniden bağlanır; uzun kayıp / kanal değişimi / atlama → **geçici nesil**.
- **Yeniden konumlama:** her anahtar karenin **ORB** noktaları ve tanımlayıcıları bir kütüphanede (en çok 300).
  Kopmadan sonra yeni kare kütüphanedeki adaylarla eşlenir (ORB: FAST köşeleri + ikili BRIEF tanımlayıcı;
  **Hamming** mesafesi, çapraz denetim, RANSAC). Başarılıysa geçici nesil eski nesle **birleştirilir**
  (`T = M' · M_prov⁻¹`); 30 karede başaramazsa kalıcı yeni nesil.
- **Güvenlik ilkesi (anti-hedef):** doğrulanmamış bağlamada bölgeler **askıya alınır**; asla yanlış yerde
  sessizce olay üretmez.
- Ölçüm (val): benzetilmiş atlamalarda 22'de 13 başarı, **yanlış bağlama 0**.

### 12.4 Hareket durumu: hareketli / duran
- Her iz için harita koordinatında merkez `c_t` ve kutu boyu `s_t = √(w·h) · ölçek`.
- **Hız:** `v = ‖c_t − c_{t−15}‖ / 15 / s_t` → birimi **"kutu boyu / kare"**. Kutu boyuna bölmek irtifa /
  yaklaşma değişimine karşı ölçek bağımsızlığı verir.
- **Histerezis:** `v ≤ 0,03` → `stationary`, `v ≥ 0,10` → `moving`, arada → **önceki karar**. Bu, eşik
  çevresinde kararın titremesini önler (termostat gibi). Yeni iz ilk 15 karede `unknown`.
- Eşikler (15; 0,03; 0,10) val keşfinde iki tepeli dağılımdan seçildi ve **ölçümden önce sabitlendi**.
- Gerçek metre / km/sa **yok**: irtifa ve kare hızı doğrulanmadı. Harita **göreli**; gerçek konum gibi
  sunulmaz.

### 12.5 Döngüsel olmayan ölçüm (Faz 4'ün en güçlü mülakat hikâyesi)
Sorun: veri setinde "hareketli/duran" etiketi yok. Etiketleri **ürünün kendi yöntemiyle** üretirsen ölçüm
döngüsel olur (ürünün hatası etikette de olur).

Çözüm: **ayrı bir referans etiketleyici**, bilerek farklı:

| | Ürün | Referans etiketleyici |
|---|---|---|
| Kayıt | anahtar kareye LK, benzerlik, yarı çözünürlük | doğrudan `t−K ↔ t+K`, **SIFT** + oran testi 0,75 + **homografi**, tam çözünürlük |
| Maske | model tespitleri | GT kutuları |
| Pencere | geçmiş (nedensel) | ortalanmış `[t−K, t+K]` |

Ve etiketleyicinin kendisi **kör insan örneklemiyle** sınandı: sen etiketi görmeden 80 rastgele örneğe
"hareketli / duran / emin değilim" dedin.
- 1. tur ❌ (%94,2 < %95): pencere sabit olduğu için park etmiş araç da kayıyor görünüyordu → **tek düzeltme
  hakkı sunuma** kullanıldı (pencere araca ortalandı), etiketleyici değişmedi, cevaplar değiştirilmedi.
- 2. tur ✅: uyum 72/73 = **%98,6** (Wilson %95 GA 0,926-0,998). Etiketleyici dondu (`motion-labels-v1`).
- **Wilson aralığı:** küçük örneklemde ve %0-100 uçlarına yakın oranlarda normal yaklaşımdan daha doğru GA.

**Oracle sonucu (val):** dengeli doğruluk **BA 0,949** (GA 0,925-0,967), R_mov 0,985, R_sta 0,913.
Dengeli doğruluk = `(R_mov + R_sta) / 2`: sınıflar dengesizken tek doğruluğun yanıltmasını önler.

### 12.6 Bölgeler, kurallar, alarmlar, olaylar (motorda)
Operatör haritada çizer; motor değerlendirir (tarayıcı kapalıyken de aynı olaylar):

| Tür | Ne yapar |
|---|---|
| `area` (alan) | giriş / çıkış / kayıp çıkış / durdu olayları, sayaçlar |
| `restricted` (yasak) | içeride iz varken `intrusion` alarmı |
| `dwell` (bekleme sınırı) | iz N kare duran kalırsa `dwell_exceeded` alarmı |
| `line` (sayım hattı) | yönlü geçiş sayımı (`a_to_b`, `b_to_a`) |

- **Titreme önleme:** giriş/çıkış için **3 ardışık gözlem**; ≤ 5 karelik boşluk tolere edilir.
- **Hat geçişi:** taraf = `sign((B − A) × (p − A))` (2B vektörel çarpım). Kararlı taraf değişince ve hareket
  doğru parçası AB'yi gerçekten kesiyorsa sayılır (hattın uzantısından dolanmak sayılmaz).
- **Alarm** onaylanır (`ack`) ama kendiliğinden kaybolmaz; temizlenince geçmişte görünür.
- **Güven kaydırıcısı** yalnız gösterimi süzer; olaylar süzgeçsiz bütün takip akışından (`tracks[]`).
- Olay günlüğü (en çok 20.000 kayıt), JSON / CSV dışa aktarma, harita PNG — hepsi internetsiz.
- Veri setinde bölge yok → doğruluk **C++ senaryo testleriyle** (Python referansı yok).

### 12.7 Donmuş kare
Kanal başına önceki kareyle ortalama mutlak fark `diff_mad ≤ 0,5` → donmuş kare. Kare bayrağı ego'ya anında
girer (donmuş IR "her şey duruyor" sanılmasın); alarm 5 karenin 3'ü kuralıyla.

### 12.8 M3OT araç kapsamı sorunu (Görev 15A-15E)
Kullanıcı gözlemi: karede 30+ araç görünüyor, motor 0-5 iz veriyor. Tanı (val):
- Model araçların en düşük eşikte bile yalnız **%40**'ını buluyor (ortanca kutu **13 px**; model DroneVehicle'da
  eğitildi). RGB'yi kapatmak sonucu değiştirmiyor → hizasızlık neden **değil**.
- Takipçi sağlam (oracle HOTA 0,97); ikinci sorun yüksek etkinleşme eşiği.

Üç çözüm:
- **A — Takipçi eşikleri:** val'de ızgara, kural önceden: HOTA en yüksek (0,005 tolerans içinde yüksek
  eşik tercih). Seçilen 0,3 / 0,3: HOTA **0,220 → 0,292**.
- **B — E3c-M:** E3c'yi M3OT train ile **ince ayar**. M3OT'ta araç türü yok → **öğretmen etiketi**
  (sözde etiket): E3c her izin en çok 10 karesinde aracı büyüterek (160×128 pencere → 640) sınıflar, çoğunluk oyu.
  Kör denetimde (100 iz) doğruluk **0,92** (GA 0,85-0,96) ≥ 0,80 → öğretmen etiketleri kullanılacak.
  İlk koşudaki hata (pencere köşesi iki kez çıkarılıyordu; 647 izin yalnız 2'si oy aldı) testlerin 64×64 karede
  yazılmasından kaçmıştı → 640×512 regresyon testi eklendi. **Ders:** birim testi gerçekçi boyutta yaz.
- **Sevk kuralı (önceden):** DroneVehicle val'de ≤ 1,0 puan kayıp ve hiçbir sınıf > 3,0 puan düşmez **ve**
  M3OT val AP50 artar. Tutmazsa E3c kalır.
- **C — Parçalı çıkarım (`tiles=4`):** kareyi 2×2 örtüşen parçaya böl, her parçayı 640'a büyüt (küçük araçlar
  büyür), kutuları geri taşı, sınıfsız NMS (IoU 0,5). Bedeli ≈ 4× çıkarım; varsayılan olması bench'te
  p95 ≤ 100 ms'ye bağlı (laptop CPU'da beklenen: hayır).

### 12.9 Ekran (Faz 4)
Göreli mozaik harita (kareler `frame_to_map` ile tuvale çizilir; kaydır, yakınlaştır, kamerayı izle), kutu
biçimi duruma göre (hareketli düz + "H", duran kesikli + "D"; yalnız renge dayanmaz), araç kartı, bölge
editörü, alarm paneli, zaman çizelgesi + olay listesi, anlık görüntü, dışa aktarma.

📝 **Deftere:**
- Ego-hareket: köşe → LK optik akış → RANSAC benzerlik → **anahtar kare** (zincir kayması 4,3×!).
- Hız birimi "kutu boyu / kare"; **histerezis** 0,03 / 0,10.
- Döngüsel ölçümden kaçış: **ayrı etiketleyici** (SIFT + homografi) + **kör insan örneklemi**.
- Yeniden konumlama ORB ile; doğrulanmazsa bölgeler **askıda**.
- E3c-M: öğretmen etiketi, kör denetim 0,92.

❓ **Kendini sına:** RANSAC hareketli araçları nasıl dışlar? Ardışık kare zinciri neden kayar? Histerezis
olmasa ne olurdu? Ürünün kendi tahminiyle etiket üretmek neden yanlış? Sayım hattında vektörel çarpım ne işe
yarar?

---

## 13. Mühendislik pratikleri (proje nasıl yönetiliyor)

| Pratik | Ne | Projede |
|---|---|---|
| **ADR** | Architecture Decision Record: karar, bağlam, alternatifler, sonuç | `docs/adr/0001-0008` |
| **Spike** | Zaman kutulu küçük deneme: "bu çalışır mı?" | `docs/spikes/` (4 kanallı TIFF, veri biçimi, Colab ortamı) |
| **Tasarım → plan → kod** | Her faz önce tasarım belgesi, sonra görev görev plan | `docs/design/`, `docs/plans/` |
| **KAPI** | Kullanıcı kararı / adımı gelmeden geçilmeyen nokta | Kör örneklem, Colab, push |
| **Kabul ölçütü önceden** | Başarı tanımı ölçümden önce yazılır | Her fazın Kabul 1-N'i |
| **Anti-hedef** | Çalışsa da başarısız sayılan şeyler | Faz 4 §3 |
| **TDD** | Önce test, sonra kod (deterministik kodda) | IoU, NMS, WBF, letterbox, geometri, bozulmalar |
| **Eşlik / fikstür** | Python referansı ↔ C++ | `fixtures/`, Release `fixtures-f4` (sha256 denetimli) |
| **Conventional Commits** | `feat(...)`, `fix(...)`, `docs(...)`, `test(...)`, `ci:` | Atomik commit, her biri tek mantıksal değişiklik |
| **Dal düzeni** | Her faz kendi dalında, sonunda `main`'e birleşir | `f4` → `v0.3.0` |
| **Deney günlüğü** | Her deneyin hipotezi, kurulumu, sonucu, kararı | `docs/experiments.md` |
| **Bağımsız inceleme** | Tasarımı başka bir modele (Codex) inceletmek, her bulguyu doğrulamak | Faz 4 tasarım §12, `docs/experiments.md` F3-F4 inceleme girdileri |

**Araç zinciri:**
- Python: **uv** (ortam ve kilit dosyası), **ruff** (lint + biçim), **pytest** (≈ 770 test fonksiyonu),
  pre-commit.
- C++: CMake, GoogleTest, clang-format (Google stili, 100 sütun), `-Wall -Wextra -Werror`, Docker çok aşamalı
  derleme.
- Ekran: Next.js, TypeScript, Vitest, Playwright, ESLint.
- CI: GitHub Actions (`.github/workflows/ci.yml`: python, ui, engine işleri; modeller ve fikstürler Release'ten
  sha256 denetimiyle).
- Deney: MLflow + `results/**/*.json`; Colab defterleri `notebooks/colab/00-10`.

**Lisans:** Ultralytics **AGPL-3.0** → repo da AGPL. Kapalı bir savunma ürününde Enterprise lisans gerekir;
izinli alternatifler RT-DETR, RF-DETR, D-FINE (Apache-2.0). Mülakatta bunu bildiğini göster (ADR 0001).

📝 **Deftere:** ADR, spike, KAPI, kabul ölçütü önceden, anti-hedef, TDD, eşlik. AGPL lisans riski.

---

## 14. Kod haritası (güncel)

```
C:\baykus
├─ python/src/baykus/          Python paketi; her alt klasör bir komut (baykus-<ad>)
│  ├─ data/        baykus-data    kırpma, OBB→HBB, 4 kanallı TIFF, COCO, aydınlatma, M3OT,
│  │                              öğretmen etiketi + kör denetim (teacher, teacher_audit), m3ot_merge
│  ├─ train/       baykus-train   Ultralytics eğitimi, modality dropout (multimodal.py), checkpoint
│  ├─ eval/        baykus-eval    pycocotools mAP, bootstrap GA, IoU, karşılaştırma
│  ├─ robustness/  baykus-robust  bozulma paketi
│  ├─ fusion/      baykus-fusion  WBF geç füzyon, val seçimi
│  ├─ edge/        baykus-edge    ONNX / INT8 / TensorRT export, doğruluk, eşlik
│  ├─ bench/       baykus-bench   p50 / p95
│  ├─ health/      baykus-health  sağlık ölçüleri, kapı, zaman kuralı, kalibrasyon, donmuş kare
│  ├─ engine_ref/  baykus-engine  C++ motorun Python referansı (pipeline, parçalı çıkarım, kapsam)
│  ├─ track/       baykus-track   ByteTrack koşucusu, eşlik, eşik taraması (sweep)
│  ├─ motion/      baykus-motion  ego, yeniden konumlama, hareket durumu, etiketleyici, kör örneklem, ölçüm
│  ├─ system/      baykus-system  docs/system.md raporu
│  ├─ report/      baykus-report  docs/results*.md tabloları ve grafikler
│  └─ io.py, classes.py, runinfo.py   ortak: görüntü I/O, sınıf sırası, sonuç JSON üst bilgisi
├─ python/tests/                pytest
├─ engine/                      C++ motor: include/baykus/*.hpp, src/*.cpp, apps/ (server, bench, detect,
│                               track, sequence, clip...), tests/ (GoogleTest), third_party/ (httplib, lsap)
├─ ui/                          Next.js: app/ (sayfa + API rotaları), components/ (Player, MapPanel,
│                               ZoneEditor, AlarmPanel, Timeline...), lib/ (saf mantık, Vitest ile test)
├─ configs/                     deney config'leri (e1…e3c_m3ot), fusion/, edge/, engine/ (tracker, motion,
│                               models, fixtures sha256)
├─ fixtures/                    eşlik fikstürleri (M3OT kareleri, sentetik bozulmalar, beklenen JSON'lar)
├─ notebooks/colab/             00_setup → 10_m3ot_train (sen çalıştırırsın)
├─ results/                     her sayının JSON kanıtı
├─ scripts/                     prepare.sh, offline kanıtları, m3ot_measure.sh, engine_test.sh
├─ docker/, docker-compose.yml, compose.offline.yml
└─ docs/                        spec, ADR, tasarımlar, planlar, raporlar, deney günlüğü, bu rehber
```

**Kodu okumaya başlamak için 6 dosya** (kısa ve öğretici):
1. `python/src/baykus/eval/iou.py` — IoU.
2. `python/src/baykus/fusion/wbf.py` — WBF.
3. `python/src/baykus/train/multimodal.py` — modality dropout.
4. `python/src/baykus/health/features.py` — sağlık ölçüleri, faz korelasyonu.
5. `engine/include/baykus/kalman.hpp` + `engine/src/tracker.cpp` — ByteTrack.
6. `python/src/baykus/motion/ego.py` — ego-hareket (§12.2'yi kodda gör).

---

## 15. Sayı kartı (ayrı bir defter sayfasına yaz)

| Ne | Sayı | Kaynak |
|---|---|---|
| DroneVehicle train / val / test | 17.954 / 1.469 / 8.980 çift | `docs/data-report.md` |
| Test mAP50 E1 / E2 / E3 / **E3c** / E4 | 73,3 / 81,0 / 82,3 / **82,5** / 82,0 | README |
| Karanlık gece E1 → E3c | 66,9 → 83,1 | README |
| IR kapalı (val) E3 → E3c | 12,8 → 72,0 | README, `docs/robustness.md` |
| Sevk: E3c ONNX FP32 laptop | p50 46 ms, p95 60 ms, 21 FPS, 38 MB, test 81,4 | README, ADR 0007 |
| E4 laptop p95 | 131,6 ms (elendi) | ADR 0007 |
| Faz 2 kapı bozuk / temiz (val) | +40,7 / −3,1 puan, yanlış alarm %4,1 | `docs/health.md` |
| Kalibrasyon ECE | 0,080 → 0,007 | `docs/health.md` |
| Motor gecikmesi (Docker) | p50 64,8, p95 82,2 ms | `docs/system.md` |
| M3OT test takip HOTA oracle / motor | 0,949 / 0,489 | ADR 0008 |
| M3OT test AP50 (motor) | 0,860 | `docs/system.md` |
| Zincir kayması (600 kare) | ölçek 4,3× / 3,0× | Faz 4 tasarım §4 |
| Kör örneklem 2. tur | %98,6 (72/73) | `docs/experiments.md` |
| Hareket oracle BA (val) | 0,949 | `docs/experiments.md` |
| Takipçi eşikleri HOTA (val) | 0,220 → 0,292 | `docs/experiments.md` |
| Öğretmen etiketi kör denetim | 0,92 | `results/data/m3ot_teacher_audit.json` |

---

## 16. Mülakat: sık sorular ve cevap iskeletleri

Her cevabı **"durum → ne yaptım → sayı → ders"** sırasıyla kur. 1 dakikayı geçme.

1. **"Projeni anlat."** → §1 paragrafı + sevk edilen sistem + bir sağlamlık sayısı (IR kapalı 12,8 → 72,0) +
   "C++ motor Python'la birebir, internetsiz çalışıyor".
2. **"Neden IR etiketleri?"** → İki set ayrı etiketlenmiş; gece RGB'de görünmeyen araçlar yalnız IR setinde;
   RGB setiyle de ölçtüm ve ayrı raporladım (ADR 0003).
3. **"Modality dropout neden işe yaradı?"** → Model eğitimde bir kameranın bazen yok olduğunu görünce tek
   kameraya yaslanmayı bırakıyor; temiz bedeli ≈ 0,6 puan, kamera kaybında +59 puan.
4. **"Neden E4 değil E3c?"** → Kuralı ölçümden önce yazdım (p95 ≤ 100 ms); E4 daha sağlam ama laptopta 131 ms.
   GPU'lu cihazda E4 seçilebilir.
5. **"INT8 neden hızlandırmadı?"** → Ölçtüm, yavaşlattı; olası neden kısmi nicemlemede sürekli tür dönüşümü
   (tahmin, ölçmedim); başlık nicemlenince doğruluk da düşüyor. Dersim: "INT8 hızlıdır" varsayımını donanımda
   ölçmeden kabul etmemek.
6. **"Başarısız bir şey anlat."** → Faz 2 kapısı: bozukta +40 puan ama temizde −3; neden karanlıkta faz
   korelasyonu; düzeltme yanıt eşiği + zaman kuralı; kuralı sonradan gevşetmedim.
7. **"C++ ile Python'un aynı olduğunu nereden biliyorsun?"** → Eşlik testleri: aynı ONNX, aynı fikstürler;
   letterbox tensör sha256'ları aynı, takip 600 karede fark 0; CI'da zorunlu.
8. **"Takipçin iyi mi?"** → Oracle satırıyla ayırdım: GT kutularla HOTA 0,95, model kutularıyla 0,49; sorun
   dedektörde. Sonra eşikleri val'de kuralla seçtim, M3OT ile ince ayar yaptım.
9. **"Hareketli/duran'ı nasıl ölçtün?"** → Döngüsel ölçüm tuzağı; ayrı etiketleyici (SIFT + homografi,
   ortalanmış pencere) + kör insan örneklemi (%98,6); oracle BA 0,949.
10. **"Gerçek dünyada ne eksik?"** → Gerçek konum (GPS) yok, kare hızı doğrulanmadı, tek seed, M3OT test 2 dizi,
    yoğun trafikte tespit zayıf, AGPL lisans.
11. **"Kamera hareketi telafisini neden takipçiye koymadın?"** → Kapsam tercihi: takipçinin eşliği ve
    ölçümü sabit sürüme bağlı; kazanç iddia edilmedi, Faz 5 notu.

---

## 17. Kendini sına: karışık soru bankası

Kısa cevap ver, sonra ilgili bölüme bakıp kontrol et.

1. mAP50 ile mAP50-95 farkı?
2. Bootstrap GA'nın içermediği belirsizlik ne?
3. HBB'nin OBB'ye göre zayıflığı ne (ADR 0002)?
4. 4 kanallı TIFF'te OpenCV ne bozuyordu, nasıl yakalandı?
5. E3 IR kapanınca neden 12,8'e çöküyor?
6. WBF'de `ir_weight` gece neden daha büyük seçildi?
7. p95 ≤ 100 ms neden "10 kare/sn" demek?
8. Statik ve dinamik nicemleme farkı?
9. Youden J ile `r_min` seçim ölçütü neden farklı (biri "dengeli", biri "önce temizde zarar yok")?
10. ECE nasıl hesaplanır, izotonik regresyon neden monoton?
11. ByteTrack'in iki aşamalı eşleşmesi hangi durumu kurtarır?
12. HOTA neden DetA ve AssA'nın geometrik ortalaması?
13. Oracle satırı neyi ölçer, neyi ölçmez?
14. LK optik akışta ileri-geri denetimi neyi eler?
15. Benzerlik dönüşümü kaç serbestlik, homografi kaç?
16. Anahtar kareye kayıt zincir kaymasını neden azaltır?
17. Hız neden kutu boyuna bölünüyor ve bunun kare hızına bağımlılığı ne?
18. Histerezis bandındaki örnekler ölçümde neden "dışlanan nüfus"?
19. Kör örneklemde örnekler neden etiketleyiciden **bağımsız** seçildi?
20. Wilson aralığı ne zaman normal yaklaşımdan iyidir?
21. Yeniden konumlama başarısızsa bölgelere ne olur, neden?
22. Sayım hattında aracın hattın uzantısından dolanması neden sayılmıyor?
23. Öğretmen etiketinin yanlılık riski ne, nasıl ölçüldü?
24. Parçalı çıkarım neden küçük araçta yardım eder, bedeli ne?
25. Tarayıcı motorla neden doğrudan konuşmuyor?

---

## 18. Çalışma planı önerisi (10 oturum, oturum başına ≈ 1,5-2 saat)

| Oturum | Oku | Deftere | Ek kaynak |
|---|---|---|---|
| 1 | Bu rehber §1-3, README | Akış diyagramı (§2), faz tablosu | — |
| 2 | §4, spec §4 (YOLO 101) | IoU, P/R, AP, NMS, letterbox | Ultralytics belgeleri: Detect, Train, Val, Export |
| 3 | §5, ADR 0002-0005, `docs/data-report.md` | Veri kararları, TIFF tuzağı | DroneVehicle makalesi (Sun ve ark., IEEE TCSVT 2022) |
| 4 | §6-7, `docs/experiments.md` E1-E4b | Füzyon türleri, WBF, bootstrap | WBF makalesi (Solovyev ve ark.) |
| 5 | §8-9, ADR 0007, `docs/robustness.md`, `docs/results.md` Edge | Sevk kuralı, INT8 bulgusu | ONNX Runtime quantization belgesi |
| 6 | §10, `docs/health.md`, `docs/design/faz2.md` §6 | Sağlık ölçüleri, kapı, ECE | Kalibrasyon: Guo ve ark. 2017 "On Calibration of Modern Neural Networks" |
| 7 | §11.1-11.2, `docs/system.md`, `engine/src/runtime.cpp` | Motor aşamaları, eşlik | — |
| 8 | §11.3-11.7, ADR 0008, `kalman.hpp`, `tracker.cpp` | ByteTrack, HOTA/MOTA/IDF1, oracle | ByteTrack (Zhang ve ark. 2022), HOTA (Luiten ve ark. 2021) |
| 9 | §12.1-12.5, `docs/design/faz4.md` §4-5.3, `motion/ego.py` | Ego-hareket, histerezis, döngüsel ölçüm | OpenCV belgeleri: optical flow, `estimateAffinePartial2D`, ORB; BoT-SORT (GMC) |
| 10 | §12.6-12.9, §13-17, `docs/design/faz4-kapsama.md` | Bölgeler, E3c-M, sayı kartı | §16'yı sesli prova et |

Her oturumun sonunda: o bölümün **❓** sorularını kapalı defterle cevapla, §17'den 3 rastgele soru seç.

---

## 19. Daha derine: belge haritası

| Soru | Belge |
|---|---|
| Bütün proje tanımı, gereksinimler | `docs/spec.md` (uzun; §1-3, §9, §12, §22 yeter) |
| Neden böyle karar verildi? | `docs/adr/` |
| Her deneyin hipotezi ve sonucu (tarih sırası) | `docs/experiments.md` |
| Sonuç tabloları | `docs/results.md` (val), `docs/results-test.md` (test), `docs/robustness.md` |
| Faz 2 | `docs/design/faz2.md`, `docs/health.md` |
| Faz 3 | `docs/design/faz3.md`, `docs/system.md`, `docs/reviews/2026-10-19-faz3-kod-incelemesi.md` |
| Faz 4 | `docs/design/faz4.md`, `docs/design/faz4-kapsama.md`, `docs/plans/F4.md` |
| M3OT | `docs/data/m3ot.md`, `docs/spikes/04-external-datasets.md` |
| Colab GPU bütçesi | `docs/gpu-budget.md` |
| Literatür araştırması | `docs/research/2026-10-04-derin-arastirma.md` |

---

## 20. Açık uçlar ve bilinen sınırlamalar

- **Tek seed:** her model bir kez eğitildi; GA'lar seed varyansını içermiyor.
- **Val sayıları iyimser:** seçimler val'de yapıldı; tarafsız sayılar test tablosunda.
- **ONNX ≈ 1 puan düşük** (nedeni tam ölçülmedi).
- **M3OT'ta hizasızlık alarmı çalışmıyor** (doğal kayma; ürün kuralı gerçek 20 px kaymayı kaçırıyor).
- **Yoğun trafik:** M3OT val'de tespit zayıf (AP50 0,40); E3c-M ve parçalı çıkarım bunun için.
- **Model 5 sınıf biliyor:** motosiklet yok.
- **Harita göreli:** GPS, metre, km/sa yok; eşikler M3OT'un (doğrulanmamış) kare hızına ait.
- **Lisans:** AGPL-3.0 (Ultralytics).
- **Faz 4 kalan:** E3c-M eğitimi ve sevk kararı → ölçümler → bench → elle kontrol → test bir kez →
  `v0.3.0`. Sonra Faz 5 (README, demo videosu, teknik yazı).
