# Derin araştırma: eksikler, literatür, GitHub ve sektör (2026-10-04)

Bu rapor spec v2.2'yi (docs/spec.md) dört kaynağa karşı kontrol ediyor: akademik makaleler, GitHub projeleri, sektör pratikleri ve Ultralytics'in kaynak kodu. Amaç, plan devam etmeden önce yanlış varsayımları ve değerli eksikleri bulmak.

Notasyon: **[DOĞRULANDI]** kaynak kodda veya birincil kaynakta gördüm. **[KAYNAK]** ikincil kaynak veya özet; sayı makalede kontrol edilmeli. **[ÇIKARIM]** benim yorumum, spike ile doğrulanmalı.

---

## 1. Özet: önem sırasına göre 10 bulgu

| # | Bulgu | Etki | Ne zaman |
|---|---|---|---|
| 1 | Ultralytics 4 kanallı modelde ilk katmana **RGB ağırlıklarını kopyalıyor, termal kanalı rastgele bırakıyor.** Spec'teki `rgb_mean` hazır değil; `random` da varsayılan değil. | E3/E3b tanımı yanlış | **H1 (Spike A)** |
| 2 | Spec kamera kaybını **ölçüyor** (E6) ama modeli buna karşı **eğitmiyor.** Literatürdeki standart çözüm *modality dropout*. Erken füzyonun bir kanal sıfırlanınca çökmesi neredeyse kesin. | En güçlü mülakat hikâyesi eksik | H3 |
| 3 | **ByteTrack-cpp'ye son commit Nisan 2024'te atılmış** (yaklaşık 2,5 yıldır sessiz). | Bağımlılık riski | H5-H6 |
| 4 | Laptoptaki Ryzen 7 8845HS'in **NPU'su var.** Ryzen AI 1.8, bu nesilde (Hawk Point) CNN INT8'i destekliyor (ONNX Runtime Vitis AI EP). Spec bunu hiç kullanmıyor. | Edge tablosuna 3. donanım | H4 (spike) |
| 5 | YOLO26 INT8'de kayıp **çalışma ortamına göre çok değişiyor.** Bir deploy reposunda TensorRT'de %5,65, ORT'de %0,72 mAP kaybı raporlanmış; fark detection head'in INT8'e çevrilmesinden geliyor. | INT8 tarifi eksik | H4 |
| 6 | Tek seed ile 1-2 puanlık mAP farkları anlamlı değil. Spec'te **belirsizlik ölçümü yok.** | "Dürüst ölçüm" iddiası zayıf | H2 |
| 7 | DroneVehicle literatürünün büyük kısmı **OBB** ve çoğu zaman farklı etiket kuralıyla raporluyor. HBB sonuçlarımız tablolarla doğrudan kıyaslanamaz. | README'de yanlış kıyas riski | H4/H8 |
| 8 | TEKNOFEST 2026 görevleri spec'tekinden geniş: **insan, UAP/UAİ, hareketli/duran taşıt, iniş uygunluğu, GPS'siz pozisyon kestirimi, referans nesne eşleme.** | §25 eksik | Bilgi / ek hedef |
| 9 | Sektörde video **GStreamer/RTSP** ile alınıyor, telemetri **STANAG 4609 / MISB 0601 KLV** ile taşınıyor. Spec'te video hattı ve meta veri yok. | Savunma sektörü dili eksik | Ek hedef |
| 10 | YOLO27 duyuruldu ama ağırlıkları yayımlanmadı ve tarih yok. YOLO26 doğru seçim; ADR 0001'de not edilmeli. | Yok (not) | H1 (ADR) |

---

## 2. Planı doğrudan etkileyen bulgular

### 2.1 İlk katman ağırlık aktarımı (E3 / E3b) — [DOĞRULANDI]

`ultralytics==8.4.172`, `ultralytics/nn/tasks.py`, `BaseModel.load()`:

```python
updated_csd = intersect_dicts(csd, self.state_dict())   # şekli uyan katmanlar
...
first_conv = "model.0.conv.weight"
if first_conv not in updated_csd and first_conv in state_dict:
    c1, c2, h, w = state_dict[first_conv].shape          # (out, in=4, k, k)
    cc1, cc2, ch, cw = csd[first_conv].shape             # (out, in=3, k, k)
    if ch == h and cw == w:
        c1, c2 = min(c1, cc1), min(c2, cc2)
        state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
```

Yani 4 kanallı modelde girdi kanalları 0-2 COCO ağırlıklarını alıyor, **kanal 3 (termal) rastgele kalıyor.** PR #20223'teki eski açıklama ("ilk katmanı atlar") artık geçerli değil. Issue #21278'deki "499/499 transferred" logu da bu kod yolundan geliyor.

**Sonuçlar:**
- Spec'teki `first_conv_init: rgb_mean | random` seçeneklerinden hiçbiri Ultralytics'in varsayılanı değil. Varsayılan: **"RGB kopya + T rastgele"**.
- `rgb_mean` için eğitimden önce küçük bir ağırlık cerrahisi gerekiyor (`w[:, 3] = w[:, :3].mean(1)`). Bunu Ultralytics'e yama yapmadan, modeli `ch=4` ile kurup `state_dict` düzenleyerek yapabiliriz. CLAUDE.md'deki "iç API'ye yama yok" kuralıyla çelişmiyor ama ADR'ye yazılmalı.
- **Kanal sırası kritik.** Kopyalanan ağırlıklar ilk üç kanalın COCO'daki sırayla (RGB) geldiğini varsayıyor. İki gözlem [DOĞRULANDI]:
  - `utils/patches.py::imread`, TIFF'i `cv2.imdecodemulti(..., IMREAD_UNCHANGED)` ile okuyor.
  - `data/augment.py:2520`, `Format` dönüşümü BGR→RGB çevirmesini **sadece 3 kanallı görüntüde** yapıyor (`img[::-1] if ... img.shape[0] == 3`). 4 kanallıda dosyadan çözülen sıra modele olduğu gibi gidiyor.

  [ÇIKARIM] OpenCV, RGB fotometrili 4 örnekli bir TIFF'i BGRA sırasına çevirerek çözer. Öyleyse R,G,B,T diye yazdığımız dosya modele B,G,R,T olarak gider. COCO'nun "R" ağırlıkları mavi kanala uygulanır. E1 (3 kanal) ise doğru RGB görür; bu da E1 ile E3 kıyasını sessizce bozar. Spike A'nın `channel_order_matches_RGBT/BGRT` ölçümü bu yüzden şart. Sonuca göre ya dosyayı B,G,R,T diye yazarız ya da `rgb_mean` cerrahisinde ağırlıkları yeniden sıralarız.

**Öneri:** E3 varyantlarını şöyle yeniden tanımla: `ul_default` (RGB kopya + T rastgele), `rgb_mean`, isteğe bağlı `random` (tam rastgele ilk katman). Spike A'ya "ilk katmanın 4. kanal ağırlıkları istatistiği" çıktısını ekle. Görev 6'daki ADR 0004'e kanal sırasıyla birlikte yaz.

### 2.2 Modality dropout: kamera kaybına karşı eğitim — [KAYNAK]

Spec'in §12.3 sağlamlık paketi "RGB = 0 veya IR = 0" senaryosunu ölçüyor, ama hiçbir model bu duruma göre eğitilmiyor. Literatürde durum net:
- Eksik modalitede IR-görünür dedektörlerin, özellikle baskın modalite gidince ciddi düştüğü gösterilmiş. Çözüm olarak *pseudo modality dropout* öneriliyor (Scarf-DETR, arXiv 2511.06406).
- Aşırı modalite dengesizliği üzerine bir çalışma (arXiv 2505.22154), sahte bozulma ve tutarlılık kısıtıyla tek sensör kaybında kaçırma oranını %55'e kadar azaltıyor.
- Düz modality dropout, iki modalite de varken performansı biraz düşürebiliyor. Bu da ölçülecek ilginç bir ödünleşim.

**Öneri: E3c deneyi.** Erken füzyon + eğitimde olasılıkla (ör. p=0.15) RGB kanallarını ya da T kanalını sıfırlama. Maliyeti düşük: tek bir augmentation ve bir eğitim koşusu. Hikâye değeri çok yüksek: *"Termal kamera kapanınca erken füzyon X'e düştü; modality dropout ile Y'de kaldı, tam modalitede bedeli Z oldu."* Bu, spec §26'daki 2. mülakat hikâyesini ölçülmüş bir çözüme çeviriyor.

Teknik not [ÇIKARIM]: Ultralytics'e özel augmentation eklemek için custom `Dataset`/`Trainer` alt sınıfı veya callback gerekir. "İç API'ye yama yok" kuralına uyan yol alt sınıf yazmak; ADR ister.

### 2.3 ByteTrack-cpp'nin bakımı durmuş — [DOĞRULANDI]

GitHub API (2026-10-04): `Vertical-Beach/ByteTrack-cpp`, 275★, MIT, **son push 2024-04-08.** Kod küçük ve kararlı olduğu için hâlâ çalışır, ama sektör tercihi olarak sunmak zayıf.

**Seçenekler:**
1. **Takipçiyi kendin yaz (öneri).** Kalman + Macar algoritması + ByteTrack'in iki aşamalı eşleşmesi C++17'de birkaç yüz satır. Spec §15.2'deki "biz yazıyoruz" listesine tam uyuyor. Python referansı olarak **roboflow/trackers** (Apache-2.0, 3,9k★, aktif; SORT, ByteTrack, OC-SORT, BoT-SORT içeriyor ve `trackers eval` ile HOTA/CLEAR/Identity hesaplıyor) kullanılırsa C++/Python eşlik testi takip için de kurulabilir.
2. ByteTrack-cpp'yi commit hash'iyle sabitleyip kullanmak ve riski ADR'de yazmak.

Ayrıca **BoxMOT** (8,3k★) sektörde çok kullanılan bir Python takip kütüphanesi ama AGPL; roboflow/trackers lisans açısından daha temiz.

### 2.4 Laptop NPU'su: üçüncü donanım satırı — [KAYNAK] + [ÇIKARIM]

- Ryzen AI Software 1.8.0 (Ağustos 2026) uyumluluk tablosunda **Phoenix/Hawk Point (PHX/HPT) için "CNN INT8 ✅"** yazıyor. 8845HS bir Hawk Point işlemcisi.
- Yol: ONNX export (opset ≥ 20) → AMD **Quark** ile INT8 kalibrasyonu → ONNX Runtime **VitisAI Execution Provider**.
- Ana platform Windows 11; 1.7.1'den beri Linux kurulumu da var. [ÇIKARIM] WSL2 NPU'yu görmeyebilir; spike'ta Windows tarafında denenmeli.
- Risk [ÇIKARIM]: 4 kanallı girdi ve YOLO26'nın bazı operatörleri NPU'da desteklenmeyebilir, CPU'ya düşebilir. Spike'ta "kaç düğüm NPU'da çalıştı" raporlanmalı.

**Öneri:** H4'e zaman kutulu (2-3 saat) bir spike ekle. Çalışırsa S6 edge tablosu "laptop CPU / laptop NPU / Colab T4" olur. Aynı ONNX modelinin üç farklı hızlandırıcıda ölçülmesi savunma ilanlarındaki "edge'e deploy" maddesini doğrudan karşılar. Çalışmazsa bulgu olarak yazılır.

### 2.5 INT8 tarifi: head'i FP32 tut — [KAYNAK]

YOLO26 makalesi INT8'de kaybın çok az olduğunu söylüyor. Ancak bağımsız bir deploy reposunda (Karan-Anchan/edge-yolo26-deployment) INT8 kaybı **TensorRT'de %5,65, ONNX Runtime'da %0,72** olarak raporlanmış. ORT yolunda 94 head düğümü FP32 kalmış, TRT yolunda kalmamış.

**Öneri (FR-X2'ye ek):** ORT `quantize_static` çağrısında head düğümlerini `nodes_to_exclude` ile FP32 bırak (karışık hassasiyet). Edge tablosuna "INT8 (tam)" ve "INT8 (head FP32)" diye iki satır koy. Bu da ölçülmüş, anlatılabilir bir ödünleşim olur.

### 2.6 İstatistiksel belirsizlik — [KAYNAK]

Tespit metriklerinde varyans yüksek: kutular görüntü içinde kümeleniyor ve bağımsız örnek sayılmıyor. Seed'ler arası fark tek başına 1 puanı geçebilir.

**Öneri (maliyet sırasıyla):**
1. **Görüntü düzeyinde bootstrap güven aralığı** (GPU gerektirmez). Test tahminleri bir kez kaydedilir; görüntüler 1000 kez yeniden örneklenip mAP50 hesaplanır, %95 aralık raporlanır. `baykus.eval`'e küçük bir fonksiyon. Ana tablo ve aydınlatma kırılımı için yeterli.
2. **Eşleştirilmiş bootstrap:** aynı yeniden örneklemle iki modelin farkı (ör. E3 − E1, karanlık gecede). "Füzyon gerçekten yardım ediyor mu?" sorusunun dürüst cevabı bu.
3. Bütçe kalırsa ana karşılaştırma (E1/E2/E3) için `n` modeliyle 3 seed.

Sonuç JSON şemasına (§12.4) `ci95` alanları eklenmeli.

### 2.7 Literatürle kıyas: HBB ve OBB — [KAYNAK]

- UAVD-Mamba (arXiv 2507.00849) tablosunda "tüm dedektörler OBB başlığı kullanıyor" notu var. YOLOv11 tek modalite sonuçları: RGB %75,3, IR %79,3 mAP. Etiket olarak "daha çok etiketi olan modaliteyi" kullanıyorlar ve görüntüyü 840×712'den 640×640'a yeniden boyutlandırıyorlar, yani kenarlığı kırpmıyorlar.
- MFE-DETR (Symmetry 2026) %78,6 mAP50 / %57,8 mAP50:95 raporluyor. CALNet %76,4.
- Arama sonuçlarında YOLOv8s'in HBB tek modalite sonuçları da geçti (RGB %72,8, IR %78,5 mAP50). Hangi makaleden geldiğini netleştiremedim; kullanmadan önce doğrulanmalı.

**Öneri:** README'de literatür tablosu verilecekse her satıra **kutu tipi, etiket seti, kenarlık kırpma, imgsz** sütunları eklenmeli. Spec §26'daki "dürüst ölçüm notları" bölümü bunu zaten hedefliyor; somut sütunlarla güçlenir. E11 (OBB) ek hedefi bu yüzden değerli.

### 2.8 Hizasızlık: veriden ölçülebilir bir analiz — [KAYNAK] + [ÇIKARIM]

Literatür DroneVehicle'daki RGB-IR konum kaymasını ciddi bir sorun olarak ele alıyor:
- arXiv 2502.09311: kaymayı etiket gürültüsü olarak modelliyor (CBC + SWCA). DroneVehicle'ın "shift subset"inde +8,6 mAP50 elde ediyor.
- DVTOD/CMA-Det, TSFADet (arXiv 2209.13801), C²Former, LER-YOLO (2026, güvenilirlik haritasıyla uzman yönlendirme) da aynı sorunu ele alıyor.

**Öneri (H1 Görev 15'e ucuz bir ek):** İki etiket setindeki kutuları IoU ile eşleştir ve merkez kayması (piksel) dağılımını çıkar. Bunu veri raporuna histogram olarak koy. Değerlendirmeye "kayma büyüklüğü" kırılımı ekle (ör. < 3 px / 3-10 px / > 10 px). Böylece E6'daki yapay kaydırma deneyi gerçek veride ölçülen kayma dağılımına bağlanır. Spec §8.2'deki "1-20 piksel" tahmini de ölçülmüş sayıya dönüşür.

---

## 3. Akademik literatür haritası

| Konu | Önemli çalışmalar | Bizim için anlamı |
|---|---|---|
| Veri seti | DroneVehicle / UA-CMDet (T-CSVT 2022, 742★ repo) | Ana veri; OBB, iki etiket seti, beyaz kenarlık |
| Transformer füzyon | C²Former (2023), MFE-DETR (2026), CALNet | Orta füzyonun tavanı; bizim kapsamımız dışında |
| Mamba / durum uzayı | WaveMamba (ICCV 2025), COMO (Inf. Fusion 2026), UAVD-Mamba (2025), MS2Fusion (2025) | 2025-26 trendi; README'de "ilgili çalışmalar"a girer |
| Hizasızlık | TSFADet (2022), CMA-Det/DVTOD (2024), Position-shift CBC/SWCA (2025), LER-YOLO (2026), CAFNet (2026) | E5/E6 ve §2.8'deki analiz |
| Eksik modalite | Scarf-DETR (2025), aşırı modalite dengesizliği (2025), MMP, CoLA | E3c modality dropout |
| Aydınlatma farkındalığı | IAF R-CNN (2018), IMHFNet, IAF-RTDETR (2026) | Geç füzyonda WBF ağırlıklarını aydınlatmaya göre seçmek (E4'e ucuz varyant) |
| Sağlamlık benchmark'ı | WiSE-OD (WACV 2026, LLVIP-C/FLIR-C) | §12.3 bozulma paketinin literatürdeki karşılığı; atıf yapılmalı |
| Termal ton eşleme | N-AGC (ICCVW 2025), Thermal Chameleon (2024) | Gerçek 14/16-bit radyometrik sensörle 8-bit veri arasındaki fark; README sınırlamaları |
| Üç modlu / yeni veri | RGB-termal-event UAV veri seti (arXiv 2604.16630), M3T (2026) | Gelecek iş |
| YOLO26 edge | Aquaculture benchmark (arXiv 2607.09835) | YOLO26n CPU'da hızlı, ama veri verimliliği YOLOv8'den düşük. 17k eğitim görüntüsüyle sorun beklenmez. |

---

## 4. GitHub haritası (GitHub API, 2026-10-04)

| Repo | ★ | Lisans | Son push | Rol |
|---|---|---|---|---|
| ultralytics/ultralytics (8.4.172) | — | AGPL-3.0 | aktif | Ana çerçeve. YOLO27 "yakında", ağırlık yok |
| wandahangFY/YOLOv11-RGBT | 724 | AGPL-3.0 | 2025-12-15 | Orta füzyon (E10). YOLO26 desteği yok, DroneVehicle hazır değil (FLIR/KAIST/VEDAI/LLVIP/M3FD var) |
| Geekgineer/YOLOs-CPP | 1095 | AGPL-3.0 | 2026-08-23 | C++ referansı; aktif |
| Vertical-Beach/ByteTrack-cpp | 275 | MIT | **2024-04-08** | Bakımı durmuş (bkz. §2.3) |
| roboflow/trackers | 3902 | Apache-2.0 | 2026-10-02 | Python takip referansı + değerlendirme |
| mikel-brostrom/boxmot | 8307 | AGPL-3.0 | 2026-09-18 | Python takip kütüphanesi |
| ZFTurbo/Weighted-Boxes-Fusion | 1831 | MIT | 2026-07-27 | WBF referansı (ensemble-boxes) |
| obss/sahi | 5528 | MIT | 2026-09-30 | Dilimli çıkarım; 640×512 girdide sınırlı fayda, gecikme maliyeti yüksek |
| roboflow/rf-detr | 9679 | Apache-2.0 | 2026-10-03 | İzinli lisanslı alternatif dedektör |
| Peterande/D-FINE, lyuwenyu/RT-DETR | 3349 / 5567 | Apache-2.0 | 2026 | İzinli lisanslı DETR ailesi |
| MultimediaTechLab/YOLO | 1740 | MIT | 2026-03 | MIT lisanslı YOLO (v9 ailesi) |

**Benzer uçtan uca portfolyo projeleri** (çıtayı görmek için):
- *GBR-RL/takt-vision:* C++20, kilitsiz aşamalar, ORT + ByteTrack, p99 gecikme, Python ile kıyas
- *Manishnayak234/edge-tracking:* Jetson Orin'de C++/CUDA, TensorRT YOLOv8n + ByteTrack, 60 fps
- *Halok600/The-Aerial-Guardian:* drone videosunda YOLOv8n-P2 + sıfırdan ByteTrack + **ECC kamera hareketi telafisi**, tarayıcıda ORT Web

Bunların hiçbiri RGB-T füzyonu, kırılımlı dürüst ölçüm ve sağlamlık analizini birlikte yapmıyor. Spec §3.2'deki konumlandırma hâlâ geçerli. Ancak "C++ + ORT + ByteTrack + p99" artık tek başına ayırt edici değil; fark füzyon ve ölçüm tarafında yaratılmalı.

---

## 5. Sektör

### 5.1 Ürünlerde ne var
- **Teledyne FLIR Prism** ailesi (2026): Prism SKR (otomatik hedef tanıma, ATR), Prism C-UAS (Nisan 2026), Prism Ground ISR (Haziran 2026). Ortak noktalar: termal ön işleme (gürültü azaltma, lokal kontrast, upsampling), dedektör + çok nesneli takip, **NVIDIA Orin ve Qualcomm** tabanlı gömülü donanım, termal + seçili görünür kamerayla multispektral yapılandırma. Bu, Baykuş'un mimarisiyle aynı şekil.
- **Video hattı:** NVIDIA DeepStream (GStreamer üstünde decode → TensorRT → tracker → metadata). Orin NX'te INT8 YOLO 12-18 ms/kare. Endüstride video "klasördeki görüntüler" değil, RTSP akışı olarak geliyor.
- **Meta veri:** STANAG 4609 / MISB ST 0601 KLV. Her karenin platform konumu, yönelimi ve kamera bakış açısı video içinde taşınıyor; tespitler buradan **coğrafi koordinata** çevriliyor. Python'da `klvdata` kütüphanesi (paretech/klvdata) var.

### 5.2 İlanlar
- Baykar Yapay Zeka Mühendisi ilanı: **C++ ve Python'a iyi derecede hâkimiyet**, algoritma/veri yapıları, matematik-geometri; tercihen PyTorch, **Linux sistem programlama, gerçek zamanlı sistemler**, Git. TensorRT/Jetson/ROS adı geçmiyor. Spec'in C++ motoru ve gecikme ölçümü bu ilanla doğrudan örtüşüyor.
- ASELSAN 2026 alımlarında "Görüntü İşleme Yazılım Tasarım Mühendisi" ve "Yapay Zeka Geliştirme ve Uygulama Mühendisi" pozisyonları var. STM ilanı erişime kapalıydı (403).

### 5.3 Lisans gerçeği
Savunma şirketleri kapalı ürün çıkarıyor. AGPL'li Ultralytics bu ürünlerde **Enterprise lisans olmadan kullanılamaz.** Bu yüzden sektörde Apache/MIT lisanslı RF-DETR, D-FINE ve RT-DETR'a ilgi artıyor. Portfolyo için AGPL sorun değil (spec §7.6 doğru). Ancak ADR 0001'e "üretimde bu seçim lisans yüzünden değişir; alternatifler şunlar" diye bir paragraf eklemek mülakatta olgunluk gösterir.

### 5.4 TEKNOFEST 2026 Havacılıkta Yapay Zeka (gerçek görevler)
Takvim: şartname 16.02.2026, başvuru 28.02, ön tasarım raporu 22.04, simülasyon 16.07, final 18-20.09.2026.

| Görev | İçerik | Baykuş'ta karşılığı |
|---|---|---|
| 1. Nesne tespiti | Taşıt, insan, UAP (uçan araba park alanı), UAİ (uçan ambulans iniş alanı); taşıt için **hareketli/duran**; UAP/UAİ için **iniş uygunluğu** | Sadece taşıt var; insan, UAP/UAİ, hareket durumu yok |
| 2. Pozisyon kestirimi | GPS kesilince görsel odometri | Yok |
| 3. Referans nesne tespiti | Önceden tanımlanmamış bir nesneyi referans görüntüyle bulma (ORB/SIFT + homografi) | Yok |
| Kısıtlar | ~7,5 FPS akış, internetsiz, sonuçlar sunucuya JSON | Spec'in internetsiz ve JSON API tasarımıyla uyumlu |

Açık kaynak takım repolarında (ör. Darülmu-AI-limin/BlackScope) yaygın yaklaşım şu: YOLO11m (COCO + VisDrone ince ayarı), hareketli/duran ayrımı için **ORB + RANSAC homografi ile kamera hareketi telafisi**, odometri için Lucas-Kanade + essential matrix.

**Öneri:** Kamera hareketi telafisiyle **hareketli/duran araç** ayrımı, takip modülüne iyi bir ek hedef. Takip zaten var; homografi + track hızı ile yapılır. Hem TEKNOFEST 2027'ye hazırlık hem savunma dili ("hareketli hedef göstergesi") açısından değerli.

---

## 6. Spec'teki diğer boşluklar

| Boşluk | Neden önemli | Öneri | Maliyet |
|---|---|---|---|
| Geç füzyonda sabit WBF ağırlıkları | Gece RGB güvenilmez; literatürün standart fikri aydınlatmaya göre ağırlık | E4b: WBF ağırlıklarını val'de aydınlatma kırılımı başına seç (meta'daki etiketle) | Düşük, GPU yok |
| Güven eşiği tek (0.25) | Operatör ekranında yanlış alarm / kaçırma dengesi önemli | Val'de aydınlatma başına PR eğrisi + seçilen çalışma noktası; isteğe bağlı kalibrasyon (ECE) | Düşük |
| Termal gerçekçilik | DroneVehicle IR'ı 8-bit ve AGC'den geçmiş; gerçek sensörler 14/16-bit radyometrik | README sınırlamaları + N-AGC atfı. Uygulama yok. | Yok |
| Veri sürümü | `results/*.json` hangi işlenmiş veriyle üretildiğini söylemiyor | İşlenmiş veri için SHA-256 manifest; run JSON'a `data_manifest_hash` | Düşük |
| Video girişi | UI klasörden okuyor; sektör RTSP kullanıyor | Ek hedef: motorda bir RTSP/dosya video girişi (FFmpeg/GStreamer) | Orta |
| Coğrafi konum | Savunmada tespitin değeri koordinatında | Ek hedef: KLV'li örnek bir video ile tespit → enlem/boylam demosu | Orta-yüksek |
| Takip veri seti | VT-MOT drone dizileri belirsiz | Alternatifler: VT-Tiny-MOT (RGB-T, araç sınıfları var, çok küçük nesneler), MMOT (multispektral, OBB; termal değil) | Bilgi |
| Sağlamlık paketinin literatür bağı | Bozulma seçimi gerekçesiz görünebilir | WiSE-OD / LLVIP-C / FLIR-C'ye atıf; şiddet seviyelerini onlara yakın seç | Yok |
| YOLO27 | Proje ortasında yeni model gelebilir | ADR 0001: "proje içinde model değiştirilmez, YOLO27 gelecek iş" | Yok |

---

## 7. Önerilen değişiklikler: nereye, ne zaman

**H1 (şimdi, plan güncellemesi gerektirir):**
1. Spike A'ya (Görev 3) ekle: ilk katmanın 4. kanal ağırlık istatistiği ve kanal sırası kararı. ADR 0004'e işle. *(§2.1)*
2. Spec §10.3/§13'te E3 varyantlarını `ul_default | rgb_mean | random` olarak düzelt. *(§2.1)*
3. Görev 15 veri raporuna RGB-IR etiket kayma histogramı ekle. *(§2.8)*
4. ADR 0001'e YOLO27 notunu ve lisans/alternatif paragrafını ekle. *(§5.3, §1-#10)*

**H2-H3:**
5. `baykus.eval`'e bootstrap güven aralığı + eşleştirilmiş fark; JSON şemasına `ci95`. *(§2.6)*
6. E3c modality dropout deneyi. *(§2.2)*
7. E4b aydınlatmaya göre WBF ağırlıkları. *(§6)*
8. Kayma büyüklüğü kırılımı. *(§2.8)*

**H4:**
9. INT8'de head FP32 varyantı. *(§2.5)*
10. NPU spike'ı (zaman kutulu). *(§2.4)*

**H5-H7:**
11. Takipçi kararı: kendin yaz + roboflow/trackers eşliği (öneri) veya ByteTrack-cpp'yi sabitle. ADR 0007'ye ekle. *(§2.3)*
12. Ek hedefler listesine: hareketli/duran araç (kamera hareketi telafisi), video girişi. *(§5.4, §6)*

**Kapsam dışı kalmalı:** Mamba/transformer füzyon mimarisi yazmak, SAHI, KLV geolokasyon (zaman kalmazsa). 8 haftalık takvim zaten dolu.

---

## 8. Doğrulanması gerekenler

- [ ] Kanal sırası: Ultralytics 4 kanallı TIFF'i hangi sırayla okuyor? (Spike A zaten ölçüyor)
- [ ] `rgb_mean` ağırlık cerrahisi `model.train()` sırasında üzerine yazılıyor mu? Trainer `pretrained` ağırlıkları yeniden yükleyebilir; spike'ta kontrol edilmeli.
- [ ] NPU: Windows'ta Ryzen AI kurulumu, 4 kanallı YOLO26 ONNX'in kaç düğümü NPU'da çalışıyor? WSL2 NPU'yu görüyor mu?
- [ ] YOLOv8s HBB DroneVehicle sonuçlarının (RGB %72,8 / IR %78,5) kaynak makalesi
- [ ] VT-MOT ve VT-Tiny-MOT lisans ve erişim durumu
- [ ] TEKNOFEST 2027 şartnamesi (Şubat 2027'de çıkınca) termal görevleri ve sınıfları

---

## 9. Kaynaklar

**Ultralytics / YOLO**
- Ultralytics repo ve kaynak kodu (`nn/tasks.py`, v8.4.172) — https://github.com/ultralytics/ultralytics
- YOLO27 sayfası — https://docs.ultralytics.com/models/yolo27
- Çok kanal PR #20223 — https://github.com/ultralytics/ultralytics/pull/20223
- Multispektral pretrained aktarım sorusu #21278 — https://github.com/ultralytics/ultralytics/issues/21278
- COCO8-Multispectral — https://docs.ultralytics.com/datasets/detect/coco8-multispectral
- YOLO26 mimari ve benchmark — https://arxiv.org/abs/2509.25164
- YOLO26 edge benchmark (aquaculture) — https://arxiv.org/abs/2607.09835
- YOLO26 deploy ölçümleri (TRT/ORT INT8) — https://github.com/Karan-Anchan/edge-yolo26-deployment

**RGB-T tespit**
- DroneVehicle — https://github.com/VisDrone/DroneVehicle · https://arxiv.org/abs/2003.02437
- C²Former — https://arxiv.org/abs/2306.16175
- WaveMamba — https://arxiv.org/abs/2507.18173
- UAVD-Mamba — https://arxiv.org/abs/2507.00849
- MFE-DETR — https://www.mdpi.com/2073-8994/18/3/417
- EGM-Det — https://arxiv.org/abs/2608.11685
- Multispectral state-space fusion — https://arxiv.org/abs/2507.14643
- Tri-modal UAV fusion — https://arxiv.org/abs/2604.16630
- Awesome-RGBT-Fusion — https://github.com/yuanmaoxun/Awesome-RGBT-Fusion

**Hizasızlık / eksik modalite / sağlamlık**
- Position shift (CBC/SWCA) — https://arxiv.org/abs/2502.09311
- TSFADet — https://arxiv.org/abs/2209.13801
- DVTOD / CMA-Det — https://www.semanticscholar.org/paper/1e0bf4d09f3fa004660e30de539a4ec112f29bd5
- LER-YOLO — https://arxiv.org/abs/2605.20667
- Scarf-DETR (eksik modalite) — https://arxiv.org/abs/2511.06406
- Aşırı modalite dengesizliği — https://arxiv.org/abs/2505.22154
- WiSE-OD — https://openaccess.thecvf.com/content/WACV2026/papers/Medeiros_WiSE-OD_Benchmarking_Robustness_in_Infrared_Object_Detection_WACV_2026_paper.pdf
- N-AGC termal ton eşleme — https://openaccess.thecvf.com/content/ICCV2025W/MIRA/papers/Javidnia_Task-Driven_Neural_Adaptive_Gain_Control_16-bit-to-8-bit_Thermal_Tone_Mapping_for_ICCVW_2025_paper.pdf
- IAF R-CNN — https://arxiv.org/abs/1803.05347
- Tespitte güven aralıkları — https://blog.humblebee.ai/posts/confidence-intervals-for-object-detection-why-map-alone-isnt-enough

**Takip**
- roboflow/trackers — https://github.com/roboflow/trackers
- ByteTrack-cpp — https://github.com/Vertical-Beach/ByteTrack-cpp
- BoxMOT — https://github.com/mikel-brostrom/boxmot
- MMOT — https://arxiv.org/abs/2510.12565
- VT-Tiny-MOT — https://arxiv.org/abs/2412.10861
- Üretimdeki takipçiler (2026) — https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/multi-object-tracking-deepsort-bytetrack-ocsort

**Edge / donanım**
- Ryzen AI Software 1.8 — https://ryzenai.docs.amd.com/ · sürüm notları: https://ryzenai.docs.amd.com/en/latest/relnotes.html
- AMD NPU'da YOLO — https://www.adwaitx.com/amd-yolo-object-detection-npu-deployment/
- Ryzen AI EP karşılaştırma aracı — https://github.com/jankuzelka/ryzen-ai-provider-check
- DeepStream + YOLO26 — https://so-development.org/deepstream-yolo26-integration-on-jetson-edge-ai-platforms/

**Benzer projeler**
- takt-vision — https://github.com/GBR-RL/takt-vision
- edge-tracking — https://github.com/Manishnayak234/edge-tracking
- The-Aerial-Guardian — https://github.com/Halok600/The-Aerial-Guardian

**Sektör / TEKNOFEST**
- Teledyne FLIR Prism SKR — https://dronelife.com/2026/04/08/teledyne-flir-oem-upgrades-prism-skr-software-for-autonomous-target-recognition/
- Prism C-UAS — https://dronexl.co/2026/05/01/teledyne-flir-prism-c-uas-drones/
- STANAG 4609 — https://impleotv.com/2025/03/11/stanag-4609-isr-video/ · klvdata: https://github.com/paretech/klvdata
- Baykar Yapay Zeka Mühendisi — https://kariyer.baykartech.com/tr/acik-pozisyonlar/detay/yapay-zeka-muhendisi
- TEKNOFEST Havacılıkta Yapay Zeka — https://teknofest.org/tr/yarismalar/havacilikta-yapay-zeka-yarismasi/
- BlackScope (TEKNOFEST 2026 takım reposu) — https://github.com/Darulmu-AI-limin/TEKNOFEST2026-HavaciliktaYZ-OTR
- İzinli lisanslı dedektörler — https://github.com/roboflow/rf-detr · https://github.com/Peterande/D-FINE · https://github.com/lyuwenyu/RT-DETR
