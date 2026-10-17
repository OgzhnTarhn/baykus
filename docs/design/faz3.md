# Faz 3 Tasarımı — Sistem (yalın)

> **Tarih:** 2026-10-17 (taslak) · 2026-10-18 (inceleme bulguları ve kullanıcı kararları işlendi) ·
> **Durum:** taslak v2, kullanıcı onayı bekliyor · **Dal:** `f3`
> **Dayanak:** spec §2.3 S7-S8, §7.4 FR-E/FR-U, §7.5 NFR-1/5/6, §15-§18, §22 H6-H9, ADR 0006, ADR 0007;
> Faz 2 sonuçları (`docs/health.md`, `docs/experiments.md` "Karar: kapı düzeltmesi Faz 3'e").
> **Kullanıcı yönergesi (2026-10-17):** sorular atlandı, "tasarıma geç"; yalın, boş iş yok (Faz 2 ile aynı ilke).
> Kodlama başka bir oturumda yapılacak: bu belge ve ondan yazılacak plan (`docs/plans/F3.md`) yeterli olmalı.

**Kullanıcı kararları (2026-10-18, inceleme sonrası; `docs/experiments.md` "F3 · Tasarım kararları"):**

1. Hizasızlık yanlış alarmının düzeltmesi iki katmanlı bir kural (§4.2): **kare kuralı** — yanıt eşiği
   `r_min`, train'deki temiz karelerde kapı alarmı **≤ %2** olan en küçük değer (Youden J değil: Faz 2
   kararı "önce temiz veride zarar yok, sonra fayda"); **zaman kuralı** — videoda kayma son 5 karenin en
   az 3'ünde aynı yönde ve büyüklükte (±3 px) ise "hizasız". Karar ağacı train'de denendi ve elendi
   (bozulma türünü ezberledi); görüntüye bakan sinir ağı ve modelin yeniden eğitimi yok (sorun modelde
   değil, tek karede bilgi eksik).
2. Dikdörtgen (640×512) export **Faz 5'e** (2026-10-17 tarihli "Faz 3'te denenecek" kararı değişti; §3).
3. M3OT test dizilerindeki sağlık ölçümü **yalnız raporlanır**, hiçbir eşik ona bakılarak değişmez.
   Demo ölçütünü tutmazsa README'ye sınırlama, demo için M3OT'un test dışı bir dizisi (§4.7).
4. **Bu fazda (S) görev yok:** bütün kodu (C++ letterbox dahil) Claude yazar; kullanıcı inceler ve onaylar.
5. Ucuz ek ölçümler **girer**: motor tespitleriyle M3OT AP50 + ECE, kapının M3OT'ta sentetik bozulmayla
   denenmesi, internetsiz kurulum paketi (`docker save`). Donmuş kare **girmez**.
6. Takip eşliğinde belgelenmiş kayan nokta eşitliği istisnası kabul (kural ölçümden önce, §6.1 Kabul 4).

## 1. Neden ve ne

Faz 1-2 bir **model** ve onun ölçümlerini üretti: sevk edilen E3c · ONNX FP32 (laptop CPU p95 60 ms, test
mAP50 0,814), kare başına sensör sağlık dedektörü, kalibrasyon tablosu. Hepsi Python'da ve notebook'larda.
Faz 3 bunları **bir operatörün kullanacağı sisteme** çevirir:

**Bir C++ motoru görüntü çiftlerini alır, araçları bulur, takip eder, iki kameranın sağlığını ölçer;
bir web ekranı videoyu oynatıp kutuları, takip numaralarını ve kamera uyarılarını gösterir; hepsi
internete çıkmadan tek komutla (`docker compose up`) açılır.**

Mülakat hikâyesi (demo): "M3OT videosu oynuyor; araçlar kutulu ve numaralı. IR kamerayı ekrandan 20 piksel
kaydırıyorum; birkaç kare sonra 'IR hizasız' uyarısı çıkıyor, isterseniz sistem IR'yi kendisi kapatıyor.
Motor C++; çıktısı Python referansıyla CI'da tolerans içinde karşılaştırılıyor; internet kablosu çekili
çalışıyor." Hikâyenin dayandığı varsayım ("temiz karede alarm yok, 20 px kaymada alarm var") M3OT'ta
ölçülür (§4.7); ölçülmeden demo metni yazılmaz.

## 2. Hedefler

1. **C++ motoru (S7):** letterbox → ONNX Runtime → YOLO26 uçtan uca çıktı çözümleme → geri ölçekleme;
   Python referansıyla eşlik testi CI'da (FR-E1, E5, E6, E7).
2. **Sağlık ve güven motorda:** Faz 2 dedektörünün C++ karşılığı (Python ile eşlikli), hizasızlık
   alarmı için **kare kuralı** (train'de seçilmiş yanıt eşiği) ve videoda **zaman kuralı**, kalibre skor. Kapı varsayılan olarak **yalnız uyarır**.
3. **Takip:** `roboflow/trackers` ByteTrack'inin C++ uyarlaması (SORT tarzı Kalman, iki aşamalı eşleşme),
   oturum başına; Python referansıyla eşlik; M3OT test'te HOTA / MOTA / IDF1 (FR-E4, spec §16).
4. **Operatör ekranı:** tek sayfa Next.js; oynatıcı, kutular ve iz numaraları, sensör sağlık paneli ve
   alarmlar, arıza simülasyonu, canlı gecikme (FR-U1-U4 + Faz 3 ekleri).
5. **Kapalı ağ (S8):** `docker compose up` ile, çalışırken dış ağa çıkmadan; GitHub Actions CI (NFR-1, §18).

## 3. Hedef dışı (bilerek kesilenler)

| Kesilen | Neden |
|---|---|
| **Geç füzyon motorda + C++ WBF** (FR-E2 late_fusion, H7 #39) | E4 laptop CPU'da sevk edilmedi (p95 131,6 ms, ADR 0007). Sevk edilen sistem tek model. |
| **Ham çıktı + NMS yolu** (FR-E3, YOLO11 yedeği) | Uçtan uca FP32 sevk edildi; INT8 sevk edilmedi, YOLO11'e geçiş yok. |
| **TensorRT / CUDA / NPU yolu motorda** | Hedef donanım laptop CPU (ADR 0007). RTX 4060 isteğe bağlı satır olarak bile bu fazda yok. |
| **Dikdörtgen (640×512) export** (Faz 1 açık işi) | **Karar değişikliği** (kullanıcı, 2026-10-18; önceki karar 2026-10-17 "Faz 3'te C++ motoruyla"): sevk edilen modeli değiştirir, Colab'da val ölçümü ve ADR 0007 sevk kuralının yeniden uygulanmasını ister. Faz 5 tamponuna. |
| **Python vs C++ hız tablosu** (H7 #41'in karşılaştırma kısmı) | Docker (Linux VM) ile Windows yerel ölçüm karşılaştırılamaz. Yalnız motorun kendi gecikmesi ölçülür (§6.4); README'de "H5 60 ms Windows yerel; motor Docker'da; karşılaştırılamaz" notu. |
| Kullanıcı girişi, veritabanı, çoklu kullanıcı, kayıt/oynatma geçmişi | Spec §17 kapsam dışı. |
| Kubernetes, imza zinciri, imaj tarama | Kapalı ağı göstermek için tek compose yeter. |
| Dev Container | VS Code kullanılmıyor; geliştirme döngüsü `docker build --target build` ile. |
| Yeni eğitim, yeni model | "Train only what ships"; model değişmez. |
| Donmuş kare tespiti (Faz 2 devri) | Kullanıcı kararı (2026-10-18): girmez. Faz 4'ün video işleme altyapısıyla; bu fazda yalnız `docs/health.md` sınırlaması. |
| Model ve mod seçici (spec §17, FR-U3 "Mod seçimi") | Sevk edilen sistem tek model; seçici yok. |

**Anti-hedefler (çalışsa bile başarısız sayılır):**
- Motor çıktısı Python referansından tolerans dışında ayrışıyor ve CI bunu yakalamıyor.
- Demo açılırken ya da çalışırken internetten bir şey indiriyor (model, font, CDN, telemetri).
- Temiz veride zarar verdiği ölçülmüş bir kapı **varsayılan olarak** modele veri kestiriyor.
- Motorun p95'i (§6.4 Kabul 5 tanımıyla) ADR 0007 hedefini (100 ms) aşıyor ve bu raporlanmıyor.

## 4. Mimari

```
                       docker compose (tek komut, internetsiz)
┌──────────────────────────────────────────────────────────────────────────┐
│  ui  (Next.js, standalone)              engine  (C++17, ORT CPU)         │
│  ┌─────────────────────────┐   HTTP    ┌──────────────────────────────┐  │
│  │ tarayıcı: oynatıcı,     │──/api/──► │ /v1/detect                   │  │
│  │ canvas, sağlık paneli,  │  (sunucu  │  decode → health → gate      │  │
│  │ arıza simülasyonu       │   proxy)  │  → letterbox → ORT → decode  │  │
│  │ Next sunucusu: kareleri │           │  → calibrate → ByteTrack     │  │
│  │ /data'dan okur          │           │  → JSON (+ timing, health)   │  │
│  └─────────────────────────┘           └──────────────────────────────┘  │
│   ağ: front (port 3000 → localhost)       ağ: back (internal: true)      │
│   volume: ./demo-data → /data (ro)        volume: ./models → /models (ro)│
└──────────────────────────────────────────────────────────────────────────┘
```

- **Motor yalnız iç ağda** (`internal: true`): dışarıya hiç çıkamaz, dışarıdan erişilemez. UI iki ağda:
  `back` (motora erişim) ve `front` (yalnız `127.0.0.1:3000` yayını). `front` normal bir köprü ağı
  olduğundan UI konteyneri **teknik olarak** dışarı çıkabilir; "çıkmıyor" kanıtı §4.5'teki kapalı katmanla.
- **Tarayıcı motoru hiç görmez:** bütün istekler Next.js sunucu rotaları üzerinden (§4.4).
- **Model ve demo verisi imaja gömülmez,** salt okunur volume olarak bağlanır (imajlar küçük kalır,
  model değişince yeniden derleme gerekmez).
- **İki aşama:** *hazırlık* (internetli, bir kez: `scripts/prepare` imajları derler, modeli Release'ten
  sha256 denetimiyle indirir, demo paketini yerine koyar) ve *çalıştırma* (internetsiz: `docker compose up`).
  `pull_policy: never` yalnız çalıştırma aşamasında anlamlı; ilk derleme internet ister (temel imajlar,
  apt, npm, ORT, OpenCV kaynağı) ve bu açıkça yazılır.
- **Taşınabilir paket:** `scripts/prepare --bundle` iki imajı `docker save` ile tek `.tar`'a, model ve demo
  verisini yanına koyar; internetsiz makinede `docker load` + `docker compose up` (README'de üç komut).

### 4.1 C++ motoru (`engine/`)

**Derleme ortamı: yalnız Docker (Linux).** Windows'ta C++ araç zinciri kurulmaz. `docker/engine.Dockerfile`
çok aşamalı: `deps` (OpenCV ve ORT; ayrı katman, önbelleğe alınır) → `build` (GCC, CMake, Eigen,
GoogleTest; `ctest` burada koşar) → `runtime` (yalnız kütüphaneler + uygulamalar). Temel imaj
`ubuntu:24.04` **digest ile** sabit. Geliştirme döngüsü `docker build --target build`. CI aynı Dockerfile'ı
kullanır: tek doğru kaynak.

**Sürümler (eşlik testi kütüphane farkını değil bizim kodumuzu ölçsün):**

| Bileşen | Python referansı | C++ motoru |
|---|---|---|
| ONNX Runtime | `onnxruntime==1.30.0` (`pyproject`'te `>=` yerine `==`) | [`onnxruntime-linux-x64-1.30.0.tgz`](https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0) (hazır paket; sha256 Dockerfile'da) |
| OpenCV | `opencv-python==5.0.0.93` (`uv.lock`) | **5.0.0 kaynaktan**, yalnız `core`, `imgproc`, `imgcodecs`; IPP açık, bayraklar wheel'e benzetilir. Ubuntu 24.04 apt paketi 4.6.0 ([packages.ubuntu.com](https://packages.ubuntu.com/noble/libopencv-dev)): büyük sürüm farkı, kullanılmaz. `cv::getBuildInformation()` ile `cv2.getBuildInformation()` çıktıları `results/engine/opencv_build.json`'a yazılır. |
| ByteTrack | `trackers==2.6.1` (PyPI, Apache-2.0) | o sürümün kodunun uyarlaması (§4.3) |

**Model dosyaları ve sha256:** `configs/engine/models.json` repoda; `e3c.onnx`, `thresholds.json` (Faz 2),
`calibration.json` (Faz 2), `gate_model.json` (§4.2) için ad ve sha256 (değerler Release hazırlanırken
dosyalardan hesaplanır). Motor açılışta `/models` altındaki dosyaları bununla denetler; eksik ya da farklıysa
açılmaz, nedenini loglar. CI ve `scripts/prepare` aynı dosyayla denetler; `/v1/model` sha256'ları gösterir.

Modüller (spec §15.1'in yalın hâli; WBF ve NMS yok):

| Modül | Sorumluluk | Python karşılığı (referans) |
|---|---|---|
| `image_io` | RGB JPG/PNG (BGR uint8) ve IR PNG (`IMREAD_GRAYSCALE`) çözme; boyutlar aynı değilse hata; 4 kanal R, G, B, T birleştirme | **yeni:** `baykus.engine_ref.compose` (mevcut `load_input` 4 kanalı TIFF'ten okur) |
| `preprocess` | letterbox (gain, yuvarlama, dolgu 114), kanal sırası, `/255`, CHW | `baykus.edge.frames.preprocess` (Faz 1'de Ultralytics'le doğrulanmış). 640×512 girdide gain = 1, yeniden boyutlandırma yok. |
| `detector` | ORT oturumu, giriş tipi/şekli modelden, uçtan uca `[1, 300, 6]` çözümleme, geri ölçekleme, kırpma, `w, h ≤ 0` atma, **`score_raw ≥ 0,1` taban süzgeci** | `baykus.edge.frames.postprocess` (süzgeç eşiği parametre), `OrtBackend` |
| `health` | Faz 2 özellikleri (std, Immerkær gürültü, Laplace varyansı, HSV V parlaklık, yarı çözünürlük Sobel + Hanning + faz korelasyonu, float64) ve durum / kapı kuralı + `r_min`, kayma vektörü `(dx, dy)`; oturum başına zaman kuralı penceresi | `baykus.health.features`, `baykus.health.gate` (`r_min` eklenir), yeni `baykus.health.temporal` (§4.2) |
| `calibration` | izotonik tablo (`x`, `y`) ile skor eşleme (`searchsorted` sol, kırpma) | `baykus.health.calibrate.apply_isotonic` |
| `kalman`, `assignment`, `tracker` | ByteTrack (§4.3) | `trackers==2.6.1` |
| `pipeline` | aşama sırası, süzgeç ve sıralama, aşama süreleri (`decode`, `health`, `pre`, `infer`, `post`, `track`, `total`) | **yeni:** `baykus.engine_ref.pipeline` |
| `server` | HTTP API (cpp-httplib), oturumlar, hata yönetimi | — |
| `apps/` | `baykus_server`; `baykus_bench` (§6.4); `baykus_detect` (bir dizi klasöründen tespit JSON'u yazar; eşlik ve M3OT ölçümü için); `baykus_track` (tespit dosyasından takip) | — |

**İşlem sırası (motor ve Python referansında aynı):**
1. Çöz: iki parça zorunlu (model 4 kanallı). "Kamerayı kapat" sıfır görüntü gönderir (Faz 2'deki
   `drop` tanımı), parçayı eksik bırakmaz.
2. Sağlık (her zaman, `gate` modundan bağımsız) → kapı kararı. `gate=auto` ve karar ≠ `none` ise ilgili
   kanal sıfırlanır (`applied=true`).
3. Letterbox → ORT → çözümleme → `score_raw ≥ 0,1` → kalibre skor.
4. Takip (açıksa): taban süzgecinden geçen **bütün** tespitler, **ham** skorla (§4.3). `conf` takipçiyi
   etkilemez; kaydırıcı oynadıkça takip davranışı değişmez.
5. Yanıt: kalibre skoru `≥ conf` olanlar; sıralama (kalibre azalan, ham azalan, çözümleme indeksi artan).
   Faz 2 Kabul 4 bulgusu: eşitlik ham skorla bozulur.

**HTTP API** (spec §15.3'ün güncel hâli):

```
GET  /health      → {"status":"ok","model_loaded":true}
GET  /v1/model    → {"name","mode","precision","input_size":[640,640],"channels":4,"classes":[...],
                     "sha256":{"model","thresholds","calibration","gate_model"},"tracker":{...parametreler}}
POST /v1/detect   multipart: rgb=<jpg/png>, ir=<png>   (ikisi de zorunlu)
     sorgu: ?conf=0.25&track=true&session=<id>&gate=off|advise|auto   (varsayılan advise)
     → {"detections":[{"cls","cls_id","score","score_raw","box":{x1,y1,x2,y2},"track_id"}],
        "health":{"rgb":{...},"ir":{...},
                  "pair":{"misalign_px","dx_px","dy_px","response","misaligned_frame","misaligned"},
                  "gate":"none|ir_off|rgb_off",
                  "reason":"...","applied":false},
        "timing_ms":{"decode","health","pre","infer","post","track","total"}}
POST /v1/session/reset?session=<id>
```

- `score` kalibre skor, `score_raw` modelin skoru. `conf` süzgeci **kalibre** skora uygulanır; pratikte
  kalibre 0,25 ≈ ham 0,2533 (tablo basamağı), yani Faz 1 eşiğine çok yakın.
- `track_id`: takipçinin o tespite verdiği kimlik; olgunlaşmamış iz ya da eşleşmeyen düşük skorlu tespit
  için `null` (referansın `-1`'i).
- `classes` `baykus.classes` sırasından (car, truck, bus, van, freight_car).
- `health` Faz 2'nin sağlık sözlüğüyle aynı alanlar (`docs/plans/F2.md` Görev 3); `gate=off` iken de
  ölçülür ve döner, yalnız `gate` kararı `none` sayılır.
- **Hatalar** (spec §15.3): parça eksik, çözülemeyen görüntü, RGB ve IR boyutu farklı, bilinmeyen `gate`,
  `conf` [0, 1] dışı → 400; parça > 10 MB → 413; iç hata → 500. Motor tek hatalı istekle çökmez.
- **Oturumlar:** takipçi durumu `session` başına; en çok 8 oturum (dolunca en eski çıkar), 10 dk boşta
  kalan silinir; oturum başına kilit. Çıkarım tek işçide (aynı anda tek ORT çağrısı; gecikme ölçümü kararlı).
  `session` verilmezse ve `track=true` ise 400.

### 4.2 Sağlık kapısı: hizasızlıkta kare ve zaman kuralı, varsayılan

Faz 2 bulgusu: temiz karelerdeki yanlış alarmların hepsi "IR hizasız", %80'i karanlık gece; bu karelerde
faz korelasyonu yanıtının medyanı 0,05 (normalde 0,56). Karanlık RGB'de kenar yok, kayma kestirimi anlamsız
bir tepe buluyor. Sorun modelde değil kuralda: yanlış alarm iyi IR'yi kapatıyor, karanlıkta RGB tek başına
zayıf kaldığı için %4 kare 3 puan kaybettiriyor. Yeniden eğitim (fine-tuning) bunu düzeltmez.

**Train keşfi (2026-10-18, laptop; `docs/experiments.md` "F3 · Hizasızlık alarmı: train keşfi"):**
train'in 500 temiz karesinin 25'inde (%5) hizasızlık alarmı var ve iki ayrı grup:
- **(a) anlamsız kestirim, 12 kare:** neredeyse hepsi karanlık, yanıt ≤ 0,08, kayma çoğu 38-248 px.
  Yanıt eşiği bunları temizler.
- **(b) makul görünen kayma, 13 kare:** çoğu aydınlık, yanıt 0,16-0,69, kayma 16-45 px. Büyük olasılıkla
  veri setinin **gerçek** RGB-IR kayması (veri raporu: merkez kayması p99 16,6 px; Faz 2 eşiği 16,16 px tam
  bu kuyrukta; sentetik şiddet 20 px). Tek karelik ölçülerle sentetik 20 px kaymadan ayrılamaz.
- **Karar ağacı denendi, elendi:** Faz 2 özellikleri üzerinde (derinlik 2-3, görüntüye göre gruplu 5 katlı
  çapraz doğrulama) ağaç kaymanın güvenilirliğini değil **bozulma türünü** öğrendi (üst dallar RGB / IR
  gürültü kestirimi: sentetik koşul kestirmesi) ve ≤ %2 temiz alarm ölçütünde `misalign-20/40`'ın yalnız
  ≈ %18'ini yakaladı; yanıt eşiği aynı ölçütte ≈ %80-89. Model değil bilgi eksik.

**Yapı (ölçümden önce sabit):** Faz 2'nin kuralları ve eşikleri aynen kalır. Hizasızlık iki katmanla:

1. **Kare kuralı:** `misaligned_frame` = Faz 2 kuralı (`misalign_px ≥ 16,158`) **ve** `response ≥ r_min`.
   Yanıt eşiği yalnız alarm **kaldırabilir**; bu yüzden val ölçümü Faz 2 dosyalarından kurulabilir.
2. **Zaman kuralı (videoda; oturum varken):** gerçek kayma kareden kareye aynı yönde ve büyüklükte kalır,
   anlamsız kestirim her karede başka yöne zıplar. Son **5** karede `misaligned_frame` olan kareler `S`,
   kayma vektörlerinin bileşen bazında ortancası `v̄`, `C = {i ∈ S : ‖vᵢ − v̄‖ ≤ 3 px}`.
   `misaligned` = `|C| ≥ 3`. Parametreler (5, 3, 3 px) ölçümden önce sabit, ayarlanmaz.
   - Oturum yoksa (tek kare istek, bench, DroneVehicle val) `misaligned` = `misaligned_frame`.
   - Kayıp ve gürültülü kuralları öncelikli ve kare başına (bekleme yok).
   - Kayma başladıktan sonra alarm en erken 3. karede (30 fps'te ≈ 0,1 s); oturum sıfırlanınca pencere boşalır.
   - Bilinen sınır: kayma vektörü sabit kalmayan gerçek arızalar (titreyen bağlantı) zaman kuralını geçmez.
   - Sınır (b) grubu için de geçerli: sabit bir doğal kayma zaman kuralını da geçer. Bu tasarımda
     "bilinen sınır" olarak yazılır; eşik yükseltilmez (doğal ve sentetik kayma 16-20 px'de üst üste).

**Kayma vektörü:** `baykus.health.features.misalign` şu an yalnız büyüklüğü döndürüyor; `(dx, dy)` (tam
çözünürlük px, yarı çözünürlük × 2) da döner ve sağlık sözlüğünün `pair` alanına `dx_px`, `dy_px` girer.
`FEATURES` listesi ve `train_features.csv` biçimi değişmez.

**`r_min` seçimi (train, kullanıcı kararı 1):** `data/health/train_features.csv`'deki Faz 2 özellik
satırlarından. Adaylar train'deki farklı `misalign_response` değerleri (+ 0). Seçilen: Faz 2 kapısının
bütün kuralları kare kuralıyla uygulandığında **train `clean` koşulundaki kapı alarm sayısı × 50 ≤ kare
sayısı** (tam sayı, ≤ %2) olan **en küçük** `r`. Raporlananlar: seçilen `r_min`, train temiz alarm oranı,
train `misalign-20` / `misalign-40`'ta hizasızlık yakalama oranı, karanlık / aydınlık kırılımıyla (seçimi
etkilemez). Keşifte bu ölçüt `r_min ≈ 0,2-0,3`'e düşüyor ve karanlıkta gerçek kaymanın yaklaşık yarısını
kaçırıyor; bedel raporda yazılır.

**Kayıt:** `results/health/gate_model.json` = `{r_min, ölçüt, train_clean_alarm_rate, train_tpr
(karanlık / aydınlık), temporal: {window: 5, min_frames: 3, tol_px: 3}, thresholds_sha256,
train_features_sha256}`. `thresholds.json` değişmez (Faz 2 sonucu). Python'da `baykus.health.gate.assess` /
`check_frame`'e isteğe bağlı `r_min` eklenir (varsayılan `None` = Faz 2 davranışı; Faz 2 sonuçları yeniden
üretilebilir kalır); zaman kuralı yeni `baykus.health.temporal.MisalignWindow` (oturum başına durum).
Motor ikisini aynı JSON'dan okur.

**Ölçüm (val, keşif; `results/health/gate_model_val.json`):** yeni Colab oturumu yok. DroneVehicle tek
karelerden oluştuğu için val'de yalnız **kare kuralı** ölçülür; zaman kuralı M3OT videolarında (§4.7).
Kare kuralı kapıyı yalnız **kaldırabilir**; val sonucu Faz 2 dosyalarından kare birleştirmeyle kurulur:

- **Birleştirme kuralı:** Faz 2 kapılı koşusunda `reason == "IR hizasız"` olan ve `response < r_min` olan
  karelerin tahmini taban koşudan (`robust/<run>_val_<koşul>/`), diğer bütün karelerin tahmini kapılı
  koşudan (`robust/<run>_val_<koşul>_gate/`).
- **Ön koşul (kodda denetlenir):** her koşulda Faz 2 kapısının dokunmadığı karelerde kapılı ve taban
  tahminleri dosyada yazıldıkları hassasiyette **eşit** (kare başına bağımsızlık ve oturumlar arası
  tekrarlanabilirlik). Faz 2'deki 1409 / 1409 kontrolü yalnız temiz koşulda ve ana değerlendirme dosyasıyla
  yapılmıştı; burada 16 koşulun hepsinde ve birleştirmede kullanılan taban dosyasıyla tekrarlanır.
  Tutmayan koşul "birleştirilemedi" diye raporlanır, sayısı verilmez.
- Eşleştirilmiş bootstrap (1000, seed 42) ve Faz 2'nin Kabul 1-3'ü aynı tanımlarla.
- Kural val sonuçlarına bakılarak bulunduğu için **"val'de keşif"**; ayrıca Faz 2'nin kapılı koşuları
  **PyTorch** E3c ile yapıldı (sevk edilen ONNX değil). README ve `docs/system.md`: "val'de keşif, PyTorch
  E3c: temiz zarar X, fayda Y". Test kullanılmaz.
- **Bilinen risk:** gece gerçek kaymada da yanıt düşük; kare kuralı o zaman gerçek alarmı da iptal eder.
  Val'deki `misalign-20/40` sonucu bunu gösterir (karanlık / aydınlık kırılımıyla).

**Ürün varsayılanı:** `gate=advise` (sağlık ölçülür, uyarı gösterilir, piksel kesilmez). `auto` kullanıcı
seçimi. Keşif ölçümünde yeni kapı Faz 2'nin Kabul 1 ve 3'ünü geçse bile varsayılan değişmez
(testte doğrulanmadı).

### 4.3 Takip

**Python referansı:** `trackers==2.6.1` ByteTrack ([kaynak](https://github.com/roboflow/trackers)).
Makaledeki ByteTrack'ten farklı bir algoritma; C++ bu sürümün kodunu izler (2026-10-18'de okunan
ayrıntılar, uygulama başında sabit sürümde yeniden doğrulanır):

- **Kalman** ([`kalman.py`](https://raw.githubusercontent.com/roboflow/trackers/main/trackers/core/bytetrack/kalman.py)):
  SORT tarzı, durum `(x1, y1, x2, y2, vx1, vy1, vx2, vy2)`, sabit hız, **float32**; Q = 0,01·I,
  R = 0,1·I, P₀ = I; kazanç `np.linalg.inv(S)`. C++ aynı tipte ve aynı işlem sırasıyla (Eigen `float`).
- **Eşleşme** ([`tracker.py`](https://raw.githubusercontent.com/roboflow/trackers/main/trackers/core/bytetrack/tracker.py)):
  ayrı kayıp iz havuzu yok. 1. aşama: yüksek skorlu tespitler (`≥ high_conf_det_threshold`) bütün izlerle;
  2. aşama: düşük skorlular kalan izlerle. Maliyet IoU; `scipy.optimize.linear_sum_assignment(maximize=True)`
  **önce** çözülür, `IoU ≥ minimum_iou_threshold` **sonra** uygulanır.
- **Atama çözücüsü:** SciPy'nin `rectangular_lsap` C kodunun (BSD-3) C++ uyarlaması: eşit maliyetlerde aynı
  sonucu vermesi için kendi Macar'ımız ya da LAPJV değil. Lisans notu `engine/third_party/`'de.
- **İz yaşam döngüsü:** yeni iz eşleşmeyen yüksek skorlu tespitten (`≥ track_activation_threshold`);
  kimlik `minimum_consecutive_frames` güncellemeden sonra verilir, öncesi `-1`; `lost_track_buffer`
  (`frame_rate`'e göre ölçekli) kadar eşleşmeyen iz silinir. Kimlik sayacı oturum başına (referansta
  sınıf düzeyinde ortak; Python koşucusu her dizi başında `reset()` çağırır).
- **Çıktı:** dönen kutu tespit kutusu (Kalman durumu değil); çıktı sırası girdi sırasından farklı olabilir.
  C++ girdi indeksini taşır; Python koşucusu çıktıyı girdiye kutu eşitliğiyle (kopya olduğu için birebir) eşler.
- **Parametreler:** referansın varsayılanları (`lost_track_buffer=30`, `frame_rate=30`,
  `track_activation_threshold=0.7`, `minimum_consecutive_frames=2`, `minimum_iou_threshold=0.1`,
  `high_conf_det_threshold=0.6`); `configs/engine/tracker.yaml`'da sabit. Eşikler **ham** skora uygulanır.
  M3OT'un kare hızı doğrulanmadı; `frame_rate=30` varsayım olarak yazılır.

**Girdi:** taban süzgecinden (`score_raw ≥ 0,1`) geçen bütün tespitler, sınıflar birleştirilmiş (M3OT tek
sınıf `vehicle`), ek NMS yok. Uçtan uca çıktı sınıflı olduğundan aynı araca iki sınıftan kutu düşebilir;
oranı raporlanır (aynı karede IoU ≥ 0,7 olan farklı sınıf çiftleri).

**Eşlik fikstürü:** `baykus_detect` ile M3OT test 960 karesinden üretilen tespitler
`fixtures/tracks/m3ot_test_dets.json.gz` olarak **repoya** girer (birkaç MB; M3OT CC BY 4.0). C++
`baykus_track` ve Python koşucusu aynı dosyayı okur. Takip eşliği böylece CI'da koşar.

**Ölçüm (M3OT test, 2 dizi × 480 kare, `vehicle`, IR etiketleri;** `results/track/m3ot_test_{oracle,engine}.json`**):**
HOTA, MOTA, IDF1 `trackers eval` ile (aynı sabit paket; TrackEval gerekmez). İki satır: (a) **oracle**
— M3OT'un kendi kutuları skor 1 ile (takipçinin modelden bağımsız ölçümü), (b) **motor tespitleri**.
Dizi başına ve iki dizinin birleşiği; bootstrap yok (iki dizi). Makaledeki ByteTrack sayılarıyla yan yana
konmaz (farklı dedektör ve algoritma).

**Motor tespitlerinin M3OT doğruluğu (kullanıcı kararı 5; `results/engine/m3ot_test_detect.json`):**
AP50 (pycocotools, `maxDets ≥ 1000`, sınıfsız) ve ECE (15 aralık, ham ≥ 0,25). Faz 1'in M3OT AP50'si
(0,889) PyTorch + `agnostic_nms` ile ölçüldü; yan yana konur, fark yorumlanır. Kalibrasyon tablosu
DroneVehicle val'den öğrenildi; M3OT ECE'si onun alan dışı davranışını gösterir. Eşik yok, seçim yok.

### 4.4 Operatör ekranı (`ui/`)

Tek sayfa, spec §17 düzeni + Faz 3 ekleri. Kullanıcının günlük yığını: Next.js (App Router) + TypeScript
+ Tailwind + shadcn/ui; tam sürümler `package-lock.json` ile kilitli; fontlar repoda (OFL) ve
`next/font/local`, CDN yok; `NEXT_TELEMETRY_DISABLED=1` hem build'de hem çalışmada.

**Demo verisi düzeni** (demo paketi CLI'ı üretir; mevcut dönüştürücü çıktısı düz `<dizi>_<kare>.jpg`):

```
/data/sequences/<dizi>/rgb/<kare:06d>.jpg
/data/sequences/<dizi>/ir/<kare:06d>.png
/data/sequences/<dizi>/index.json   {"name","frames":[...],"fps":30,"size":[640,512],"source":"M3OT test, CC BY 4.0"}
```

**Next sunucu rotaları:** `GET /api/sequences` (dizi listesi), `GET /api/frame/<dizi>/<kare>/<rgb|ir>`
(görüntü baytları), `POST /api/detect` (gövde: `{dizi, kare, conf, track, gate, session, fault}`).

**Veri akışı kuralları:**
- **Arıza yokken** Next sunucusu diskteki **orijinal baytları** motora iletir; tarayıcı yalnız kare
  numarasını gönderir (tarayıcı JPEG çözümü ve yeniden kodlama pikselleri Python referansından uzaklaştırmasın).
- **Arıza açıkken** tarayıcı canvas'ta bozar ve bozulmuş kareyi PNG olarak gönderir (JPEG yeniden sıkıştırması
  ek bozulma eklemesin); Next sunucusu motora iletir. Tanımlar spec §12.3 ve
  `baykus.robustness.corruptions` ile aynı (karartma çarpanı, IR gürültü σ, IR kaydırma px ve dolgu değeri,
  bulanıklık çekirdeği, kamerayı kapat = sıfır görüntü); rastgele sayılar numpy ile aynı değil (demo, ölçüm değil).
- **Akış kontrolü:** aynı anda tek istek; sonraki kare yanıt gelince istenir. Hız ayarı bir üst sınırdır;
  kare **atlanmaz** (takipçi her istekte bir kare ilerlediğini varsayar).
- **Oturum:** sayfa açılışında UUID; dizi değişince, ileri atlama ya da geri adımda `session/reset`.

**Bileşenler:**
- **Oynatıcı:** oynat / durdur / kare kare / hız (FR-U1). RGB / IR / yan yana; kutu, sınıf, kalibre skor,
  iz numarası (FR-U2).
- **Üst çubuk:** conf kaydırıcısı, takip aç/kapa, kapı modu (`kapalı / uyar / otomatik`), oturum sıfırla.
- **Sağ panel:** tespit listesi; canlı gecikme grafiği (son 100 kare, p50 / p95, aşama kırılımı; düz canvas
  ya da hafif bir kütüphane, yerelde paketli) (FR-U3).
- **Sensör sağlık paneli:** her kamera için durum rozeti (`tamam / kayıp / gürültülü / karanlık / bulanık`)
  ve çift için `hizasız`; son 50 karenin durum şeridi; ölçülen değerler (gürültü σ, kayma px, yanıt).
- **Alarm:** `kayıp` ya da `gürültülü` **son 5 karenin en az 3'ünde** varsa alarm (tek karelik titremeyi
  süzer; sabit, yapılandırılmaz). `hizasız` alarmı doğrudan motorun `pair.misaligned`'ı (zaman kuralı
  zaten 5 karenin 3'ünü ve yön tutarlılığını istiyor; ikinci bir pencere eklenmez). `karanlık` ve `bulanık` yalnız bilgi rozeti.
  Alarm metni Faz 2'nin `reason`'ı ("IR hizasız" gibi). Kapının ne yaptığı ayrı bir rozette ve **her karenin
  `applied` alanından**: "IR modele verilmiyor" (yalnız `applied=true` olan karede) / "yalnız uyarı".
  Alarm temizlenince geçmiş listesinde kalır (oturum boyunca, bellekte).
- **Skor açıklaması:** "kalibre skor: DroneVehicle val, iki kamera açık; M3OT'ta ve kapalı kanalda
  doğrulanmadı" (bilgi simgesinde).
- **Atıf:** alt bilgide M3OT atfı ve lisans bağlantısı (CC BY 4.0).
- **Kapsam dışı:** giriş, veritabanı, video yükleme (diziler klasörden), çoklu oturum yönetimi.

### 4.5 Kapalı ağ ve CI

- `docker-compose.yml`: `engine` (yalnız `back`, `internal: true`), `ui` (`back` + `front`, port
  `127.0.0.1:3000:3000`). `pull_policy: never`.
- `compose.offline.yml` (kanıt katmanı): **bütün ağlar `internal: true`**, port yayını yok. Docker'ın
  belgesine göre `internal` yalnız o ağı yalıtır; dışa açık bir ağa da bağlı servis oradan çıkabilir
  ([networks](https://docs.docker.com/reference/compose-file/networks/)). Bu katmanda dışa açık ağ kalmaz.
- **NFR-1 kanıtı (otomatik, CI'da):**
  1. `docker compose -f docker-compose.yml -f compose.offline.yml up` sonrası `offline_check` (Node betiği,
     `ui` imajında, `docker compose exec ui`) bir kareyi `/api/detect` üzerinden uçtan uca işletir ve
     `/v1/model` sha256'larını `models.json` ile karşılaştırır.
  2. Aynı katmanda `ui` ve `engine` konteynerinden dış adrese (ör. `1.1.1.1:443`) bağlantı denemesi
     **başarısız** olmalı.
  3. İkincil: `.next/` içinde `http(s)://` taraması **izin listesiyle** (React hata bağlantıları,
     `http://www.w3.org/2000/svg` gibi ad alanları); listede olmayan bir kaynak varsa başarısız.
  Ultralytics ya da Python çalışma zamanında yok.
- **Elle kanıt (en güçlü):** laptopta Wi-Fi kapalıyken normal `docker compose up`; demo açılıyor ve
  oynatılıyor (deney günlüğüne tarih ve ekran görüntüsü).
- **CI (`.github/workflows/ci.yml`):** `python` (uv, ruff, pytest — mevcut), `engine` (Dockerfile `build`
  hedefi: CMake, ctest, eşlik testleri; buildx önbelleği `type=gha`, OpenCV katmanı her commit'te yeniden
  derlenmesin), `ui` (npm ci, eslint, `next build`, tarama), `compose` (imajlar, kanıt 1-2). Model CI'da
  GitHub Release'ten indirilir (`models.json` ile sha256 denetimli). Takip fikstürü repoda (§4.3).
- **Kod kalitesi (NFR-4):** `clang-format` (CI'da denetim), pre-commit kancası; `ctest` bir kez ASan/UBSan
  ile (CI'da ayrı yapılandırma).

### 4.6 Model, fikstürler, demo verisi

- **Model:** E3c · ONNX FP32 (38 MB) + `thresholds.json` + `calibration.json` + `gate_model.json`, bir
  GitHub Release'e (`models-v1`) eklenir; AGPL ağırlık yayımı spec §7.6 ile uyumlu. CI ve kurulum oradan
  indirir; sha256'lar `configs/engine/models.json`'da.
- **Eşlik fikstürleri:** DroneVehicle'ın lisansı yeniden dağıtıma izin vermiyor (spec §7.6) → repoya
  yalnız **M3OT** kareleri girer: her test dizisinden eşit aralıklı 5 RGB-IR çifti (toplam 10), sabit liste.
  Ek olarak sentetik durumlar (siyah IR, gürültülü IR σ 20, 20 px kaydırılmış IR, siyah RGB; Faz 2 bozulma
  fonksiyonlarıyla Python'da üretilip **dosya olarak** kaydedilir, C++ rastgele sayı üretmez).
- **Fikstür kuralı (sınır uzaklığı; üretici denetler):** her sağlık özelliği her eşikten (ve `r_min`'den),
  her ham skor `conf` (kalibre 0,25'in ham karşılığı) ve kalibrasyon blok sınırlarından ile taban süzgecinden
  (0,1) **en az 10 × tolerans** uzak. Tutmayan kare seçilmez, bir sonraki eşit aralıklı kareye geçilir;
  sentetik durumlarda değer eşikten zaten uzaktır (denetlenir).
- **Beklenen çıktılar** Python'dan (`fixtures/expected/*.json`): tespitler (ham, kalibre), ham ORT çıktısı
  `[1, 300, 6]` (fikstür başına 7 KB), sağlık sözlüğü ve letterbox tensörünün **sha256**'sı (tensör repoya
  girmez; 4 × 640 × 640 float32 = 6,5 MB).
- **Atıf:** `fixtures/m3ot/NOTICE.md` (kaynak, CC BY 4.0 bağlantısı, değişiklikler: PNG → JPEG, IR tek
  kanal, sentetik bozulmalar). Repo AGPL olsa da fikstürler CC BY kalır; README'de yazılır. Zenodo'daki CC0
  kaydı doğrulanmadı; daha katı olan CC BY esas.
- **Demo verisi:** M3OT test dizileri `1-03` ve `2-03` (`demo-data/`, git'e girmez). Boyut bilinen:
  RGB 128 MB + IR 180 MB (`docs/data/m3ot.md`). Hazırlık tek bir **Colab CPU** oturumu (adım 1): Drive'daki
  `m3ot_test.zip` → mevcut `baykus-data m3ot` → demo paketi CLI'ı (§4.4 düzeni) → laptop paketi
  (`du -sh` ile görülüp indirilir). GPU yok.

### 4.7 M3OT'ta sağlık kontrolü (Faz 2 sözü; kullanıcı kararları 3 ve 5)

`docs/health.md` Sınırlamalar ve spec §23: "Faz 3'te M3OT dizilerinde niteliksel kontrol". M3OT'ta etiketler
arasında doğal kayma var (dy ortanca 5,5 px, std 10,8; uçlar −103 / +54) ve eşikler DroneVehicle'dan
seçildi; demo hikâyesi bu yüzden ölçülmeden yazılmaz.

- **Ölçüm (laptop CPU, Python referans pipeline'ı, ONNX; `results/health/m3ot_health.json`):** M3OT test
  960 çiftinde, kareler dizi sırasıyla (oturum her dizi başında sıfırlanır). Koşullar (Faz 2 bozulma
  fonksiyonları, seed 42): `clean`, `misalign-20`, `misalign-20@240` (kayma 240. karede başlar), `noise-20`,
  `drop_ir`. **Video koşullarında kayma yönü dizi başına sabit** (seed dizi adından; Faz 2'deki gibi kare
  başına değil): gerçek bir kamera kayması böyle davranır. Üç kural yan yana: Faz 2 kuralı, kare kuralı,
  kare + zaman kuralı:
  1. kare başına hizasızlık ve diğer durum oranları; alarm oranı (§4.4 tanımıyla);
  2. `misalign-20@240`'ta gecikme: kaymanın başladığı kareden ilk alarma kadar kare sayısı;
  3. `gate=auto` ile AP50 (IR etiketleri, sınıfsız) — kapının M3OT'ta bağımsız denenmesi; kapısız AP50
     ile fark, dizi düzeyinde (bootstrap yok).
- **Demo ölçütü (ölçümden önce sabit; kare + zaman kuralı, `gate=advise`):** `clean`'de alarm oranı
  **≤ %2** kare, `misalign-20`'de **≥ %90** kare, `misalign-20@240`'ta gecikme **≤ 10 kare**.
- **Kurallar:** test dizisi olduğu için sonuç **yalnız raporlanır**; hiçbir eşik ya da zaman kuralı parametresi ona bakılarak
  değişmez. Demo ölçütü tutmazsa: README ve `docs/system.md`'ye sınırlama notu (sayılarla) ve demo için
  M3OT'un **test dışı** (train / val) bir dizisi; dönüştürülmesi ek bir Colab CPU oturumu ister ve aynı
  ölçüm orada da raporlanır.

## 5. Faz 2'den devreden girdiler

| Girdi | Bu tasarımda |
|---|---|
| Karanlıkta hizasızlık yanlış alarmı | §4.2 kare kuralı (yanıt eşiği, train'de ≤ %2 temiz alarm ölçütüyle; val'de keşif) ve zaman kuralı (videoda; M3OT'ta ölçülür); ürün varsayılanı `advise` |
| Kalibrasyonda eşitlik | §4.1: sıralama (kalibre, ham, indeks); motor `score` + `score_raw` |
| Dedektör hızı (laptop 11,3 ms; hedef < 5 ms) | §6.4: motorda `health` aşaması ayrı ölçülür; p95 bütçesine dahil |
| Donmuş kare | Hedef dışı (§3, kullanıcı kararı 5); `docs/health.md` sınırlaması |
| Eşik seçiminin belleği (`searchsorted`) | Bu fazda gerekmiyor: `r_min` tek parametreli bir tarama, `t` sabit; Faz 2 eşikleri yeniden seçilmez |
| M3OT'ta niteliksel kontrol (`docs/health.md` Sınırlamalar, spec §23) | §4.7 ölçümü ve demo ölçütü |

## 6. Değerlendirme ve kabul ölçütleri (ölçümden önce sabit)

Her sayı bir JSON'a bağlı; `docs/system.md`, `docs/health.md` gibi bu dosyalardan üretilir
(`baykus-system report`), elle düzenlenmez.

| Sonuç dosyası | İçerik |
|---|---|
| `results/engine/parity_detect.json`, `parity_health.json`, `parity_track.json` | Kabul 1-4 (CI çıktısı da aynı biçimde) |
| `results/engine/opencv_build.json` | iki tarafın OpenCV derleme bilgisi |
| `results/engine/m3ot_test_detect.json` | motor tespitleriyle M3OT AP50 + ECE |
| `results/bench/e3c_onnx_fp32_engine_laptop-docker.json` | Kabul 5 |
| `results/track/m3ot_test_oracle.json`, `m3ot_test_engine.json` | takip ölçümü |
| `results/health/gate_model.json`, `gate_model_val.json`, `m3ot_health.json` | `r_min` ve zaman kuralı, keşif, M3OT kontrolü |

### 6.1 Eşlik (S7, FR-E6) — CI'da, her commit'te
- **Kabul 1 — tespit:** M3OT + sentetik fikstürlerde C++ ile Python referansı:
  (a) ham ORT çıktısı `[1, 300, 6]` bütün satırlarda mutlak fark ≤ 1e-4;
  (b) taban süzgecinden geçen kutu sayısı aynı, eşleşen kutularda IoU ≥ 0,99, ham skor farkı ≤ 1e-3
  (H5'teki tolerans), sıralama aynı;
  (c) kalibrasyon ayrı birim testle: C++ kalibre skoru = tablo(C++ ham skoru), blok sınırları dahil
  **birebir**. Kalibre skor pipeline düzeyinde ham skordan karşılaştırılmaz (basamaklı fonksiyon: ham
  0,2533'te 0,231 → 0,282 sıçrar).
- **Kabul 2 — letterbox:** fikstürlerde C++ tensörünün baytları Python tensörünün sha256'sıyla aynı.
- **Kabul 3 — sağlık:** her fikstürde durumlar, kapı kararı ve `reason` **aynı**; `std` / `noise` / `blur` /
  `brightness` göreli fark ≤ 1e-4; `misalign_px`, `dx_px`, `dy_px` mutlak fark ≤ 0,5 px, `response`
  mutlak fark ≤ 0,02. **Zaman kuralı** ayrı birim testle: aynı `pair` dizisinde (Python'da üretilmiş
  sentetik 50 karelik diziler: sabit kayma, zıplayan kestirim, ortada başlayan kayma, sıfırlama) C++ ve
  Python her karede aynı `misaligned`.
  OpenCV sürümü iki tarafta aynı (5.0.0); tolerans laptop (Zen 4, AVX-512) ile CI işlemcisi arasındaki
  SIMD farkı için. Fikstür kuralı (§4.6) eşiğe yakın kareyi dışarıda tuttuğundan durum eşitliği toleransla çelişmez.
- **Kabul 4 — takip (FR-E4):** `fixtures/tracks/m3ot_test_dets.json.gz` üzerinde C++ ve Python takipçisi.
  **Tanım:** tespitler girdi indeksiyle eşlenir; `-1` (`null`) yalnız `-1` ile eşleşir; dizi başına tek bir
  C++ → Python kimlik eşlemesi vardır ve birebirdir. Ölçüt: bu eşlemeyle **farklı kare sayısı 0**.
  **İstisna (kullanıcı kararı 6, ölçümden önce):** farklı her kare tek tek bir kayan nokta eşitliğine
  bağlanıp `parity_track.json`'da belgelenirse (`|IoU − 0,1| < 1e-5`, eşik sınırında skor ya da atamada
  eşit maliyet; ilk farklı karede ve sonraki zincirleme farklar o kareye bağlı sayılır) Kabul 4 ✅
  "istisnayla" yazılır. Bağlanamayan bir fark varsa Kabul 4 ❌.

### 6.2 Kare kuralıyla kapı (val, keşif; §4.2)
Faz 2 tablosunun aynısı (16 koşul, E3c vs E3c + kapı, eşleştirilmiş bootstrap 1000, seed 42) yeni kuralla,
kare birleştirmeyle. Faz 2'nin Kabul 1-3 tanımlarıyla değerlendirilir; sonuç **keşif** olarak raporlanır,
Faz 3'ün kabul ölçütü değildir ve ürün varsayılanını değiştirmez. Gereken dosyalar Drive'da: 16 val
koşulunun `robust/<run>_val_<koşul>_gate/` (sağlık kaydı + tahminler) ve taban tahminleri
`robust/<run>_val_<koşul>/` (tahmin ≈ 70 MB; kullanıcı adımı olarak adım 3'te indirilir).

### 6.3 Takip ve M3OT doğruluğu (M3OT test)
HOTA, MOTA, IDF1; oracle ve motor tespitleri; dizi başına ve iki dizinin birleşiği. Motor tespitleriyle
AP50 ve ECE. Eşik yok (ilk ölçüm); sonuç ADR 0008 (takip değerlendirmesi) ile kayda geçer.

### 6.4 Gecikme ve bellek (NFR-5, NFR-6)
`baykus_bench`, H5 protokolü (§14: 50 ısınma + 500 ölçüm, 20 bench görüntüsü — laptopta zaten var),
motor konteynerinde, laptop prize takılı. Girdi **bellekteki kodlanmış baytlar** (JPG + PNG; çözme dahil,
disk okuma hariç); `gate=advise`, takip açık (tek oturum, 20 kare döngüde); ORT iş parçacığı sayısı H5
gibi varsayılan ve kayıtta. Kayıtta ayrıca `nproc`, `.wslconfig` (CPU, bellek), Docker Desktop sürümü.
Aşama başına p50 / p95 ve tepe RSS.
**Kabul 5:** motor p95 (`decode` + `health` + `pre` + `infer` + `post` + `track`, HTTP hariç) ≤ **100 ms**.
ADR 0007'nin ölçüsünden (pre + infer + post) daha geniş bir tanım; Docker (WSL2) ölçümü olduğu yazılır.
HTTP dahil p95 ayrıca raporlanır (eşik yok), `back` ağındaki bir istemciden (Windows port yönlendirmesi
ölçüme girmesin).
*Bütçe tahmini (ölçülmedi):* H5 p95 59,9 + sağlık ≈ 11 + çözme ≈ 5 → ≈ 75-80 ms.

### 6.5 Kapalı ağ (S8, NFR-1)
**Kabul 6:** §4.5'in otomatik kanıtları (1-3) CI'da yeşil + elle Wi-Fi kapalı demo kaydı.

### 6.6 Ekran ve sağlamlık
- **Kabul 7 — ekran:** FR-U1-U4, sağlık paneli, alarm ve kapı rozeti çalışıyor; her biri ekran görüntüsü ya
  da GIF ile deney günlüğünde (M3OT demo dizisinde, arıza simülasyonu açık / kapalı).
- **Kabul 8 — API sağlamlığı (CI):** eksik parça → 400, farklı boyut → 400, bozuk PNG → 400, sınır üstü →
  413; ardından 1000 rastgele baytlık istek ve `/health` hâlâ `ok`.
- **Demo ölçütü** (§4.7): raporlanır; tutmazsa §4.7'deki yol. Fazın kabulünü engellemez.

### 6.7 Faz çıkışı
S7, S8 işaretli; `docs/system.md` (mimari, API, eşlik ve gecikme tabloları, takip ve M3OT tabloları, keşif
kapı tablosu, M3OT sağlık kontrolü); README'ye sistem bölümü, ekran görüntüsü ve internetsiz kurulum
(üç komut); `v0.2.0`.

## 7. İş yükü ve takvim

Bütün kodu Claude yazar (kullanıcı kararı 4); kullanıcı adımları: Colab CPU oturumu, Drive'dan indirmeler,
Docker Desktop ayarı, Wi-Fi kapalı elle test, onaylar. Önerilen sıra (her adım bir öncekini kullanır;
plan haftalara böler):

| # | İş | Tahmin |
|---|---|---|
| 1 | **Colab CPU** (M3OT demo paketi + fikstür kareleri) · motor iskeleti: Dockerfile (`deps` / `build` / `runtime`, OpenCV 5.0.0 kaynaktan), CMake, GoogleTest, CI `engine` işi ve önbellek · `models.json`, Release `models-v1` | 3-4 gün |
| 2 | **Python referansı:** `baykus.engine_ref` (birleştirme, pipeline, süzgeç ve sıralama), fikstür üretici (sınır kuralıyla), sürüm sabitleme | 2-3 gün |
| 3 | `preprocess` + `detector` + `calibration` C++ + Kabul 1-2 | 3-4 gün |
| 4 | `health` C++ + Kabul 3 · `r_min` seçimi, kayma vektörü, zaman kuralı (Python + C++), `gate_model.json`, Python `gate` değişikliği · keşif ölçümü (laptop; Drive'dan ≈ 70 MB) · §4.7 M3OT sağlık kontrolü | 4-5 gün |
| 5 | `baykus_detect` + takip fikstürü + M3OT AP50 / ECE · Python takip koşucusu · Kalman, `rectangular_lsap` uyarlaması, ByteTrack + Kabul 4 (**zaman kutusu 2 gün**; aşılırsa farklar listelenir ve Kabul 4 ❌ ya da istisnayla raporlanır, iş durdurulmaz) · M3OT takip ölçümü, ADR 0008 | 5-6 gün |
| 6 | HTTP API, oturumlar, hatalar + Kabul 8 · compose iskeleti ve `compose.offline.yml` (kanıt 1-2 erken) · `baykus_bench` + Kabul 5 | 3-4 gün |
| 7 | UI: rotalar, oynatıcı, kutular, sağlık paneli, alarm, arıza simülasyonu, gecikme grafiği + Kabul 7 | 5-7 gün |
| 8 | CI `ui` + `compose` işleri, tarama, `scripts/prepare --bundle`, Wi-Fi kapalı elle test (Kabul 6) | 2-3 gün |
| 9 | `baykus-system report` → `docs/system.md`, README, spec v4.5, `v0.2.0` | 1-2 gün |

**Toplam ≈ 6-7 hafta** (spec'te 8 hafta, 7 Ara – 31 Oca; takvimin ≈ 7 hafta önündeyiz; Ocak final dönemi
için tampon kalıyor, ADR 0006). En riskli adımlar: 5 (takip eşliği: eşit maliyetler, float32 Kalman) ve
1 (OpenCV derlemesi, CI süresi). GPU yok; Colab yalnız CPU (adım 1; §4.7 tutmazsa bir oturum daha).
Laptop'a inecekler: M3OT demo paketi (≈ 310 MB, boyut önce görülür), Faz 2'nin val tahmin ve sağlık
dosyaları (≈ 70 MB).

## 8. Riskler

| Risk | Önlem |
|---|---|
| Windows'ta C++ araç zinciri | Hiç kurulmaz: derleme ve test yalnız Docker'da; CI aynı Dockerfile. |
| OpenCV'yi kaynaktan derlemek CI'ı yavaşlatır | Ayrı `deps` katmanı, buildx `type=gha` önbelleği, yalnız üç modül. |
| OpenCV aynı sürümde bile işlemciye göre farklı sonuç (SIMD) | Kabul 3 toleransları + fikstür sınır kuralı; derleme bilgisi kayıtta. |
| `trackers` ByteTrack ayrıntıları ve eşit maliyetli atamalar → sıfır fark tutmaz | SciPy çözücüsünün uyarlaması, float32 Kalman; zaman kutusu ve önceden yazılmış istisna kuralı (§6.1). |
| M3OT'ta doğal kayma ya da gece kareleri → temiz karede alarm, demo bozulur | §4.7 ölçümü önce; tutmazsa sınırlama + test dışı dizi. |
| Docker Desktop (WSL2) gecikmesi yerel ölçümden kötü → Kabul 5 kalır | Olumsuz sonuç raporlanır; aşama kırılımı neyin pahalı olduğunu gösterir (ör. `health` 11 ms). Bütçe aşılırsa `health` her N karede bir çalıştırılabilir (yeni ölçüm gerektirir, bu fazda değil). |
| `C:\` bağlamaları Docker Desktop'ta yavaş → UI kare okuması takılır | Kabul 5'i etkilemez (bench bellekten). Gerekirse demo verisi bir kez adlandırılmış volume'a kopyalanır. |
| UI kapsamı büyür | Tek sayfa, §4.4 listesi dışında özellik yok; yeni istek Faz 5'e. |
| M3OT tek sınıf ve küçük kutular; DroneVehicle'dan farklı → takip sayıları düşük çıkar | Oracle satırı takipçiyi modelden ayırır; mutlak değer değil iki satırın farkı yorumlanır; motor AP50 yanında. |
| Yeni kapı keşifte de temiz veride zararlı ya da gece gerçek kaymayı kaçırıyor | Varsayılan zaten `advise`; sonuç olduğu gibi raporlanır. |
| Zaman kuralı sabit doğal kaymayı da "hizasız" sayar; titreyen gerçek arızayı kaçırır | Bilinen sınır olarak yazılır; M3OT ölçümü (§4.7) temiz videoda ne kadar alarm verdiğini gösterir; parametreler ayarlanmaz. |

## 9. Builder'a bırakılanlar (varsayılanlarıyla; uygulamada kararlaştırılır, plana yazılır)

- Ubuntu temel imajının digest'i; ORT paketi ve OpenCV kaynağının sha256'sı (Dockerfile'da).
- OpenCV CMake bayraklarının ayrıntısı (varsayılan: wheel'in `getBuildInformation`'ına en yakın).
- UI'nin grafik bileşeni (varsayılan: düz canvas ya da uPlot, yerelde paketli), canvas ayrıntıları, renkler.
- Fikstür karelerinin sabit listesi (varsayılan: dizi başına eşit aralıklı 5, sınır kuralıyla kaydırılmış;
  liste gerekçesiyle `fixtures/m3ot/README.md`'de).
- Oturum sınırları ve parça boyutu (varsayılan: 8 oturum, 10 dk, 10 MB).
- Next.js ve paket sürümleri (`package-lock.json`).

## 10. Spec'e yansıyacaklar (onaydan sonra, v4.5)

§2.3 S7/S8 metni (eşlik tanımları §6.1), §7.4 FR-E2/E3 (late_fusion ve NMS yolu Faz 3'te yok), FR-E4
(referans `trackers==2.6.1`), FR-E5 API (§4.1), FR-U eklemeleri (sağlık paneli, alarm, kapı modu) ve FR-U3
"mod seçimi"nin düşmesi, §15.1 modül listesi, §15.3 API, §15.4 madde 4 (CI'da küçük `n` modeli değil E3c;
E8-n eğitilmedi), §16 (Python referansı `model.track` değil `trackers`), §17 düzen (model/mod seçici yok),
§18 compose ağları, CI işleri (OpenCV apt değil kaynaktan; `compose` işi) ve NFR-1 kanıtları, §22 Faz 3
satırı (≈ 6-7 hafta) ve H6-H9 tablolarının bu tasarıma göre güncellenmesi (H6 #36 letterbox ve #37'deki
(S) işaretleri kalkar, kullanıcı kararı 4), §23 risk satırları.
