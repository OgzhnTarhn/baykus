# Faz 3 Tasarımı — Sistem (yalın)

> **Tarih:** 2026-10-17 (taslak) · 2026-10-18 (inceleme bulguları ve kullanıcı kararları işlendi; ikinci
> bağımsız inceleme işlendi) · **Durum:** v3, onaylı (2026-10-18; plan `docs/plans/F3.md`) · **Dal:** `f3`
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
3. M3OT **test** dizileri yalnız son ölçüm ve raporlama için; hiçbir eşik, fikstür ya da demo seçimi onlara
   bakılarak yapılmaz. Geliştirme, CI fikstürleri, demo dizisi ve demo ölçütü M3OT **val** (val yoksa
   train) dizilerinden (§4.6, §4.7). (İlk hâli "demo test dizisinde, tutmazsa test dışı dizi" idi; ikinci
   incelemede test sonucuna bağlı sunum seçimi olduğu için daraltıldı.)
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
| `detector` | ORT oturumu, giriş tipi/şekli modelden, uçtan uca `[1, 300, 6]` çözümleme, geri ölçekleme, kırpma, `w, h ≤ 0` atma, **taban süzgeci** (parametre; aşağıdaki çıktı katmanları) | `baykus.edge.frames.postprocess` (süzgeç eşiği parametre), `OrtBackend` |
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
3. Letterbox → ORT → çözümleme → taban süzgeci (sunucuda 0,1; aşağıdaki tablo) → kalibre skor.
4. Takip (açıksa): taban süzgecinden geçen **bütün** tespitler, **ham** skorla (§4.3). `conf` takipçiyi
   etkilemez; kaydırıcı oynadıkça takip davranışı değişmez.
5. Yanıt: kalibre skoru `≥ conf` olanlar; sıralama (kalibre azalan, ham azalan, çözümleme indeksi artan).
   Faz 2 Kabul 4 bulgusu: eşitlik ham skorla bozulur.

**Çıktı katmanları (hangi ölçüm hangisini kullanır):**

| Katman | Süzgeç | Kullanan |
|---|---|---|
| Değerlendirme | `score_raw ≥ 0,001` (spec FR-V9; 300 satırın hepsi pratikte) | `baykus_detect` dosyası → M3OT AP50 / ECE; Kabul 1 (b) bu katmanda |
| Takip girdisi | `score_raw ≥ 0,1` | takipçi (motor ve `baykus_track`); takip fikstürü değerlendirme dosyasından bu süzgeçle |
| Gösterim | kalibre `≥ conf` | `/v1/detect` yanıtı, UI |

AP50 hiçbir zaman API yanıtından (conf süzgeçli) hesaplanmaz.

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

1. **Kare kuralı:** `misaligned_frame` = Faz 2 kuralı (`misalign_px ≥ thresholds.misaligned`, Faz 2
   JSON'undan; değeri 16,158) **ve** `response ≥ r_min`. Yanıt eşiği yalnız alarm **kaldırabilir**; bu
   yüzden val ölçümü Faz 2 dosyalarından kurulabilir.
2. **Zaman kuralı (videoda; oturum varken):** gerçek kayma kareden kareye aynı yönde ve büyüklükte kalır,
   anlamsız kestirim her karede başka yöne zıplar. Pencere: geçerli kare dahil oturumun son **5** karesi
   (daha az kare varsa olanlar). `S` = penceredeki `misaligned_frame` kareleri; `v̄` = `S`'deki
   `(dx_px, dy_px)` vektörlerinin bileşen bazında ortancası (`numpy.median` gibi: çift sayıda ortadaki iki
   değerin ortalaması); `C = {i ∈ S : ‖vᵢ − v̄‖₂ ≤ 3 px}` (sınır dahil, float64). `misaligned` =
   `|S| ≥ 3` ve `|C| ≥ 3`. Parametreler (5, 3, 3 px) ölçümden önce sabit, ayarlanmaz.
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
- **Sabit sürümde doğrulama (2026-10-18, `trackers==2.6.1` kaynağı, plan F3 Görev 13):** yukarıdaki maddeler
  `main` dalından okunmuştu; 2.6.1'de dört fark var ve **C++ 2.6.1'i izler** (bu madde yukarıdakilerin önüne geçer):
  1. **Kalman float64**, ayrı `KalmanFilter` (`trackers/utils/kalman_filter.py`), XYXY durum tahmincisi; P₀ = I,
     Q = 0,01·I (sabit hızlı kipte `frame_step = 1`), R = 0,1·I, H = I₄ₓ₈, F sabit hız (dt = 1); kazanç
     `P Hᵀ inv(S)`; kovaryans güncellemesi **Joseph biçimi** `(I−KH) P (I−KH)ᵀ + K R Kᵀ`. C++ Eigen `double`.
  2. **IoU:** `supervision.box_iou_batch`: köşeler float64'te, oran float32'ye yazılır (`np.divide(..., out=float32)`);
     atama ve eşik bu float32 matris üzerinde.
  3. **Kimlik sayacı takipçi nesnesi başına** (0'dan; `reset()` sıfırlar); sınıf düzeyinde ortak değil.
     Yeni iz `number_of_successful_consecutive_updates = 1` ile başlar; eşleşmeyen her yüksek skorlu tespit
     `-1` döner, yalnız `≥ track_activation_threshold` olanı iz açar. Kaçırılan karede sayaç 0'a iner.
     Yaşayan iz: `time_since_update ≤ maximum_frames_without_update` (30) ve (kimliği var ya da sayaç ≥ 2 ya da bu
     karede güncellendi). Zaman damgasız kipte eşleşme öncesi budama yok (yalnız eşleşme sonrası, kare sayısıyla).
  4. **Çıktı girdi indeksiyle kurulur** (`detections[idx]`): sıra 1. aşama eşleşmeleri, 2. aşama eşleşmeleri,
     eşleşmeyen düşük skorlular (`-1`), eşleşmeyen yüksek skorlular (`-1`). Python koşucusu girdi indeksini
     `sv.Detections.data` ile taşır; kutu eşitliğiyle eşlemeye gerek yok.
  Parametre varsayılanları 2.6.1'de aynı (`frame_rate` float `30.0`). Kalman ve atama birim fikstürleri
  (`fixtures/expected/{kalman_steps,lsap_cases}.json`) bu sürümden üretilir.

**Girdi:** taban süzgecinden (`score_raw ≥ 0,1`) geçen bütün tespitler, sınıflar birleştirilmiş (M3OT tek
sınıf `vehicle`), ek NMS yok. Uçtan uca çıktı sınıflı olduğundan aynı araca iki sınıftan kutu düşebilir;
oranı raporlanır (aynı karede IoU ≥ 0,7 olan farklı sınıf çiftleri).

**Eşlik fikstürü:** `baykus_detect` ile M3OT **val** dizilerinden (val yoksa train; §4.6) üretilen
değerlendirme katmanı tespitleri, takip süzgeciyle (`score_raw ≥ 0,1`) `fixtures/tracks/m3ot_val_dets.json.gz`
olarak **repoya** girer (birkaç MB; M3OT CC BY 4.0). C++ `baykus_track` ve Python koşucusu aynı dosyayı
okur. Takip eşliği böylece CI'da koşar; test dizileri eşlikte kullanılmaz.

**Ölçüm (M3OT test, 2 dizi × 480 kare, `vehicle`, IR etiketleri;** `results/track/m3ot_test_{oracle,engine}.json`**):**
HOTA, MOTA, IDF1 `trackers eval` ile (aynı sabit paket; TrackEval gerekmez). İki satır: (a) **oracle**
— M3OT'un kendi kutuları skor 1 ile (takipçinin modelden bağımsız ölçümü), (b) **motor tespitleri**.
Dizi başına ve iki dizinin birleşiği; bootstrap yok (iki dizi). Makaledeki ByteTrack sayılarıyla yan yana
konmaz (farklı dedektör ve algoritma).

**Motor tespitlerinin M3OT doğruluğu (kullanıcı kararı 5; `results/engine/m3ot_test_detect.json`):**
AP50 (pycocotools, `maxDets ≥ 1000`, sınıfsız) ve ECE (15 aralık, ham ≥ 0,25), **değerlendirme katmanından**
(`score_raw ≥ 0,001`). Faz 1'in M3OT AP50'si (0,889) PyTorch + `agnostic_nms` ile ölçüldü; yan yana konur.
Fark birden çok nedene bağlı (PyTorch → ONNX, uçtan uca çıktının 300 kutu sınırı, sınıflar arası çift
kutu, NMS yokluğu); tek bir nedene (yalnız export) bağlanmaz. Kalibrasyon tablosu
DroneVehicle val'den öğrenildi; M3OT ECE'si onun alan dışı davranışını gösterir. Eşik yok, seçim yok.

### 4.4 Operatör ekranı (`ui/`)

Tek sayfa, spec §17 düzeni + Faz 3 ekleri. Kullanıcının günlük yığını: Next.js (App Router) + TypeScript
+ Tailwind + shadcn/ui; tam sürümler `package-lock.json` ile kilitli; fontlar repoda (OFL) ve
`next/font/local`, CDN yok; `NEXT_TELEMETRY_DISABLED=1` hem build'de hem çalışmada.

**Demo verisi düzeni** (demo paketi CLI'ı üretir; mevcut dönüştürücü çıktısı düz `<dizi>_<kare>.jpg`):

```
/data/sequences/<dizi>/rgb/<kare:06d>.jpg
/data/sequences/<dizi>/ir/<kare:06d>.png
/data/sequences/<dizi>/index.json   {"name","split","frames":[...],"fps":30,"size":[640,512],"source":"M3OT <split>, CC BY 4.0"}
```

**Next sunucu rotaları:** `GET /api/sequences` (dizi listesi), `GET /api/frame/<dizi>/<kare>/<rgb|ir>`
(görüntü baytları), `POST /api/detect` (iki biçim, aşağıda).

**`POST /api/detect` biçimleri:**
- **Arıza yok:** `Content-Type: application/json`, gövde
  `{"seq","frame","conf","track","gate","session"}`. Next sunucusu iki kanalın **orijinal baytlarını**
  diskten okuyup motora iletir (tarayıcı JPEG çözümü ve yeniden kodlama pikselleri Python referansından
  uzaklaştırmasın).
- **Arıza açık:** `Content-Type: multipart/form-data`; `meta` alanı yukarıdaki JSON + `"fault":{...}`
  (yalnız kayıt ve gösterim için); bozulan her kanal için bir dosya alanı: `rgb` ve/veya `ir`,
  `image/png`, kare boyutunda (640×512), parça başına ≤ 10 MB. Gönderilmeyen kanalı Next sunucusu
  diskteki orijinal baytlardan tamamlar. Boyut ya da tip uymazsa 400 / 413 (motorla aynı).

**Veri akışı kuralları:**
- **Arıza açıkken** tarayıcı canvas'ta bozar ve bozulmuş kanalı PNG olarak gönderir (JPEG yeniden sıkıştırması
  ek bozulma eklemesin). Tanımlar spec §12.3 ve
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
- **Alarm sözleşmesi:** her alarm ayrı bir durumdur; aynı anda birden çok alarm açık olabilir.

  | Alarm kimliği | Metin (sabit tablo) | Açık olma koşulu |
  |---|---|---|
  | `rgb_lost` | "RGB kamera kayıp" | son 5 karenin en az 3'ünde `"lost" ∈ rgb.flags` |
  | `ir_lost` | "IR kamera kayıp" | son 5 karenin en az 3'ünde `"lost" ∈ ir.flags` |
  | `ir_noisy` | "IR gürültülü" | son 5 karenin en az 3'ünde `"noisy" ∈ ir.flags` |
  | `misaligned` | "IR hizasız" | geçerli karede motorun `pair.misaligned`'ı (zaman kuralı zaten pencereli; ikinci pencere yok) |

  - Metin alarm kimliğinden gelir, motorun kapı `reason`'ından türetilmez (kapı nedeni tek, alarmlar çok).
  - Koşul her karede yeniden değerlendirilir: koşul tutunca **başlar**, tutmayınca **temizlenir**
    (histerezis yok). Başlama ve temizlenme kare numarasıyla geçmiş listesine yazılır (oturum boyunca, bellekte).
  - `session/reset` (dizi değişimi, atlama, geri adım, düğme) pencereleri, açık alarmları ve geçmişi temizler.
  - `karanlık` ve `bulanık` yalnız bilgi rozeti, alarm değil. Pencere sabit, yapılandırılmaz.
  - **Kapı rozeti** alarmlardan ayrı ve **her karenin** `gate` / `applied` alanından: "IR modele verilmiyor"
    / "RGB modele verilmiyor" (yalnız `applied=true` olan karede) ya da "yalnız uyarı". Kapı anında, kayıp
    ve gürültü alarmı gecikmeli çalışır; bu bilinçli (alarm titremeyi süzer, kapı her kareyi korur).
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
  3. **Tarayıcı akışı (normal compose, ağ açık):** Playwright (yalnız CI'da, imaja girmez) başsız Chromium
     ile `http://127.0.0.1:3000`'ı açar, bir diziyi 30 kare oynatır, arıza simülasyonunu bir kez açar;
     tarayıcının yaptığı **bütün** istekler kaydedilir. `127.0.0.1:3000` dışına tek bir istek girişimi
     (başarısız olsa bile) varsa başarısız. Ağ engelliyken çalışmak dış çağrı denenmediğini göstermez; bu
     kontrol tarayıcı tarafını, kanıt 2 sunucu tarafını kapatır (dış bağlantı denemesi kapalı katmanda zaten
     başarısız olur; `NEXT_TELEMETRY_DISABLED=1`).
  4. İkincil: `.next/` içinde `http(s)://` taraması **izin listesiyle** (React hata bağlantıları,
     `http://www.w3.org/2000/svg` gibi ad alanları); listede olmayan bir kaynak varsa başarısız. Dinamik
     kurulan adresleri yakalamaz; bu yüzden ikincil.
  Ultralytics ya da Python çalışma zamanında yok.
- **Elle kanıt (en güçlü):** laptopta Wi-Fi kapalıyken normal `docker compose up`; demo açılıyor ve
  oynatılıyor (deney günlüğüne tarih ve ekran görüntüsü).
- **CI (`.github/workflows/ci.yml`):** `python` (uv, ruff, pytest — mevcut), `engine` (Dockerfile `build`
  hedefi: CMake, ctest, eşlik testleri; buildx önbelleği `type=gha`, OpenCV katmanı her commit'te yeniden
  derlenmesin), `ui` (npm ci, eslint, `next build`, tarama), `compose` (imajlar, kanıt 1-3). Model CI'da
  GitHub Release'ten indirilir (`models.json` ile sha256 denetimli). Takip fikstürü repoda (§4.3).
- **Kod kalitesi (NFR-4):** `clang-format` (CI'da denetim), pre-commit kancası; `ctest` bir kez ASan/UBSan
  ile (CI'da ayrı yapılandırma).

### 4.6 Model, fikstürler, demo verisi

- **Model:** E3c · ONNX FP32 (38 MB) + `thresholds.json` + `calibration.json` + `gate_model.json`, bir
  GitHub Release'e (`models-v1`) eklenir; AGPL ağırlık yayımı spec §7.6 ile uyumlu. CI ve kurulum oradan
  indirir; sha256'lar `configs/engine/models.json`'da.
- **Eşlik fikstürleri:** DroneVehicle'ın lisansı yeniden dağıtıma izin vermiyor (spec §7.6) → repoya
  yalnız **M3OT val** (val yoksa train) kareleri girer; **test kareleri fikstür olmaz** (kullanıcı kararı 3).
  - **Tek kareler:** val dizilerinden eşit aralıklı 10 RGB-IR çifti, sabit liste.
  - **Sentetik durumlar:** siyah IR, gürültülü IR σ 20, 20 px kaydırılmış IR, siyah RGB; Faz 2 bozulma
    fonksiyonlarıyla Python'da üretilip **dosya olarak** kaydedilir (C++ rastgele sayı üretmez).
  - **Video klibi (zaman kuralı uçtan uca):** bir val dizisinden ardışık 15 kare (≈ 5 MB), üç değişkeni:
    temiz, baştan `shift(ir, 20, 0)`, 8. kareden sonra `shift(ir, 20, 0)`. Kaydırma tam sayı ve rastgelesiz
    (`baykus.robustness.corruptions.shift`: sağa kaydırma, boşalan kenar 0); değişkenler de dosya olarak
    kaydedilir (≈ 2 × 15 IR PNG ek).
- **Fikstür kuralı (sınır uzaklığı; üretici denetler):** her sağlık özelliği her eşikten **en az 10 ×
  tolerans** uzak; `response` için `|response − r_min| ≥ 0,2` yalnız `misalign_px ≥ t − 5` olan karelerde
  (daha küçük kaymada `misalign_px` toleransı 0,5 px olduğundan Faz 2 kuralı iki tarafta da ateşlemez ve
  `response` kararı etkilemez; birim testle gösterilir). Ham skorlar için kare elenmez (kare başına 300
  satır); Kabul 1'in satır düzeyindeki sınır kuralı geçerli. Zaman kuralı için klipte `S`'ye giren her karede
  `|‖vᵢ − v̄‖ − 3| ≥ 2 px` (iki taraftaki 0,5 px'lik bileşen farkının vektör ve ortancada en kötü toplamı
  ≈ 1,41 px). Tek karede tutmayan kare seçilmez, bir sonraki eşit aralıklı kareye geçilir; klipte başlangıç
  karesi kaydırılır. Seçim yalnız val karelerinde ve yalnız eşlik testi için (model ya da eşik seçimi değil).
  - **Karar güvenliği (2026-10-18, fikstür üretiminde):** yanıt `r_min − 0,2`'nin altındaysa kare kuralı
    kaymayı her `misalign_px`'te reddeder; o karede `misalign_px` payı aranmaz (pay 10 × tolerans kalır).
    Gerekçe: M3OT val'de yanıt hep düşük (p95 0,18), doğal kayma kareden kareye eşik çevresinde dolaşıyor;
    kelimesi kelimesine kuralla 15 karelik hiçbir klip penceresi geçmiyordu (0 / 1172), bu ayrıntıyla 9 geçiyor.
- **Beklenen çıktılar** Python'dan (`fixtures/expected/*.json`): tespitler (ham, kalibre), ham ORT çıktısı
  `[1, 300, 6]` (fikstür başına 7 KB), sağlık sözlüğü ve letterbox tensörünün **sha256**'sı (tensör repoya
  girmez; 4 × 640 × 640 float32 = 6,5 MB).
- **Atıf:** `fixtures/m3ot/NOTICE.md` (kaynak, CC BY 4.0 bağlantısı, değişiklikler: PNG → JPEG, IR tek
  kanal, sentetik bozulmalar). Repo AGPL olsa da fikstürler CC BY kalır; README'de yazılır. Zenodo'daki CC0
  kaydı doğrulanmadı; daha katı olan CC BY esas.
- **M3OT val verisi:** Drive'da yalnız test bölümü var (`m3ot_raw/m3ot_test.zip`). Tek bir **Colab CPU**
  oturumu (adım 1): kullanıcı figshare'den `M3OT.zip`'i (10 GB) tarayıcıdaki "Copy as cURL" çerezleriyle
  **Colab'a** indirir (çerezler hiçbir dosyaya yazılmaz; `docs/data/m3ot.md`), `baykus-data m3ot --splits val`
  (val'de iki modaliteli dizi yoksa `train`) → Drive `m3ot_raw/m3ot_val.zip` + dönüştürülmüş val paketi.
  Dizi sayısı ve boyut `du -sh` ile görülür, sonra karar: laptopa yalnız demo dizisi ve fikstür kareleri.
- **Aday dizi kuralı:** Spike 04'e göre ham zipte platform ve modalite başına 9 dizi var (test 1), ama bazı
  dizilerde kare numaraları **ardışık değil** (örneklenmiş). Zaman kuralı ve takip ardışık kare varsayar:
  kare numaraları ardışık olmayan dizi demo, klip ve takip fikstürü adayı olamaz (ölçümden önce kural).
- **Demo verisi:** M3OT val'in kuralı geçen dizilerinden ad sırasıyla **ilk** dizisi (ölçümden önce sabit); §4.7 demo ölçütünü
  tutmazsa ad sırasıyla sonraki val dizisi; hiçbiri tutmazsa ilk dizi + sınırlama notu. Test dizileri
  (`1-03`, `2-03`; RGB 128 MB + IR 180 MB) demo paketine ek dizi olarak girebilir ama demo ölçütü ve seçim
  onlarda yapılmaz. `demo-data/` git'e girmez; demo paketi CLI'ı §4.4 düzenini yazar. GPU yok.

### 4.7 M3OT'ta sağlık kontrolü (Faz 2 sözü; kullanıcı kararları 3 ve 5)

`docs/health.md` Sınırlamalar ve spec §23: "Faz 3'te M3OT dizilerinde niteliksel kontrol". M3OT'ta etiketler
arasında doğal kayma var (dy ortanca 5,5 px, std 10,8; uçlar −103 / +54) ve eşikler DroneVehicle'dan
seçildi; demo hikâyesi bu yüzden ölçülmeden yazılmaz.

**Protokol (laptop CPU, Python referans pipeline'ı, ONNX; iki kez aynı kodla):**
- **(V) M3OT val dizileri** → `results/health/m3ot_val_health.json`: demo ölçütü ve demo dizisi seçimi
  burada (§4.6).
- **(T) M3OT test dizileri** (`1-03`, `2-03`, 960 çift) → `results/health/m3ot_test_health.json`:
  **yalnız raporlanır**, sistem sabitken bir kez; hiçbir eşik, parametre, fikstür ya da demo seçimi
  ona bakılarak değişmez. Tutmayan sonuç README ve `docs/system.md`'de olduğu gibi yazılır.

Kareler dizi sırasıyla; oturum her dizi ve her koşul başında sıfırlanır. Koşullar (rastgelesiz olanlar
Faz 2 fonksiyonlarıyla aynen; gürültü seed 42, Faz 2'deki gibi kare adıyla):

| Koşul | Tanım |
|---|---|
| `clean` | bozulmasız |
| `misalign-20` | her karede `shift(ir, 20, 0)` (Faz 2 tanımı: yatay, +20 px, boşalan kenar 0; vektör her karede aynı) |
| `misalign-20@K` | `K = ⌊n/2⌋` (dizinin kare sayısının yarısı) karesinden itibaren `shift(ir, 20, 0)`, öncesi temiz |
| `noise-20` | Faz 2 IR gürültüsü σ 20 |
| `drop_ir` | IR sıfır |

Üç kural yan yana: Faz 2 kuralı, kare kuralı, kare + zaman kuralı. Ölçüler **dizi başına** (ayrıca birleşik
de yazılır):
1. kare başına durum oranları ve §4.4 alarm sözleşmesine göre alarm açık kare oranı (alarm kimliği başına);
2. `misalign-20@K`'da gecikme = `K` ve sonrasındaki ilk `misaligned` alarm karesinin numarası − `K`; hiç
   alarm yoksa `null` (gecikme ölçütü tutmamış sayılır); `K − 1` karesinde alarm zaten açıksa bu ayrıca
   `alarm_before_onset: true` yazılır (gecikme yine aynı tanımla);
3. `gate=auto` ile AP50 ve aynı bozulmuş girdide `gate=off` ile AP50 (IR etiketleri, sınıfsız,
   değerlendirme katmanı), farkı; dizi düzeyinde, bootstrap yok.

**Demo ölçütü (ölçümden önce sabit; kare + zaman kuralı, `gate=advise`; (V)'de değerlendirilir, (T)'de
yalnız raporlanır):** (V)'de demo adayı dizi için, (T)'de her dizi için ayrı ayrı: `clean`'de `misaligned` alarmı açık kare oranı **≤ %2**,
`misalign-20`'de **≥ %90**, `misalign-20@K`'da gecikme **≤ 10 kare** (`null` değil).

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
| `results/health/gate_model.json`, `gate_model_val.json` | `r_min` ve zaman kuralı, val keşfi |
| `results/health/m3ot_val_health.json`, `m3ot_test_health.json` | M3OT sağlık kontrolü: demo ölçütü (val), son rapor (test) |

### 6.1 Eşlik (S7, FR-E6) — CI'da, her commit'te
- **Kabul 1 — tespit:** M3OT + sentetik fikstürlerde C++ ile Python referansı (sınır kuralları plan
  incelemesinde açık karar olarak eklendi, 2026-10-18; ölçümden önce):
  (a) ham ORT çıktısı `[1, 300, 6]` satır satır, 6 bileşenin hepsinde mutlak fark ≤ 1e-4. Tek istisna:
  ardışık iki Python satırının skoru `1e-5` içindeyse (ORT'nin ilk 300 sıralaması işlemciye göre yer
  değiştirebilir) o iki satır kutu IoU'suyla eşlenir, sonra aynı ≤ 1e-4 uygulanır; yer değiştirme sayısı
  raporlanır.
  **Referans aynı makinede (kullanıcı kararı, 2026-10-18; eşik değişmedi):** ilk CI koşusunda (a) yalnız
  işlemci farkından düştü: laptopta üretilmiş fikstüre karşı GitHub makinesinde ORT'nin 300'lük
  kuyruğunda son 5 satır (skor 5e-6-4e-5, değerlendirme tabanının çok altı) değişti ve kutu köşelerinde
  1,22e-4 fark (≈ 4 float32 adımı) çıktı; (b), Kabul 2 ve 3 geçti. Bu fark ne bizim kodumuz ne de iki
  taraf arasındaki fark. Bu yüzden CI, Python referansının ham çıktısını aynı makinede yeniden üretir
  (`baykus-engine raw-ref`) ve (a)'yı ona karşı aynı ≤ 1e-4 ve aynı eşleme kuralıyla sınar; laptopta
  fikstürün kendisi referanstır (fark 0,0). Kanıt dosyası referans kaynağını yazar (`raw_reference`).
  (b) **değerlendirme katmanı** (`score_raw ≥ 0,001`), takip katmanı (≥ 0,1) ve gösterim katmanı
  (kalibre ≥ conf) için **pipeline'ın kendi sıralı çıktısı**: satırlar ORT satır indeksiyle eşlenir; iki
  tarafın kümeleri aynı, eşleşen kutularda IoU ≥ 0,99, ham skor farkı ≤ 1e-3 (H5'teki tolerans), sıra aynı.
  Sınır kuralı: herhangi bir taraftaki ham skoru katman sınırına (0,001; 0,1; kalibre conf'un geçtiği ham
  blok sınırı) `1e-4`'ten yakın satır kümede olup olmamakta serbesttir; sırada yalnız kalibre skorları eşit
  ve ham skorları `1e-4` içinde olan komşular yer değiştirebilir. Serbest satırlar ve yer değiştirmeler
  sayılır; her fikstürde her katmanda **karşılaştırılan satır oranı ≥ %99**, değilse ❌.
  (c) kalibrasyon ayrı birim testle: tablo, karşılaştırma ve çıktı **double** (ham skor float32 → double);
  C++ kalibre skoru = tablo(C++ ham skoru), blok sınırları dahil **birebir**; sınamada her `x`'in float32
  karşılığı ve iki float32 komşusu. Kalibre skor pipeline düzeyinde ham skordan karşılaştırılmaz
  (basamaklı fonksiyon: ham 0,2533'te 0,231 → 0,282 sıçrar).
- **Kabul 2 — letterbox:** fikstürlerde C++ tensörünün baytları Python tensörünün sha256'sıyla aynı.
- **Kabul 3 — sağlık:** her fikstürde durumlar, kapı kararı ve `reason` **aynı**; `std` / `noise` / `blur` /
  `brightness` göreli fark ≤ 1e-4; `misalign_px`, `dx_px`, `dy_px` mutlak fark ≤ 0,5 px, `response`
  mutlak fark ≤ 0,02. **Zaman kuralı** iki testle:
  (a) **uçtan uca:** video klibinin (§4.6) üç değişkeni iki tam pipeline'dan (kodlanmış baytlardan,
  oturumlu, `gate=auto`) geçer; her karede `misaligned_frame`, `misaligned`, `gate`, `applied` ve alarm
  durumları **aynı**;
  (b) **birim:** aynı `pair` dizisinde (Python'da üretilmiş sentetik 50 karelik diziler: sabit kayma,
  zıplayan kestirim, ortada başlayan kayma, tam 3 px sınırında vektör, çift sayıda `S`, sıfırlama) C++ ve
  Python her karede aynı `misaligned`.
  OpenCV sürümü iki tarafta aynı (5.0.0); tolerans laptop (Zen 4, AVX-512) ile CI işlemcisi arasındaki
  SIMD farkı için. Fikstür kuralı (§4.6) eşiğe yakın kareyi dışarıda tuttuğundan durum eşitliği toleransla çelişmez.
- **Kabul 4 — takip (FR-E4):** `fixtures/tracks/m3ot_val_dets.json.gz` üzerinde C++ ve Python takipçisi.
  **Tanım:** tespitler girdi indeksiyle eşlenir; `-1` (`null`) yalnız `-1` ile eşleşir; dizi başına tek bir
  C++ → Python kimlik eşlemesi vardır ve birebirdir. Ölçüt: bu eşlemeyle **farklı kare sayısı 0**.
  **İstisna (kullanıcı kararı 6, ölçümden önce):**
  - Her dizide karşılaştırma **ilk farklı karede durur** (sonraki kareler takipçi durumu ayrıştığı için
    karşılaştırılmaz; "zincirleme fark" yorumu yok).
  - O kare istisna sayılır ancak koşucular o karenin girdisini döker (iz kutuları, tespitler, IoU matrisi,
    iki aşamanın atamaları) ve şu kanıtlardan **biri** mekanik olarak gösterilirse: (i) atamaya giren bir
    IoU için `|IoU − 0,1| < 1e-5` (float32); (ii) bir tespit skoru için `|score − 0,6| < 1e-6` ya da
    `|score − 0,7| < 1e-6`; (iii) iki tarafın atamalarının toplam IoU'su `1e-6` içinde eşit (eş optimum)
    ve ikisi de eşik sonrası geçerli.
  - Kabul 4 ✅ "istisnayla" ancak **her dizide karşılaştırılan kare oranı ≥ %90** ise (ilk fark dizinin
    son %10'unda). Kanıtsız fark ya da daha erken fark → Kabul 4 ❌. Hepsi `parity_track.json`'da.

### 6.2 Kare kuralıyla kapı (val, keşif; §4.2)
Faz 2 tablosunun aynısı (16 koşul, E3c vs E3c + kapı, eşleştirilmiş bootstrap 1000, seed 42) yeni kuralla,
kare birleştirmeyle. Faz 2'nin Kabul 1-3 tanımlarıyla değerlendirilir; sonuç **keşif** olarak raporlanır,
Faz 3'ün kabul ölçütü değildir ve ürün varsayılanını değiştirmez. Gereken dosyalar Drive'da: 16 val
koşulunun `robust/<run>_val_<koşul>_gate/` (sağlık kaydı + tahminler) ve taban tahminleri
`robust/<run>_val_<koşul>/` (tahmin ≈ 70 MB; kullanıcı adımı olarak adım 4'te indirilir).

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
**Kabul 6:** §4.5'in otomatik kanıtları (1-4) CI'da yeşil + elle Wi-Fi kapalı demo kaydı.

### 6.6 Ekran ve sağlamlık
- **Kabul 7 — ekran:** şu listenin her maddesi çalışıyor ve ekran görüntüsü ya da GIF ile deney günlüğünde
  (M3OT demo dizisinde, arıza simülasyonu açık / kapalı): FR-U1 oynatıcı (oynat / durdur / kare kare / hız);
  FR-U2 RGB / IR / yan yana, kutu, sınıf, kalibre skor, iz numarası; FR-U3 conf kaydırıcısı, takip ve kapı
  modu, canlı gecikme grafiği (**mod seçimi yok**, §3); FR-U4 arıza simülasyonu (karartma, IR gürültü,
  IR kaydırma, bulanıklık, kamerayı kapat); sağlık paneli; §4.4 alarm sözleşmesi (başlama, temizlenme,
  sıfırlama); kapı rozeti.
- **Kabul 8 — API sağlamlığı (CI):** eksik parça → 400, farklı boyut → 400, bozuk PNG → 400, sınır üstü →
  413; ardından 1000 rastgele baytlık istek ve `/health` hâlâ `ok`.
- **Demo ölçütü** (§4.7): demo dizisini seçer; tutmazsa §4.6'daki yol. Fazın kabulünü engellemez.

### 6.7 Faz çıkışı

İşe devam etmek ile fazı kapatmak ayrıdır: bir ölçüt tutmasa da sonraki adımlara geçilebilir, ama faz
aşağıdaki tabloya göre kapanır (spec "tamamlanma tanımı": kabul ölçütleri karşılanır).

| Ölçüt | Etkisi |
|---|---|
| Kabul 1, 2, 3 (eşlik) | **çıkışı engeller** (S7) |
| Kabul 4 (takip eşliği) | **çıkışı engeller**; §6.1'deki tanımlı istisnayla geçer |
| Kabul 5 (gecikme) | **çıkışı engeller**; aşılırsa faz yalnız kullanıcının yazılı kararıyla (ADR) kapanır, README'de olumsuz sonuç |
| Kabul 6 (kapalı ağ) | **çıkışı engeller** (S8) |
| Kabul 7, 8 (ekran, API) | **çıkışı engeller** |
| Val keşfi (§6.2), M3OT takip ve AP50 / ECE (§6.3), M3OT sağlık (§4.7), demo ölçütü | **yalnız raporlanır** (eşik yok ya da keşif) |

Çıkış: S7, S8 işaretli; `docs/system.md` (mimari, API, eşlik ve gecikme tabloları, takip ve M3OT tabloları, keşif
kapı tablosu, M3OT sağlık kontrolü); README'ye sistem bölümü, ekran görüntüsü ve internetsiz kurulum
(üç komut); `v0.2.0`.

## 7. İş yükü ve takvim

Bütün kodu Claude yazar (kullanıcı kararı 4); kullanıcı adımları: Colab CPU oturumu, Drive'dan indirmeler,
Docker Desktop ayarı, Wi-Fi kapalı elle test, onaylar. Önerilen sıra (her adım bir öncekini kullanır;
plan haftalara böler):

| # | İş | Tahmin |
|---|---|---|
| 1 | **Colab CPU** (M3OT val dönüşümü ve paketi, §4.6; test demo paketi) · motor iskeleti: Dockerfile (`deps` / `build` / `runtime`, OpenCV 5.0.0 kaynaktan), CMake, GoogleTest, CI `engine` işi ve önbellek | 3-4 gün |
| 2 | **Python referansı ve sağlık sözleşmesi:** `baykus.engine_ref` (birleştirme, pipeline, çıktı katmanları, sıralama) · kayma vektörü, `r_min` seçimi (train), zaman kuralı (`baykus.health.temporal`), `gate_model.json` · fikstür üretici (sınır kuralıyla, `r_min` artık belli) · sürüm sabitleme · **bütün model dosyaları hazır olunca** `models.json` + Release `models-v1` | 3-4 gün |
| 3 | `preprocess` + `detector` + `calibration` C++ + Kabul 1-2 | 3-4 gün |
| 4 | `health` + zaman kuralı C++ + Kabul 3 (a, b) · val keşif ölçümü (laptop; Drive'dan ≈ 70 MB) · §4.7 (V) ölçümü ve demo dizisi | 3-4 gün |
| 5 | `baykus_detect` + takip fikstürü (val) · Python takip koşucusu · Kalman, `rectangular_lsap` uyarlaması, ByteTrack + Kabul 4 (**zaman kutusu 2 gün**; aşılırsa farklar ve kanıtları listelenir, sonraki adımlara geçilir, Kabul 4 açık kalır ve §6.7'ye göre çıkışı engeller) | 5-6 gün |
| 6 | HTTP API, oturumlar, hatalar + Kabul 8 · compose iskeleti ve `compose.offline.yml` (kanıt 1-2 erken) · `baykus_bench` + Kabul 5 | 3-4 gün |
| 7 | UI: rotalar, oynatıcı, kutular, sağlık paneli, alarm, arıza simülasyonu, gecikme grafiği + Kabul 7 | 5-7 gün |
| 8 | CI `ui` + `compose` işleri, tarama, `scripts/prepare --bundle`, Wi-Fi kapalı elle test (Kabul 6) | 2-3 gün |
| 9 | **Son ölçümler (sistem sabit, test bir kez):** M3OT test AP50 / ECE, takip (HOTA / MOTA / IDF1), §4.7 (T) · ADR 0008 · `baykus-system report` → `docs/system.md`, README, spec v4.5, `v0.2.0` | 2-3 gün |

**Toplam ≈ 6-7 hafta; planda 5 günlük tamponla ≈ 7-8 hafta** (spec'te 8 hafta, 7 Ara – 31 Oca; takvimin ≈ 7 hafta önündeyiz; Ocak final dönemi
için tampon kalıyor, ADR 0006). En riskli adımlar: 5 (takip eşliği: eşit maliyetler, float32 Kalman) ve
1 (OpenCV derlemesi, CI süresi). GPU yok; Colab yalnız CPU (adım 1; kullanıcı adımı: figshare çerezleriyle `M3OT.zip` indirme).
Laptop'a inecekler: M3OT demo paketi (val dizisi + test dizileri; boyut önce görülür, test ≈ 310 MB), fikstür kareleri (≈ 15 MB), Faz 2'nin val tahmin ve sağlık
dosyaları (≈ 70 MB).

## 8. Riskler

| Risk | Önlem |
|---|---|
| Windows'ta C++ araç zinciri | Hiç kurulmaz: derleme ve test yalnız Docker'da; CI aynı Dockerfile. |
| OpenCV'yi kaynaktan derlemek CI'ı yavaşlatır | Ayrı `deps` katmanı, buildx `type=gha` önbelleği, yalnız üç modül. |
| OpenCV aynı sürümde bile işlemciye göre farklı sonuç (SIMD) | Kabul 3 toleransları + fikstür sınır kuralı; derleme bilgisi kayıtta. |
| `trackers` ByteTrack ayrıntıları ve eşit maliyetli atamalar → sıfır fark tutmaz | SciPy çözücüsünün uyarlaması, float32 Kalman; zaman kutusu ve önceden yazılmış istisna kuralı (§6.1). |
| M3OT val'de iki modaliteli, ardışık kareli dizi yok ya da figshare indirmesi engellenir | Önce train bölümü (aynı kural); o da yoksa fikstür ve demo test dizilerinden, bu README ve `docs/system.md`'de "test verisi geliştirmede kullanıldı" diye açıkça yazılır (kullanıcıya sorulur). |
| M3OT'ta doğal kayma ya da gece kareleri → temiz karede alarm, demo bozulur | §4.7 (V) ölçümü demo dizisini seçer; hiçbir val dizisi tutmazsa sınırlama notu. (T) sonucu ne çıkarsa yazılır. |
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
- Fikstür karelerinin sabit listesi (varsayılan: val dizilerinden eşit aralıklı 10 kare ve 15 karelik klip, sınır
  kuralıyla kaydırılmış; liste gerekçesiyle `fixtures/m3ot/README.md`'de).
- Oturum sınırları ve parça boyutu (varsayılan: 8 oturum, 10 dk, 10 MB).
- Next.js ve paket sürümleri (`package-lock.json`).

## 10. Spec'e yansıyacaklar (onaydan sonra, v4.5)

§2.3 S7/S8 metni (eşlik tanımları §6.1), §7.4 FR-E2/E3 (late_fusion ve NMS yolu Faz 3'te yok), FR-E4
(referans `trackers==2.6.1`), FR-E5 API (§4.1), FR-U eklemeleri (sağlık paneli, alarm, kapı modu) ve FR-U3
"mod seçimi"nin düşmesi, §15.1 modül listesi, §15.3 API, §15.4 madde 4 (CI'da küçük `n` modeli değil E3c;
E8-n eğitilmedi), §16 (Python referansı `model.track` değil `trackers`), §17 düzen (model/mod seçici yok),
§18 compose ağları, CI işleri (OpenCV apt değil kaynaktan; `compose` işi) ve NFR-1 kanıtları, §22 Faz 3
satırı (≈ 6-7 hafta) ve H6-H9 tablolarının bu tasarıma göre güncellenmesi (H6 #36 letterbox ve #37'deki
(S) işaretleri kalkar, kullanıcı kararı 4), §23 risk satırları; §8.4 "eşikler sadece val'de seçilir" kuralına sağlık eşiklerinin istisnası (Faz 2 eşikleri ve `r_min` train'den seçilir, val keşif ve doğrulama için; `docs/design/faz2.md`, bu belge §4.2); FR-V9 ile uyum (değerlendirme katmanı `score_raw ≥ 0,001`, §4.1); M3OT val bölümünün kullanımı (fikstür, demo; §4.6).
