# Faz 3 Tasarımı — Sistem (yalın)

> **Tarih:** 2026-10-17 · **Durum:** taslak, kullanıcı onayı bekliyor · **Dal:** `f3` (açılacak)
> **Dayanak:** spec §2.3 S7-S8, §7.4 FR-E/FR-U, §7.5 NFR-1/5/6, §15-§18, §22 H6-H9, ADR 0006, ADR 0007;
> Faz 2 sonuçları (`docs/health.md`, `docs/experiments.md` "Karar: kapı düzeltmesi Faz 3'e").
> **Kullanıcı yönergesi (2026-10-17):** sorular atlandı, "tasarıma geç"; yalın, boş iş yok (Faz 2 ile aynı ilke).
> Kodlama başka bir oturumda yapılacak: bu belge ve ondan yazılacak plan (`docs/plans/F3.md`) yeterli olmalı.

## 1. Neden ve ne

Faz 1-2 bir **model** ve onun ölçümlerini üretti: sevk edilen E3c · ONNX FP32 (laptop CPU p95 60 ms, test
mAP50 0,814), kare başına sensör sağlık dedektörü, kalibrasyon tablosu. Hepsi Python'da ve notebook'larda.
Faz 3 bunları **bir operatörün kullanacağı sisteme** çevirir:

**Bir C++ motoru görüntü çiftlerini alır, araçları bulur, takip eder, iki kameranın sağlığını ölçer;
bir web ekranı videoyu oynatıp kutuları, takip numaralarını ve kamera uyarılarını gösterir; hepsi
internete çıkmadan tek komutla (`docker compose up`) açılır.**

Mülakat hikâyesi (demo): "M3OT videosu oynuyor; araçlar kutulu ve numaralı. IR kamerayı ekrandan 20 piksel
kaydırıyorum; birkaç kare sonra 'IR hizasız' uyarısı çıkıyor, isterseniz sistem IR'yi kendisi kapatıyor.
Motor C++; çıktısı Python referansıyla CI'da birebir karşılaştırılıyor; internet kablosu çekili çalışıyor."

## 2. Hedefler

1. **C++ motoru (S7):** letterbox → ONNX Runtime → YOLO26 uçtan uca çıktı çözümleme → geri ölçekleme;
   Python referansıyla eşlik testi CI'da (FR-E1, E5, E6, E7).
2. **Sağlık ve güven motorda:** Faz 2 dedektörünün C++ karşılığı (Python ile eşlikli), Faz 2'nin
   önerdiği **yanıt korumalı** hizasızlık kuralı, kalibre skor. Kapı varsayılan olarak **yalnız uyarır**.
3. **Takip:** kendi C++ ByteTrack'imiz (Kalman + Macar + iki aşamalı eşleşme), oturum başına; Python
   referansıyla eşlik; M3OT test'te HOTA / MOTA / IDF1 (FR-E4, spec §16).
4. **Operatör ekranı:** tek sayfa Next.js; oynatıcı, kutular ve iz numaraları, sensör sağlık paneli ve
   alarmlar, arıza simülasyonu, canlı gecikme (FR-U1-U4 + Faz 3 ekleri).
5. **Kapalı ağ (S8):** `docker compose up` ile, çalışırken dış ağa çıkmadan; GitHub Actions CI (NFR-1, §18).

## 3. Hedef dışı (bilerek kesilenler)

| Kesilen | Neden |
|---|---|
| **Geç füzyon motorda + C++ WBF** (FR-E2 late_fusion, H7 #39) | E4 laptop CPU'da sevk edilmedi (p95 131,6 ms, ADR 0007). Sevk edilen sistem tek model. |
| **Ham çıktı + NMS yolu** (FR-E3, YOLO11 yedeği) | Uçtan uca FP32 sevk edildi; INT8 sevk edilmedi, YOLO11'e geçiş yok. |
| **TensorRT / CUDA / NPU yolu motorda** | Hedef donanım laptop CPU (ADR 0007). RTX 4060 isteğe bağlı satır olarak bile bu fazda yok. |
| **Dikdörtgen (640×512) export** (Faz 1 açık işi) | Sevk edilen modeli değiştirir; val doğruluğu yeniden ölçülmeli (Colab). Faz 5 tamponuna. |
| **Python vs C++ hız tablosu** (H7 #41'in karşılaştırma kısmı) | Docker (Linux VM) ile Windows yerel ölçüm karşılaştırılamaz. Yalnız motorun kendi gecikmesi ölçülür (§6.4). |
| Kullanıcı girişi, veritabanı, çoklu kullanıcı, kayıt/oynatma geçmişi | Spec §17 kapsam dışı. |
| Kubernetes, imza zinciri, imaj tarama | Kapalı ağı göstermek için tek compose yeter. |
| Yeni eğitim, yeni model | "Train only what ships"; model değişmez. |
| Donmuş kare tespiti (Faz 2 devri) | Ardışık kare karşılaştırması basit görünse de eşiği ölçülmemiş yeni bir alarm olur; Faz 4'ün video işleme altyapısıyla birlikte. Bu fazda yalnız `docs/health.md` sınırlaması. |

**Anti-hedefler (çalışsa bile başarısız sayılır):**
- Motor çıktısı Python referansından tolerans dışında ayrışıyor ve CI bunu yakalamıyor.
- Demo açılırken ya da çalışırken internetten bir şey indiriyor (model, font, CDN, telemetri).
- Temiz veride zarar verdiği ölçülmüş bir kapı **varsayılan olarak** modele veri kestiriyor.
- Sistemin uçtan uca p95'i ADR 0007 hedefini (100 ms) aşıyor ve bu raporlanmıyor.

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
  `back` (motora erişim) ve `front` (yalnız `127.0.0.1:3000` yayını).
- **Tarayıcı motoru hiç görmez:** bütün istekler Next.js sunucu rotası (`/api/detect`) üzerinden.
- **Model ve demo verisi imaja gömülmez,** salt okunur volume olarak bağlanır (imajlar küçük kalır,
  model değişince yeniden derleme gerekmez). Model dosyaları `docker compose up`'tan önce yerelde olmalı
  (GitHub Release'ten bir kez indirilir; §4.6).

### 4.1 C++ motoru (`engine/`)

**Derleme ortamı: yalnız Docker (Linux).** Windows'ta C++ araç zinciri kurulmaz. `docker/engine.Dockerfile`
çok aşamalı: `build` (Ubuntu 24.04, GCC, CMake, OpenCV, Eigen, ONNX Runtime C++ hazır paketi;
`ctest` burada koşar) → `runtime` (yalnız kütüphaneler + `baykus_server`). Geliştirme döngüsü
`docker build --target build` ile; VS Code kullanılıyorsa aynı imajdan bir Dev Container (`.devcontainer/`).
CI aynı Dockerfile'ı kullanır: tek doğru kaynak.

**ONNX Runtime sürümü** Python referansını üreten sürümle **aynı** (`python/uv.lock`'taki onnxruntime;
şu an 1.30.0). Eşlik testi böylece kütüphane farkını değil bizim kodumuzu ölçer.

Modüller (spec §15.1'in yalın hâli; WBF ve NMS yok):

| Modül | Sorumluluk | Python karşılığı (referans) |
|---|---|---|
| `image_io` | RGB (BGR uint8) ve IR (gri uint8) okuma, 4 kanal R, G, B, T birleştirme | `baykus.edge.frames.load_input` |
| `preprocess` | letterbox (gain, yuvarlama, dolgu 114), kanal sırası, `/255`, CHW | `baykus.edge.frames.preprocess` (piksel düzeyinde aynı, Faz 1'de Ultralytics'le doğrulanmış) |
| `detector` | ORT oturumu, giriş tipi/şekli modelden, uçtan uca `[1, 300, 6]` çözümleme, geri ölçekleme, kırpma, `conf` süzgeci | `baykus.edge.frames.postprocess`, `OrtBackend` |
| `health` | Faz 2 özellikleri (std, Immerkær gürültü, Laplace varyansı, HSV V parlaklık, yarı çözünürlük Sobel + Hanning + faz korelasyonu, float64) ve durum / kapı kuralı | `baykus.health.features`, `baykus.health.gate` |
| `calibration` | izotonik tablo (`x`, `y`) ile skor eşleme; sıralama (kalibre, ham) | `baykus.health.calibrate.apply_isotonic` |
| `kalman`, `assignment`, `tracker` | ByteTrack (§4.3) | Python referans takipçi |
| `pipeline` | sıralama ve aşama süreleri (`decode`, `health`, `pre`, `infer`, `post`, `track`, `total`) | — |
| `server` | HTTP API (cpp-httplib), hata yönetimi | — |
| `apps/` | `baykus_server`, `baykus_bench` (§6.4), `baykus_track` (tespit dosyasından takip; eşlik ve M3OT ölçümü için) | — |

**HTTP API** (spec §15.3'ün güncel hâli; değişen yerler kalın):

```
GET  /health      → {"status":"ok","model_loaded":true}
GET  /v1/model    → {"name","mode","precision","input_size":[640,640],"channels":4,"classes":[...],
                     "thresholds_sha256","calibration_sha256"}
POST /v1/detect   multipart: rgb=<jpg/png>, ir=<png>   (model kanalına göre biri ya da ikisi)
     sorgu: ?conf=0.25&track=true&session=<id>&gate=off|advise|auto   (varsayılan advise)
     → {"detections":[{"cls","cls_id","score","score_raw","box":{x1,y1,x2,y2},"track_id"}],
        "health":{"rgb":{...},"ir":{...},"pair":{...},"gate":"none|ir_off|rgb_off",
                  "reason":"...","applied":false},
        "timing_ms":{"decode","health","pre","infer","post","track","total"}}
POST /v1/session/reset?session=<id>
```

- `score` kalibre skor, `score_raw` modelin skoru; `conf` süzgeci **kalibre** skora uygulanır (operatör
  eşiği 0,25 olasılık anlamı taşısın). Sıralama (kalibre, ham) — Faz 2 Kabul 4 bulgusu.
- `health` Faz 2'nin sağlık sözlüğüyle aynı alanlar (`docs/plans/F2.md` Görev 3); `applied` kapının
  pikselleri gerçekten sıfırlayıp sıfırlamadığı (`gate=auto` ve karar ≠ `none`).
- Hata yanıtları spec §15.3 gibi (400 / 413 / 500); motor tek hatalı istekle çökmez.
- Motor açılışta `/models` altından okur: `e3c.onnx`, `thresholds.json` (Faz 2), `calibration.json`
  (Faz 2). Dosya yoksa ya da sha256 beklenenden farklıysa açılmaz, nedenini loglar.

### 4.2 Sağlık kapısı: yanıt koruması ve varsayılan

Faz 2 bulgusu: temiz karelerdeki yanlış alarmların hepsi "IR hizasız", %80'i karanlık gece; bu karelerde
faz korelasyonu yanıtının medyanı 0,05 (normalde 0,56). **Kural (ölçümden önce sabit):** `misaligned`
yalnız `misalign_px ≥ t` **ve** `response ≥ r_min` iken. `t` Faz 2'deki eşik (16,158 px), değişmez.
`r_min` **train**'deki Faz 2 özellik satırlarından (`train_features.csv`) seçilir: Faz 2'nin `misaligned`
pozitif / negatif kümeleriyle, `t` sabitken birleşik kuralın Youden J'sini en büyük yapan `r`
(tam sayı J, Faz 2'nin eşitlik kuralı).

**Ölçüm (val, keşif):** yeni Colab oturumu yok. Kural yalnız kapıyı **kaldırabilir** (eklemez); bu yüzden
her val koşulunun kapılı sonucu Faz 2 dosyalarından birebir kurulur: yeni kuralda kapı kalkan karelerin
tahmini taban koşudan, kalanlar kapılı koşudan (kare başına tahminlerin bağımsız ve iki koşuda aynı olduğu
Faz 2'de doğrulandı: 1409 / 1409). Eşleştirilmiş bootstrap ve Faz 2'nin Kabul 1-3'ü aynı tanımlarla. Kural val
sonuçlarına bakılarak bulunduğu için bu ölçüm **"val'de keşif"** diye raporlanır; test kullanılmaz.

**Ürün varsayılanı:** `gate=advise` (sağlık ölçülür, uyarı gösterilir, piksel kesilmez). `auto` kullanıcı
seçimi. Keşif ölçümünde yanıt korumalı kapı Faz 2'nin Kabul 1 ve 3'ünü geçse bile varsayılan değişmez (testte
doğrulanmadı); README'de "val'de keşif: temiz zarar X, fayda Y" diye yazılır.

### 4.3 Takip

- **Algoritma:** ByteTrack (yüksek / düşük skorlu tespitlerle iki aşamalı eşleşme, sabit hızlı Kalman,
  IoU maliyeti, Macar ataması, iz yaşam döngüsü). Parametreler Python referansının varsayılanları;
  `configs/engine/tracker.yaml`'da sabit.
- **Python referansı:** `roboflow/trackers` ByteTrack (Apache-2.0; spec §5). Uygulama sırasında sürümü
  ve ByteTrack ayrıntıları (Kalman durum vektörü, eşleşme sırası, iz silme süresi) okunur; C++ bunu
  izler. Ayrıntı farkı zorunlu kalırsa belgelenir (FR-E4).
- **Eşlik:** aynı tespit dizisi verildiğinde iz atamaları **ID permütasyonu dışında aynı**. Girdi: M3OT
  test dizilerinde motorun kendi tespitleri (JSON dosyası); `baykus_track` C++ ve Python referansı aynı
  dosyayı okur.
- **Ölçüm (M3OT test, 2 dizi × 480 kare, `vehicle`, IR etiketleri):** HOTA, MOTA, IDF1; iki satır:
  (a) **oracle** — M3OT'un kendi kutuları (modelden bağımsız takipçi ölçümü), (b) **motor tespitleri**
  (E3c · ONNX FP32, kalibre skor, sınıfsız). Metrikler hazır bir kütüphaneden (TrackEval ya da
  `trackers` değerlendirmesi), kendimiz yazmayız. Kareler ardışık video: tek dizi düzeyinde sayılar
  raporlanır, bootstrap güven aralığı verilmez (iki dizi var).

### 4.4 Operatör ekranı (`ui/`)

Tek sayfa, spec §17 düzeni + Faz 3 ekleri. Kullanıcının günlük yığını: Next.js (App Router) + TypeScript
+ Tailwind + shadcn/ui; fontlar yerelde (`next/font/local`), CDN yok.

- **Oynatıcı:** Next.js sunucusu `/data/sequences/<dizi>/{rgb,ir}/` altındaki kareleri sunar; oynat /
  durdur / kare kare / hız (FR-U1). RGB / IR / yan yana; kutu, sınıf, kalibre skor, iz numarası (FR-U2).
- **Üst çubuk:** conf kaydırıcısı, takip aç/kapa, kapı modu (`kapalı / uyar / otomatik`), oturum sıfırla.
- **Sağ panel:** tespit listesi; canlı gecikme grafiği (son 100 kare, p50 / p95, aşama kırılımı) (FR-U3).
- **Sensör sağlık paneli:** her kamera için durum rozeti (`tamam / kayıp / gürültülü / karanlık / bulanık`)
  ve çift için `hizasız`; son 50 karenin durum şeridi; ölçülen değerler (gürültü σ, kayma px, yanıt).
- **Alarm:** bir arıza durumu **son 5 karenin en az 3'ünde** varsa alarm (tek karelik titremeyi süzer);
  alarm metni Faz 2'nin `reason`'ı ("IR hizasız" gibi) ve kapının ne yaptığı ("IR modele verilmiyor" /
  "yalnız uyarı"). Alarm temizlenince geçmiş listesinde kalır (oturum boyunca, bellekte).
- **Arıza simülasyonu (FR-U4):** tarayıcıda, kareyi motora göndermeden önce canvas üzerinde (bozulmuş kare PNG olarak gönderilir; JPEG yeniden sıkıştırması ek bozulma eklemesin): RGB karart
  (çarpan), IR gürültü (σ), IR kaydır (px), bulanıklaştır, kamerayı kapat. Sağlamlık paketindekiyle aynı
  tanımlar (spec §12.3); rastgele sayılar numpy ile aynı değil (demo amaçlı, ölçüm değil).
- **Kapsam dışı:** giriş, veritabanı, video yükleme (diziler klasörden), çoklu oturum yönetimi.

### 4.5 Kapalı ağ ve CI

- `docker-compose.yml`: `engine` (yalnız `back`, `internal: true`), `ui` (`back` + `front`, port
  `127.0.0.1:3000:3000`). `pull_policy: never`; imajlar yerelde derlenmiş olmalı.
- **NFR-1 kanıtı (otomatik):** (1) `docker compose up` sonrası `scripts/offline_check` UI üzerinden bir
  kareyi uçtan uca işletir; (2) motor konteynerinden dış adrese bağlantı denemesi **başarısız** olmalı;
  (3) UI'nin derlenmiş çıktısında (`.next/`) `localhost` dışı `http(s)://` kaynağı yok (statik tarama);
  Ultralytics ya da Python çalışma zamanında yok. **Elle kanıt:** laptopta Wi-Fi kapalıyken demo
  açılıyor ve oynatılıyor (deney günlüğüne tarih ve ekran görüntüsü).
- **CI (`.github/workflows/ci.yml`):** `python` (uv, ruff, pytest — mevcut), `engine` (Dockerfile
  `build` hedefi: CMake, ctest, eşlik testi dahil), `ui` (npm ci, eslint, `next build`, tarama). Eşlik
  testi için model CI'da GitHub Release'ten indirilir (sha256 denetimli).

### 4.6 Model ve fikstürler

- **Model:** E3c · ONNX FP32 (38 MB) + `thresholds.json` + `calibration.json`, bir GitHub Release'e
  (ör. `models-v1`) eklenir; AGPL ağırlık yayımı spec §7.6 ile uyumlu. CI ve kurulum oradan indirir.
- **Eşlik fikstürleri:** DroneVehicle'ın lisansı yeniden dağıtıma izin vermiyor (spec §7.6) → repoya
  yalnız **M3OT** (CC-BY-4.0, atıflı) kareleri girer: test dizilerinden ≈ 10 RGB-IR çifti, sabit seçim.
  Ek olarak sentetik durumlar (siyah IR, gürültülü IR, kaydırılmış IR; Faz 2 bozulma fonksiyonlarıyla
  üretilmiş). Beklenen çıktılar Python'dan (`fixtures/expected/*.json`: tespitler + sağlık).
- **Demo verisi:** M3OT test dizilerinden biri ya da ikisi (`demo-data/`, git'e girmez). Hazırlık için
  tek bir **Colab CPU** oturumu: Drive'daki `m3ot_test.zip` → mevcut `baykus-data m3ot` → laptop paketi
  (boyut `du -sh` ile görülmeden indirilmez; tahmin birkaç yüz MB). GPU yok.

## 5. Faz 2'den devreden girdiler

| Girdi | Bu tasarımda |
|---|---|
| Karanlıkta hizasızlık yanlış alarmı | §4.2 yanıt koruması; val'de keşif ölçümü; ürün varsayılanı `advise` |
| Kalibrasyonda eşitlik | §4.1: sıralama (kalibre, ham); motor `score` + `score_raw` |
| Dedektör hızı (laptop 11,3 ms; hedef < 5 ms) | §6.4: motorda `health` aşaması ayrı ölçülür; p95 bütçesine dahil |
| Donmuş kare | Hedef dışı (§3); `docs/health.md` sınırlaması |
| Eşik seçiminin belleği (`searchsorted`) | Bu fazda gerekmiyor: `r_min` seçimi tek parametreli bir tarama, `t` sabit; Faz 2 eşikleri yeniden seçilmez |

## 6. Değerlendirme ve kabul ölçütleri (ölçümden önce sabit)

### 6.1 Eşlik (S7, FR-E6) — CI'da, her commit'te
- **Kabul 1 — tespit:** M3OT + sentetik fikstürlerde C++ ile Python referansı: kutu sayısı aynı, eşleşen kutularda
  IoU ≥ 0,99, ham skor farkı ≤ 1e-3, kalibre skor farkı ≤ 1e-3 (H5'teki tolerans).
- **Kabul 2 — letterbox:** fikstürlerde Python girdisiyle en büyük piksel farkı 0 (float32 değerleri birebir).
- **Kabul 3 — sağlık:** her fikstürde durumlar, kapı kararı ve `reason` **aynı**; `std` / `noise` / `blur` /
  `brightness` göreli fark ≤ 1e-4; `misalign_px` mutlak fark ≤ 0,5 px, `response` mutlak fark ≤ 0,02.
  (OpenCV sürümü Python'daki `opencv-python` ile aynı olmayabilir; tolerans bunun için.)
- **Kabul 4 — takip (FR-E4):** M3OT test dizilerinde motor tespitlerinden C++ ve Python takipçisi: her karede
  aynı kutu kümesi iz numarası permütasyonu dışında aynı izlere atanmış (fark sıfır kare).

### 6.2 Yanıt korumalı kapı (val, keşif; §4.2)
Faz 2 tablosunun aynısı (16 koşul, E3c vs E3c + kapı, eşleştirilmiş bootstrap 1000, seed 42) yeni kuralla.
Faz 2'nin Kabul 1-3 tanımlarıyla değerlendirilir; sonuç **keşif** olarak raporlanır, Faz 3'ün kabul ölçütü değildir ve ürün varsayılanını değiştirmez. Gereken dosyalar Drive'da: 16 val koşulunun `robust/<run>_val_<koşul>_gate/health.jsonl.gz` ve tahminleri, taban tahminleri `robust/<run>_val_<koşul>/` (tahmin ≈ 70 MB).

### 6.3 Takip (M3OT test)
HOTA, MOTA, IDF1; oracle ve motor tespitleri; dizi başına ve iki dizinin birleşiği. Eşik yok (ilk ölçüm);
sonuç ADR 0008 (takip değerlendirmesi) ile kayda geçer.

### 6.4 Gecikme ve bellek (NFR-5, NFR-6)
`baykus_bench`, H5 protokolü (§14: 50 ısınma + 500 ölçüm, 20 bench görüntüsü — laptopta zaten var),
motor konteynerinde, laptop prize takılı: aşama başına p50 / p95 ve tepe RSS.
**Kabul 5:** uçtan uca p95 (`decode` dahil, `health` dahil, `track` dahil, HTTP hariç) ≤ **100 ms**
(ADR 0007). HTTP dahil p95 ayrıca raporlanır (eşik yok). Docker (WSL2) ölçümü olduğu yazılır.

### 6.5 Kapalı ağ (S8, NFR-1)
**Kabul 6:** §4.5'in üç otomatik kanıtı CI'da ya da yerelde yeşil + elle Wi-Fi kapalı demo kaydı.

### 6.6 Faz çıkışı
S7, S8 işaretli; `docs/system.md` (mimari, API, eşlik ve gecikme tabloları, takip tablosu, keşif kapı
tablosu); README'ye sistem bölümü ve ekran görüntüsü; `v0.2.0`.

## 7. İş yükü ve takvim

Önerilen sıra (her adım bir öncekini kullanır; plan haftalara böler):

| # | İş | Tahmin |
|---|---|---|
| 1 | Motor iskeleti: Dockerfile (build/runtime), CMake, GoogleTest, CI `engine` işi | 2-3 gün |
| 2 | `preprocess` + `detector` + Python referans fikstürleri + tespit eşliği (S7'nin çekirdeği) | 3-4 gün |
| 3 | `health` + `calibration` C++ + sağlık eşliği; `r_min` seçimi ve keşif ölçümü (laptop) | 3-4 gün |
| 4 | Kalman, Macar, ByteTrack + Python referans + takip eşliği; Colab CPU (M3OT paketi); M3OT ölçümü, ADR 0008 | 5-6 gün |
| 5 | HTTP API + `baykus_bench` + gecikme ölçümü | 2-3 gün |
| 6 | UI: oynatıcı, kutular, sağlık paneli, alarm, arıza simülasyonu, gecikme grafiği | 5-7 gün |
| 7 | Compose, kapalı ağ kanıtları, CI `ui` işi, Release | 2-3 gün |
| 8 | `docs/system.md`, README, spec v4.5, `v0.2.0` | 1-2 gün |

**Toplam ≈ 5-6 hafta** (spec'te 8 hafta, 7 Ara – 31 Oca; takvimin ≈ 7 hafta önündeyiz). GPU yok; tek
Colab **CPU** oturumu (M3OT demo paketi). Laptop'a inecekler: M3OT paketi (boyut önce görülür), Faz 2'nin
val tahmin ve sağlık dosyaları (keşif ölçümü için; tahmin ≈ 70 MB).

## 8. Riskler

| Risk | Önlem |
|---|---|
| Windows'ta C++ araç zinciri | Hiç kurulmaz: derleme ve test yalnız Docker'da; CI aynı Dockerfile. |
| OpenCV sürüm farkı (C++ apt vs `opencv-python`) faz korelasyonunu ve Laplace'ı değiştirir | Sağlık eşliği toleranslı (§6.1); durum ve kapı kararı birebir. Tutmazsa OpenCV kaynaktan aynı sürümle derlenir (derleme süresi artar). |
| ORT C++ ve Python sürümleri farklı → eşlik testi kütüphane farkını ölçer | Aynı sürüm sabit (§4.1). |
| `roboflow/trackers` ByteTrack ayrıntıları makaleden farklı | C++ referansı izler; zorunlu farklar belgelenir; eşlik permütasyon dışında birebir istenir. |
| Docker Desktop (WSL2) gecikmesi yerel ölçümden kötü → Kabul 5 kalır | Olumsuz sonuç raporlanır; aşama kırılımı neyin pahalı olduğunu gösterir (ör. `health` 11 ms). Bütçe aşılırsa `health` her N karede bir çalıştırılabilir (yeni ölçüm gerektirir, bu fazda değil). |
| UI kapsamı büyür | Tek sayfa, §4.4 listesi dışında özellik yok; yeni istek Faz 5'e. |
| M3OT tek sınıf ve küçük kutular; DroneVehicle'dan farklı → takip sayıları düşük çıkar | Oracle satırı takipçiyi modelden ayırır; mutlak değer değil iki satırın farkı yorumlanır. |
| Yanıt korumalı kapı keşifte de temiz veride zararlı | Varsayılan zaten `advise`; sonuç olduğu gibi raporlanır. |

## 9. Builder'a bırakılanlar (uygulamada kararlaştırılır, plana yazılır)

- Ubuntu ve OpenCV paket sürümleri; ORT C++ paketinin (1.30.0) indirme yolu ve sha256'sı.
- Macar algoritması uygulaması (kendimiz, `O(n³)`; ya da LAPJV) — Python referansının kullandığıyla aynı
  sonucu vermesi şart.
- TrackEval mi `trackers` değerlendirmesi mi (HOTA'yı ikisi de veriyor); M3OT'un MOT biçimine dönüşüm.
- UI'nin grafik bileşeni (hafif, yerelde paketlenmiş), canvas çizim ayrıntıları, renkler.
- Alarm penceresinin (5 karenin 3'ü) yapılandırılabilir olup olmayacağı.
- `offline_check` betiğinin dili (bash ya da Python) ve CI'da compose'un nasıl koşturulacağı.
- Eşlik fikstürü olarak seçilecek M3OT kareleri (sabit liste, gerekçesiyle).

## 10. Spec'e yansıyacaklar (onaydan sonra, v4.5)

§2.3 S7/S8 metni (eşlik toleransları §6.1), §7.4 FR-E2/E3 (late_fusion ve NMS yolu Faz 3'te yok),
FR-E5 API (§4.1), FR-U eklemeleri (sağlık paneli, alarm, kapı modu), §15.1 modül listesi, §15.3 API,
§17 düzen, §18 compose ağları ve NFR-1 kanıtları, §22 Faz 3 satırı (≈ 5-6 hafta) ve H6-H9 tablolarının
bu tasarıma göre güncellenmesi, §23 risk satırları.
