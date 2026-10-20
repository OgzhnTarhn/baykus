# Faz 3 — Sistem: C++ motor, operatör ekranı, kapalı ağ

> Bu dosya `baykus-system report` ile `results/**/*.json`'dan üretildi; elle düzenlenmez. Kabul kanıtı yoksa satır "kanıt yok" ile düşer; eksik sayı `—`.

## Mimari

- **Motor (`engine/`, C++17):** ONNX Runtime 1.30.0 + OpenCV 5.0.0; tek model E3c (erken füzyon, RGB + IR 4 kanal, ONNX FP32). Çözme → sağlık (Faz 2 dedektörü, kare + zaman kuralı) → kapı → letterbox → çıkarım → kalibre skor → ByteTrack (`trackers` 2.6.1 uyarlaması, oturum başına). HTTP sunucusu yalnız compose'un iç `back` ağında.
- **Ekran (`ui/`, Next.js 16):** tek sayfa; oynatıcı, RGB / IR / yan yana görünüm, sağlık paneli, alarmlar (5 karenin 3'ü), arıza simülasyonu, kapı modu, gecikme grafiği. Tarayıcı yalnız `127.0.0.1:3000`'e konuşur; motor isteklerini Next sunucusu iletir.
- **Kurulum:** `docker compose up` (iki servis); `compose.offline.yml` bütün ağları `internal` yapar. `scripts/prepare.sh --bundle` internetsiz bilgisayara taşınacak paketi hazırlar.
- **Referans:** her C++ adımı Python referansıyla (`python/src/baykus/`) eşlik testli; CI her commit'te sınar.

## API

| Uç | Nerede | İş |
|---|---|---|
| `GET /health` | motor | durum (`ok`) |
| `GET /v1/model` | motor | model ve dosya sha256'ları |
| `POST /v1/detect` | motor | RGB + IR (multipart) → tespitler, sağlık, kapı, izler, süreler |
| `POST /v1/session/reset` | motor | oturumun takipçisini ve zaman penceresini sıfırlar |
| `GET /api/sequences`, `GET /api/frame/…` | ekran | demo dizileri ve kare görüntüleri |
| `POST /api/detect`, `POST /api/session/reset` | ekran | motora iletir (arızalı kanal multipart) |

Hatalar: eksik parça, farklı boyut, bozuk görüntü → 400; sınır üstü görüntü → 413 (Kabul 8).

## Kabul 1-8

| Kabul | Ölçüt | Sonuç | Ayrıntı | Kanıt |
|---|---|---|---|---|
| 1 | tespit eşliği (C++ ↔ Python) | ✅ | 17 fikstür; ham ORT çıktısında en büyük fark 0.000000 | `results/engine/parity_detect.json` |
| 2 | letterbox baytları | ✅ | 17/17 fikstürde tensör sha256 aynı | `results/engine/parity_detect.json` |
| 3 | sağlık eşliği (kare + zaman kuralı) | ✅ | 17/17 kare durumlar aynı; zaman kuralı uçtan uca ve birim geçti | `results/engine/parity_health.json` |
| 4 | takip eşliği | ✅ | karar `pass`; 1_08 600/600 kare, 2_08 600/600 kare | `results/engine/parity_track.json` |
| 5 | gecikme p95 ≤ 100 ms | ✅ | motor p95 82.2 ms ≤ 100 ms (Docker, HTTP hariç) | `results/bench/e3c_onnx_fp32_engine_laptop-docker.json` |
| 6 | kapalı ağ (otomatik + Wi-Fi kapalı) | ✅ | otomatik kanıt 1-4 geçti (tarayıcıdan dış istek 0); Wi-Fi kapalı demo kayıtlı | `results/system/offline.json` |
| 7 | ekran (FR-U1-U4, alarmlar, kapı) | ✅ | 20/20 ekran maddesi | `results/system/manual.json` |
| 8 | API sağlamlığı | ✅ | 15/15 hata durumu; 1000 rastgele istekten sonra `/health` ok | `results/system/api.json` |

Kabul 1-8'in hepsi çıkışı engeller (tasarım §6.7); faz çıkışına **izin veriyor**.

## Gecikme (Kabul 5)

| Aşama | Bellek içi p50 / p95 (ms) | HTTP kipi p50 / p95 (ms) |
|---|---|---|
| decode | 6.51 / 7.68 | 6.71 / 7.60 |
| health | 8.72 / 10.77 | 7.56 / 9.43 |
| pre | 3.18 / 4.39 | 3.25 / 4.44 |
| infer | 45.88 / 62.11 | 45.72 / 62.06 |
| post | 0.00 / 0.01 | 0.00 / 0.01 |
| track | 0.13 / 0.20 | 0.12 / 0.18 |
| total | 64.81 / 82.19 | 63.31 / 80.80 |
| HTTP dahil gidiş-dönüş | — | 65.84 / 84.68 |

Bellek içi 14.8 FPS, tepe RSS 287 MB; AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, Docker Desktop 4.62.0 (219486), güç AC (Win32_Battery BatteryStatus 2). Protokol: 50 ısınma + 500 ölçüm, 20 görüntü, `gate=advise`, takip açık.

## M3OT tespit doğruluğu (motor, sınıfsız `vehicle`, IR etiketleri)

Test bölümü sistem dondurulduktan sonra bir kez ölçüldü; val aynı komutların provası.

| Bölüm | Dizi | AP50 | ECE ham → kalibre | Çift kutu | Kare |
|---|---|---|---|---|---|
| val (prova) | 1_08 | 0.413 | 0.204 → 0.209 | 0.004 | 600 |
| val (prova) | 2_08 | 0.382 | 0.221 → 0.213 | 0.013 | 600 |
| val (prova) | birleşik | 0.396 | 0.213 → 0.209 | 0.009 | 1200 |
| **test** | 1_03 | 0.896 | 0.368 → 0.350 | 0.001 | 480 |
| **test** | 2_03 | 0.820 | 0.352 → 0.343 | 0.024 | 480 |
| **test** | birleşik | 0.860 | 0.361 → 0.347 | 0.012 | 960 |

Faz 1 E3c'nin M3OT test AP50'si 0.889 (PyTorch + `agnostic_nms`, E7). Fark tek bir nedene bağlanmaz: ONNX, uçtan uca çıktının 300 kutu sınırı, sınıflar arası çift kutu, NMS yokluğu (tasarım §4.3). Kalibrasyon DroneVehicle val'den; M3OT ECE'si alan dışı davranışı gösterir.

## Takip (M3OT)

| Bölüm | Satır | Dizi | HOTA | MOTA | IDF1 |
|---|---|---|---|---|---|
| val (prova) | oracle | 1_08 | 0.988 | 0.992 | 0.991 |
| val (prova) | oracle | 2_08 | 0.964 | 0.993 | 0.953 |
| val (prova) | oracle | birleşik | 0.974 | 0.993 | 0.969 |
| val (prova) | motor | 1_08 | 0.192 | 0.122 | 0.272 |
| val (prova) | motor | 2_08 | 0.238 | 0.173 | 0.312 |
| val (prova) | motor | birleşik | 0.220 | 0.152 | 0.296 |
| **test** | oracle | 1_03 | 0.977 | 0.982 | 0.984 |
| **test** | oracle | 2_03 | 0.916 | 0.979 | 0.894 |
| **test** | oracle | birleşik | 0.949 | 0.981 | 0.941 |
| **test** | motor | 1_03 | 0.535 | 0.591 | 0.744 |
| **test** | motor | 2_03 | 0.432 | 0.497 | 0.599 |
| **test** | motor | birleşik | 0.489 | 0.547 | 0.677 |

Oracle: M3OT'un kendi kutuları skor 1 ile (takipçinin modelden bağımsız ölçümü); motor: `baykus_detect` tespitleri, takip katmanı ham ≥ 0,1. `trackers` 2.6.1 varsayılan parametreleri, IR etiketleri, sınıfsız. Mutlak değer değil iki satırın farkı yorumlanır (ADR 0008).

## Kare kuralıyla kapı (DroneVehicle val)

Etiket: **val'de keşif, PyTorch E3c** (sevk edilen ONNX değil); Faz 3'ün kabul ölçütü değil, ürün varsayılanını (`gate=advise`) değiştirmez.

| Koşul | Kapı oranı | E3c mAP50 | E3c + kapı | Fark (puan) [%95 GA] |
|---|---|---|---|---|
| clean | %0.7 | 0.851 | 0.848 | -0.2 [-0.5, -0.0] |
| darken-0.5 | %0.7 | 0.841 | 0.839 | -0.2 [-0.6, -0.0] |
| darken-0.25 | %0.7 | 0.836 | 0.835 | -0.1 [-0.3, -0.0] |
| darken-0.1 | %0.5 | 0.832 | 0.831 | -0.1 [-0.3, -0.0] |
| noise-5 | %0.6 | 0.841 | 0.839 | -0.2 [-0.4, -0.0] |
| noise-10 | %0.5 | 0.770 | 0.769 | -0.1 [-0.4, -0.0] |
| noise-20 | %100.0 | 0.535 | 0.720 | +18.4 [+16.0, +21.0] |
| blur-3 | %0.6 | 0.831 | 0.830 | -0.1 [-0.2, -0.0] |
| blur-7 | %0.4 | 0.744 | 0.744 | -0.0 [-0.1, -0.0] |
| blur-11 | %0.3 | 0.652 | 0.652 | -0.0 [-0.0, +0.0] |
| misalign-5 | %1.9 | 0.834 | 0.831 | -0.3 [-0.6, +0.0] |
| misalign-10 | %6.9 | 0.696 | 0.690 | -0.5 [-0.9, -0.1] |
| misalign-20 | %83.6 | 0.266 | 0.622 | +35.7 [+31.9, +39.6] |
| misalign-40 | %86.7 | 0.090 | 0.564 | +47.4 [+43.4, +51.8] |
| drop_rgb | %100.0 | 0.829 | 0.829 | +0.0 [+0.0, +0.0] |
| drop_ir | %100.0 | 0.720 | 0.720 | +0.0 [+0.0, +0.0] |

Faz 2 Kabul tanımlarıyla: K1 temiz fark -0.0022 ≥ -0.005 ✅ · K2 bozuk IR kazancı GA [0.313, 0.367] ✅ · K3 temiz yanlış alarm %0.7 ≤ %2.0 ✅.

- misalign-20 yakalama: tümü %83.6, karanlık %58.8, aydınlık %93.4
- misalign-40 yakalama: tümü %86.7, karanlık %61.4, aydınlık %96.6

## M3OT sağlık kontrolü (yalnız raporlanır)

**val (V, demo seçimi)**

| Kural | temiz hizasız | misalign-20 | drop_ir IR kayıp | noise-20 IR gürültülü | temiz AP50 auto / off |
|---|---|---|---|---|---|
| Faz 2 | %39.7 | %74.8 | %99.7 | %99.7 | 0.148 / 0.396 |
| kare (`r_min`) | %0.0 | %0.0 | %99.7 | %99.7 | 0.396 / 0.396 |
| kare + zaman (ürün) | %0.0 | %0.0 | %99.7 | %99.7 | 0.396 / 0.396 |

Demo ölçütü (temiz ≤ %2, misalign-20 ≥ %90, gecikme ≤ 10): 1_08 tutmadı, 2_08 tutmadı.

**test (T)**

| Kural | temiz hizasız | misalign-20 | drop_ir IR kayıp | noise-20 IR gürültülü | temiz AP50 auto / off |
|---|---|---|---|---|---|
| Faz 2 | %35.1 | %83.0 | %99.6 | %99.6 | 0.569 / 0.860 |
| kare (`r_min`) | %0.0 | %0.4 | %99.6 | %99.6 | 0.860 / 0.860 |
| kare + zaman (ürün) | %0.0 | %0.0 | %99.6 | %99.6 | 0.860 / 0.860 |

Demo ölçütü (temiz ≤ %2, misalign-20 ≥ %90, gecikme ≤ 10): 1_03 tutmadı, 2_03 tutmadı.

## Sınırlamalar

- **Doğal kayma:** M3OT'ta RGB ve IR kameralar arasında doğal kayma var ve faz korelasyonu yanıtı düşük; ürün kuralı (`r_min` + zaman) temizde yanlış alarm vermiyor ama gerçek 20 px kaymayı da yakalamıyor. Sabit bir doğal kayma zaman kuralını geçer; eşik ayarlanmadı.
- **Titreyen arıza:** kayma vektörü sabit kalmayan gerçek arıza (titreyen bağlantı) zaman kuralını geçmez; kayıp ve gürültü alarmı 5 karenin 3'ü kuralıyla gecikmeli açılır.
- **Kalibrasyonun alanı:** kalibrasyon DroneVehicle val'den öğrenildi; M3OT'ta skorlar fazla iyimser (ECE yüksek) ve kalibrasyon az düzeltiyor.
- **Karşılaştırılamaz hız:** H5'teki 60 ms Windows yerel Python ORT ölçümü; motor Docker'da (WSL2) ve çözme + sağlık dahil. İki sayı yan yana konmaz.
- **Sınıflar:** model DroneVehicle'ın 5 sınıfıyla eğitildi (car, truck, bus, van, freight_car); motosiklet yok. Yoğun sahnede (M3OT val, kare başına ≈ 11 araç) tespit belirgin düşük.
- **Az test verisi:** M3OT test iki dizi, ardışık video kareleri; güven aralığı yok.
