# Faz 3 kod incelemesi (main, v0.1.1..v0.2.0)

Tarih: 2026-10-19 · İnceleyen: Claude · Kapsam: `main` = `v0.2.0` (`6f60c39`), Faz 3 değişiklikleri
(`v0.1.1..v0.2.0`, 115 commit). Kod değiştirilmedi; bu dosya yalnız not.

**Okunan:** `engine/src/*` (server, runtime, sessions, pipeline, image_io, health, temporal,
detector, preprocess, detections, tracker, kalman, calibration, bench, egress, models, files),
`engine/apps/{server,detect}.cpp`, `ui/lib/*`, `ui/app/api/**`, `ui/components/{Console,Player,Overlay}.tsx`,
`python/src/baykus/{health/guard,health/temporal,health/gate,health/m3ot,track/runner,track/cli,
engine_ref/measure,engine_ref/pipeline,system/report,data/demo}.py`, `scripts/{prepare.sh,m3ot_measure.sh,
offline_check.mjs,egress_check.mjs,engine_test.sh}`, `docker/*.Dockerfile`, `docker-compose.yml`,
`compose.offline.yml`, CI başlığı.
**Okunmayan / yüzeysel:** `engine_ref/fixtures.py` (fikstür üreticisi), `engine/src/{parity,lsap}.cpp`,
`engine/apps/{bench,parity_*,track}.cpp`, test dosyalarının çoğu, notebook'lar, `ui/components` kalanı.

Önceki dış incelemenin (ledger "Final") düzeltilmiş bulguları tekrar edilmedi; ertelenen S-bulguları
en sonda "hâlâ açık" olarak listelendi.

## Durum (2026-10-20, dal `f3-fix`; Faz 4 Görev 1, dal `f4`)

| Bulgu | Durum | Commit | Doğrulama |
|---|---|---|---|
| Ö1 | düzeltildi | `676ad78` | önce düşen 2 vitest + hata yolu testi; UI 65/65, lint, tip |
| K1 | düzeltildi (UI: takip yeniden açılınca oturum sıfırlanır) | `af00651` | önce düşen vitest; UI 65/65 |
| K2 | düzeltildi (sinyalle durma bayrağı; dinleme hatası → çıkış 1) | `72e0402` | motor 102/102; `docker stop` → çıkış 0. Hata yolu (sinyalsiz `false`) elle tetiklenemedi |
| K3 | düzeltildi (runtime'da yalnız server, bench, detect, track) | `feba050` | imajda 4 ikili, `ldd` eksiksiz; kapalı ağ kanıtı 1-4 geçti |
| S1 | düzeltildi (Faz 4 Görev 1: çözme oturumdan önce) | `bd36cad` | önce düşen `ServerTest.BadFirstRequestKeepsSessionNew`; motor 104/104 |
| S3 | düzeltildi (yarı çözünürlükte kenar < 2 px → kayma yok; C++ ve Python) | `7db79dc` | önce düşen `Health.TinyImageNoThrow`, `test_misalign_vector_tiny_image_none` |
| S6 | düzeltildi (`compare` karar `fail` → çıkış 1; istisnalı geçiş 0) | `68d6206` | önce düşen `test_compare_exit_code_on_fail` |
| S11 | düzeltildi (motor isteği 10 sn sonra 504 "motor zaman aşımı") | `acc30cf` | önce düşen `detect aborts after timeout` |
| S12 | düzeltildi (yanıt damgası: oturum + dönem; eski yanıt izlemeye yazılmaz) | `9786e02` | önce düşen `session.test.ts` |
| K7 | düzeltildi (bench protokolü ve sürümler JSON'dan; `docs/system.md` yeniden üretildi) | `970179f` | önce düşen 3 rapor testi |
| K4-K6, K8-K11 | açık | — | — |

## Önemli

### Ö1 · Oynatıcı: tek başına oturum sıfırlaması sürerken "Oynat" → oynatma başlamıyor
- **Yer:** `ui/lib/player.ts:80-92` (`resetOnly`), `:168-173` (`play`), `:183-186` (`seek` → `start`).
- **Ne oluyor:** `resetOnly()` `busy = true` yapar ama bitince devam yolunu (`proceed`) çağırmaz.
  Bu sırada `play()` gelirse durum `playing` olur, `busy` olduğu için istek başlatılmaz; sıfırlama
  bitince kimse döngüyü açmaz. Ekran "Durdur" düğmesini gösterir, hiç kare istenmez; tekrar "Oynat"
  `state === "playing"` yüzünden etkisizdir. Kurtulmak için Durdur + Oynat gerekir.
  Aynı kökten: sıfırlama sürerken konum kaydırıcısıyla atlama `pendingSeek`'e yazılır ve
  tüketilmez; sonradan ilk istekten sonra beklenmedik bir atlama olarak çalışır.
- **Tetikleyiciler:** (a) dururken "Oturumu sıfırla" + hemen Oynat / boşluk / kaydırıcı;
  (b) istek yoldayken duraklatıp sıfırla basılırsa `proceed` → `resetOnly` yolu (aynı durum).
- **Doğrulama:** geçici vitest (`requestReset()` → `play()` → sıfırlama çözülür → `detect` hiç
  çağrılmadı, durum `playing`) düştü; test silindi, depoda iz yok.
- **Öneri:** `resetOnly` bittiğinde `proceed(deps.now())` ile aynı devam yoluna girsin (ya da
  `play`/`seek` `busy` iken bir "sıfırlama sonrası başlat" bayrağı bıraksın). Önce RED testi:
  "play pressed during a standalone reset starts playback" ve "seek during a standalone reset
  is applied once".

## Küçük

### K1 · Takip aç/kapa oturumu sıfırlamıyor; takipçi boşluklu kare görüyor
- **Yer:** `engine/src/runtime.cpp:248-256` (track=false iken `tracker.update` çağrılmaz),
  `ui/lib/session.ts:167-170` (sıfırlama yalnız konuma bağlı), `ui/components/Console.tsx:184-186`.
- **Ne oluyor:** Takip kapatılıp N kare sonra açılınca takipçi tek `predict` adımıyla N kare
  sonraki tespitlerle eşleşir (sabit hızlı Kalman, dt = 1 varsayımı). Kimlikler yanlış devam eder
  ya da kopar. `session.ts`'teki "takipçi her istekte tam bir kare ilerler" varsayımı bozulur.
- **Öneri:** `track` false → true geçişinde (ya da her değişimde) `player.requestReset()`; ya da
  motor track=false iken de takipçiyi güncelleyip yalnız kimlikleri yanıta koymasın.

### K2 · `baykus_server` dinleme hatasında 0 ile çıkıyor
- **Yer:** `engine/apps/server.cpp:83` — `server.listen_after_bind()` dönüşü yok sayılıyor;
  `main` her durumda 0 döner. Docker `unless-stopped` yine yeniden başlatır ama günlükte hata
  görünmez. **Öneri:** `return server.listen_after_bind() ? 0 : 1;` ve kısa ileti.

### K3 · Runtime imajına test ikilileri de giriyor
- **Yer:** `docker/engine.Dockerfile:134` — `COPY --from=build /build/bin/ /usr/local/bin/`
  bütün ikilileri (`baykus_tests`, `baykus_fixture_probe`, parity araçları) kopyalıyor. Testler
  derleme imajında koşuyor (`scripts/engine_test.sh`), runtime'da gereksiz; paket (`docker save`)
  boyutuna ekleniyor. **Öneri:** yalnız `baykus_server`, `baykus_detect`, `baykus_track`,
  `baykus_bench` (+ kullanılıyorsa `baykus_buildinfo`) kopyalansın.

### K4 · Paketteki `configs/engine/` kullanılmıyor
- **Yer:** `scripts/prepare.sh:76-79` paket klasörüne `configs/engine/{models,tracker}.json`
  kopyalıyor; `docker-compose.yml` bunu bağlamıyor, motor imajdaki `/etc/baykus/*.json`'u okuyor.
  Paketteki `tracker.json`'u düzenleyen biri etkisini görmez. **Öneri:** ya kopyalamayı kaldır ya da
  compose'da `./configs/engine:/etc/baykus:ro` bağla (models.json sha denetimi aynı kalır).

### K5 · compose'da sağlık denetimi yok
- **Yer:** `docker-compose.yml:28` — `depends_on: [engine]` koşulsuz; motor modelleri yükleyip
  sha256 denetlerken ekran açılıyor, ilk istekler 502 ("motor yanıt vermiyor"). Ekran bunu
  gösteriyor, çökme yok. **Öneri:** motora `healthcheck` (`/health`) + `condition: service_healthy`.
  (Runtime imajında curl yok; `baykus_server --health-check` gibi küçük bir alt komut gerekir.)

### K6 · Arıza özeti (`fault`) gönderiliyor ama hiçbir yerde kullanılmıyor
- **Yer:** `ui/components/Console.tsx:147` meta'ya `fault: faultSummary(...)` koyuyor;
  `ui/app/api/detect/route.ts:14-25` `parseMeta` bunu atıyor. `lib/faults.ts:192` yorumu "kayıt ve
  gösterim için" diyor ama kayıt yok. **Öneri:** ya sunucu günlüğüne yaz ya da alanı kaldır.

### K7 · Sistem raporunda JSON'dan gelmeyen sabit sayılar
- **Yer:** `python/src/baykus/system/report.py:1-4` "bütün sayılar JSON'lardan" diyor; ama
  `_latency` protokol satırı ("50 ısınma + 500 ölçüm, 20 görüntü, gate=advise") sabit — bench
  JSON'unda `protocol` alanı var; `ARCH` (ONNX Runtime 1.30, OpenCV 5.0 — bench JSON'unda
  `system.onnxruntime/opencv` var), `LIMITS` ("≈ 11 araç", "H5'teki 60 ms", "iki dizi") sabit.
  CLAUDE.md "kanıtsız sayı yazma" kuralına yakın. **Öneri:** protokol ve sürümleri JSON'dan oku;
  kalan sayılara kaynak dosyası yaz ya da docstring'i düzelt.

### K8 · Faz 1 M3OT dosyası ada göre son seçiliyor
- **Yer:** `report.py:35,48-49` — `sorted(glob("m3ot/*e3c*_m3ot_test_ir.json"))[-1]`. Şimdi tek
  eşleşme var; ikinci bir e3c koşusu (ör. başka tarih/commit) eklenirse sessizce o seçilir.
  **Öneri:** dosya adını sabitle ya da birden çok eşleşmede hata ver.

### K9 · `baykus_detect --gate auto` ürün kapısını yansıtmıyor
- **Yer:** `engine/src/runtime.cpp:274-308` — `detect_sequence` oturumsuz çalışır, zaman kuralı
  uygulanmaz; `auto`'da kapı tek kare kuralıyla verilir (üründe oturumla zaman kuralı). Ölçümler
  `advise` ile yapıldığı için sonuçlar etkilenmedi. **Öneri:** `auto`'yu reddet ya da dizi için
  oturum aç; en azından kullanım metnine yaz.

### K10 · `gt_mot` GT'yi dizinin karelerine süzmüyor (gizli varsayım)
- **Yer:** `python/src/baykus/track/runner.py:112-123`; `oracle_frames` (`:126-141`) süzüyor.
  index'te olmayan karelerde GT satırı olsaydı her iki satır da FN alırdı. Şimdiki dev/test
  verisinde fark 0 (dört dizide de "index dışı GT karesi" 0, yerelde sayıldı). **Öneri:** aynı
  süzgeç ya da kare kümesi farklıysa hata.

### K11 · Temizlik
- `engine/include/baykus/runtime.hpp:363` `tracker_cfg_` hiç okunmuyor.
- `scripts/m3ot_measure.sh:13,19` `split` atanıyor, yalnız echo'da kullanılıyor.

## Önceki incelemeden ertelenen, hâlâ açık (kodda yeniden görüldü)
- **S1** `runtime.cpp:234-237` oturum çözümden önce açılıyor; bozuk ilk istekte `session_new` kaybolur.
- **S3** çok küçük görüntü: `health.cpp:57` yarı çözünürlükte 0 boyut → `cv::Exception` → 500.
- **S6** `track/cli.py:304-329` `compare` karar `fail` olsa da çıkış 0.
- **S11** `ui/lib/engine.ts:119` motor isteğinde zaman aşımı yok.
- **S12** `Console.tsx:150-158` eski dizinin geç gelen yanıtı alarm/şerit/gecikmeye yazılıyor
  (sonraki istekteki sıfırlamada temizleniyor).
- S5, S7, S8, S9: değişmedi (ledger'daki tanımlar geçerli).

## Sağlam bulunanlar (kısa)
- Hata sınırları: başlıktan boyut okuma (413) çözmeden önce, `guarded` tür → kod eşlemesi, httplib
  hata/istisna işleyicileri JSON.
- Eşzamanlılık: oturum başına kilit + tek ORT kilidi, `shared_ptr` ile çıkarılan oturum güvenli,
  kilit sırası tutarlı (tablo → oturum → çıkarım), kilitlenme yolu görmedim.
- UI yol geçişi: dizi adı deseni + kare `index.frames` içinde + kanal beyaz listesi; test bölümü iki
  katmanda saklanıyor.
- Python `select_r_min`, `splice_condition` ön koşulu, `MisalignWindow` C++ ↔ Python aynı tanım.
