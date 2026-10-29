# Faz 4 kod incelemesi (f3-freeze..f4, v0.3.0 adayı)

Tarih: 2026-10-29 · İnceleyen: Claude · Kapsam: `f3-freeze..0faff5d` (dal `f4`; `v0.3.0` etiketi henüz
yok, f4 main'e birleştirilmedi), `engine/`, `python/src/baykus/motion/`, `ui/` (130 dosya, +19 554 / −240).
Test seti yeniden ölçülmedi; düzeltmeler ölçülen sayıları (val, test) değiştirmez.

**Okunan:** `engine/src/{server,sessions,runtime,zones,event_log,track_summary,ego,reloc,motion,frozen,
pipeline,tiles,detector,health}.cpp`, ilgili başlıklar, `engine/include/baykus/polygon.hpp`,
`engine/apps/server.cpp`; `ui/app/api/**`, `ui/lib/{engine,session,client,player,alarms,timeline,
snapshots,mosaic,editor,zone-sync,exporting}.ts`, `ui/components/{Console,ZoneEditor,ZoneList,
ExportMenu}.tsx`, `MapPanel.tsx` (kuyruk çizimi); `python/src/baykus/motion/{measure,report}.py`.
**Yüzeysel / okunmayan:** `motion/{fixtures,cli,audit,parity,drift,labels,explore,ego,reloc,state}.py`
(C++ eşliği Kabul 1 ile CI'da sınanıyor), `engine/apps/{sequence,clip,bench}.cpp`, `bench.cpp`,
`cpu.cpp`, `motion_config.cpp`, `MapPanel.tsx` kalanı, testlerin çoğu.

## Önemli

### Ö1 · Bölge eşitleme: üst üste gelen düzenlemede bir değişiklik sessizce kaybolur
- **Yer:** `ui/lib/zone-sync.ts:72-87` (`settle`), `ui/lib/editor.ts` (`load`).
- **Ne oluyor:** PUT yoldayken yapılan düzenleme `pending`'e alınır. İlk PUT dönünce `settle` taslağı
  motorun **eski** kümesiyle yükler (`draft.load(result.zones)`), sonra bekleyeni gönderir. Ekran bir an
  eski kümeye döner; bu arada yapılan üçüncü düzenleme bu eski taslağın üstüne kurulur ve gönderilen
  küme ikinci düzenlemeyi içermez → motor ikinci düzenlemeyi hiç görmez. `load` ayrıca açık çizimi
  (`tool`, `draftPoints`) siler.
- **Tetikleyici:** oynatma sırasında motor PUT'u oturum kilidinde çıkarım bitene kadar bekletir
  (≈ 100 ms); bu sürede iki köşe sürükleme / ad + yön değişikliği yeterli.
- **Doğrulama:** `zone-sync.test.ts` birleştirme testi ilk yanıttan sonra taslağı denetlemiyor;
  düzeltmeden önce düşen test (aşağıda durum tablosu).

### Ö2 · `track=false` + `session`: motor ego ve bölgeleri boş iz listesiyle yürütür
- **Yer:** `engine/src/runtime.cpp:86` (ego bloğu `s && s->motion` ile açılıyor, `req.track`
  denetlenmiyor), `:129`, `:149`.
- **Ne oluyor:** Ekran takip kapalıyken de `session` gönderir (Faz 3: zaman kuralı penceresi). Motor
  bu karelerde ego'yu, hareketi ve bölgeleri çalıştırır; takipçi güncellenmediği için `tracks[]` boştur.
  30 kare sonra içerdeki her iz için `exit_lost`, yasak bölge alarmları için `alarm_clear` günlüğe yazılır,
  özetler "bitmiş" olur. Bunlar dışa aktarılan günlükte gerçek olay gibi durur. Tasarım §5.7: ego,
  hareket, bölgeler, günlük yalnız `track=true` + `session` ile.
- **Doğrulama:** `MotionRuntimeTest` yalnız oturumsuz durumu sınıyor (`NoSessionNoEgo`); düzeltmeden
  önce düşen test.

### Ö3 · Kare dışı günlük kayıtları ekrana ulaşmaz: alarm bandı kapanmaz, geçmiş karesi yanlış
- **Yer:** `engine/src/server.cpp` PUT zones / alarms/ack ve `sessions.cpp` `scope=tracks` kayıtları
  yalnız günlüğe yazılır; `ui/components/Console.tsx:302-333` çizelgeyi ve `ZoneAlarmBook`'u yalnız
  `/v1/detect` yanıtının `events[]`'iyle besler; `ui/lib/alarms.ts:124`.
- **Ne oluyor:** Açık alarmlı bir bölge silinince / değiştirilince motor `alarm_clear` yazar ama bu
  kayıt hiçbir `/v1/detect` yanıtında gelmez: çizelgedeki alarm bandı sonsuza dek açık kalır
  (`to: null`), alarm geçmişinde kapanış karesi o karenin başka olayından ya da `since_frame`'den
  uydurulur. `alarm_ack`, `zones_updated`, `session_reset_tracks` da çizelgede / olay listesinde
  görünmez; dışa aktarılan dosyada görünür (ekran ile dosya tutarsız).

## Küçük

### K1 · İz özeti: kutu boyları sınırsız birikir, her odak isteğinde baştan sıralanır
- **Yer:** `engine/src/track_summary.cpp:115` (`rec.sizes.push_back` her gözlemde), `:195`
  (`median` kopyalayıp sıralar; `focus` ile her karede).
- **Ne oluyor:** Uzun yaşayan iz (park etmiş araç) için bellek doğrusal büyür (30 kare/sn'de saatte
  ≈ 0,9 MB / iz; bitmiş 500 iz de saklanıyor) ve odaklı her istekte O(n log n) sıralama gecikmeye
  eklenir. Tasarım §5.12 belleği "yaşayan izler + bitmiş son 500 iz" diye sınırlıyor.

### K2 · Harita: bitmiş izlerin kuyrukları hiç silinmez
- **Yer:** `ui/lib/mosaic.ts:139-146`; çizim `ui/components/MapPanel.tsx:360-367`.
- **Ne oluyor:** Kuyruk iz başına 150 noktayla sınırlı ama görülmeyen izin kuyruğu katmanda kalır ve
  her karede çizilir. Uzun oturumda harita eski gri kuyruklarla dolar, bellek ve çizim süresi büyür.
  Tasarım §5.4: "iz kuyrukları (son 150 kare)".

### K3 · "Olaylar (CSV, Excel)": formül kaçışı yok
- **Yer:** `ui/lib/exporting.ts:102`.
- **Ne oluyor:** Bölge adı operatör metnidir (`=HYPERLINK(...)`, `+`, `-`, `@` ile başlayabilir);
  Excel için üretilen dosya çift tıklamada formül olarak çalıştırır (CSV enjeksiyonu). Motorun CSV
  sözleşmesi değişmemeli; kaçış yalnız Excel kopyasında.

### K4 · Bölge nesli ve kare numarası doğrulaması gevşek
- **Yer:** `engine/src/server.cpp:240` (`gen ≤ en büyük nesil`), `engine/src/zones.cpp:106`,
  `parse_zone_defs` (`get<int>()`), `engine/src/server.cpp:190` (`frame`).
- **Ne oluyor:** (a) Birleşmiş (`merged_into`) bir nesle bölge kabul edilir; o nesil bir daha geçerli
  olmayacağı için bölge kalıcı askıda kalır. Tasarım §5.5: "`gen` var olan bir nesil". (b) `gen` /
  `dwell_frames` için `int` dışı büyük tam sayı sessizce kesilir (2³²+1 → 1). (c) Negatif `frame`
  kabul edilir; yeniden konumlamadaki `|Δframe|` farkı `int` taşmasına (UB) açık.

## Sağlam bulunanlar (kısa)
- Bölge kuralları (§5.5) satır satır tasarımla uyumlu: art arda / `yok` boşluğu, `exit_lost` karesi,
  `stopped` kalış başına bir kez, sayım hattı kararlı taraf ve uzantıdan dolanma, alarm açma/temizleme
  sırası, askıya alma.
- Eşzamanlılık: bütün oturum uçları oturum kilidiyle; kilit sırası Faz 3'teki gibi (tablo → oturum →
  çıkarım). Bilinmeyen oturum 404, oturum açmaz.
- S1, S3, S6, S11, S12, K7 düzeltmeleri yerinde; motor isteği zaman aşımı, yanıt damgası çalışıyor.
- CSV RFC 4180 alıntılama, `send_json` geçersiz UTF-8'de düşmüyor, gövde sınırı 413.
- Python ölçüm / rapor: tanımsız BA (test'te duran araç yok) `—` ve iddia "tutmadı" olarak ele alınıyor.
