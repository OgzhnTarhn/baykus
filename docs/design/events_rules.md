# Faz 4 olay günlüğü, araç özeti, oturum uçları: kesin kurallar (Görev 15; tasarım faz4.md §5.7, §5.12, §5.13)

Python referansı yok; davranış bu kurallar + C++ testleri. Parametreler `LogCfg` (max_events 20000),
`SummaryCfg` (max_segments 200, ended_keep 500), `ZonesCfg.body_max_bytes` (65536). Bölge kuralları
docs/design/zones_rules.md.

## EventLog (engine/include/baykus/event_log.hpp, src/event_log.cpp)
- `explicit EventLog(const LogCfg&)`.
- `nlohmann::json append(ZoneEvent e, const std::string& zone_name, const std::string& zone_kind)`:
  seq = son seq + 1 (oturumda 1'den başlar), kaydı saklar ve kaydı döner. Kayıt sayısı max_events'i
  aşarsa en eski çıkar, dropped++.
- Kayıt JSON'u (anahtar sırası sabit değil, alanlar sabit): {"seq", "n", "frame", "kind", "zone_id",
  "zone_name", "zone_kind", "track_id", "dir", "gen", "alarm_id"}. Boş zone_id / zone_name / zone_kind →
  null; track_id / dir / alarm_id yoksa null.
- `nlohmann::json since(long seq) const` → {"events": [seq'ten büyük kayıtlar, artan], "last_seq":
  verilen en büyük seq (hiç yoksa 0), "dropped"}.
- `std::string csv_since(long seq) const`: başlık `seq,n,frame,kind,zone_id,zone_name,zone_kind,track_id,dir,gen,alarm_id`
  ve her kayıt bir satır; satır sonu "\r\n" (başlık dahil); null → boş alan; virgül, çift tırnak, CR ya
  da LF içeren alan çift tırnakla sarılır, içteki tırnak ikilenir (RFC 4180). UTF-8 olduğu gibi.
- `long dropped() const` · `long last_seq() const` · `void reset()` (kayıtlar, seq ve dropped sıfır).

## Harita olayları (Runtime, her karede bölge olaylarından ÖNCE günlüğe)
Oturum son ego'yu tutar (`has_ego`, `last_gen`, `last_provisional`). Bu karenin ego'su `e`:
1. `e.merge` varsa: geçici nesil önceki karede yazılmadıysa (`!(last_provisional && last_gen ==
   merge.from)`, açıldığı karede birleşme) önce `map_provisional` (gen = merge.from); sonra
   `map_relocalized` ve `map_merged` (ikisinde gen = merge.to).
2. Değilse, önceki ego geçiciydi (`last_provisional`) ve `e.map_gen == last_gen` ve `!e.provisional`:
   `map_new_generation` (gen = e.map_gen) — pencere birleşmeden kapandı.
3. Değilse, önceki geçiciydi ve `e.map_gen != last_gen`: `map_new_generation` (gen = last_gen); sonra
   `e.provisional` ise `map_provisional` (gen = e.map_gen).
4. Değilse `e.provisional` ve (`!has_ego` ya da `e.map_gen != last_gen`): `map_provisional`.
Harita olaylarında zone / track / dir / alarm alanları boş; n = oturum n'i, frame = istek frame'i
(yoksa n). Sonra ego bilgisi güncellenir. Ardından `zones.update` olayları, her biri bölgenin adı ve
türüyle (`Zones::find(id)`). Bu karenin günlük kayıtları yanıtta `events`.

## Oturum günlük olayları
- `scope=tracks` sıfırlaması: `session_reset_tracks` (n = oturum n'i, frame = son istek frame'i, gen =
  son ego nesli; hiç ego yoksa 1). `scope=all`: günlük ve özet silinir (olay yazılmaz).
- `PUT zones`: önce `set`'in `cleared` alarm_clear'ları (ad / tür ESKİ tanımlardan), sonra
  `zones_updated` (zone alanları boş, n / frame / gen yukarıdaki gibi).
- `POST alarms/ack`: `alarm_ack` (zones.ack çıktısı, bölge adı / türüyle).

## TrackSummaries (engine/include/baykus/track_summary.hpp, src/track_summary.cpp)
- `explicit TrackSummaries(const SummaryCfg&)`;
  `void update(int n, int frame, const std::vector<TrackOut>& tracks, const std::map<int, TrackMotion>& motion,
  const std::vector<ZoneEvent>& events, const EgoResult& ego)`; `std::optional<nlohmann::json> get(int id) const`;
  `void reset()`.
- İz başına (API kimliği), bu karede `tracks`'te olan her iz için:
  - yeni iz: first_frame = frame; ilk bölüm `{state: hareket, from_frame: frame, to_frame: frame}`,
    motion_since_frame = frame.
  - last_frame = frame; sınıf sayacı[cls_id]++; class = en çok sayılan cls_id (eşitlikte küçük).
  - hareket m = motion[id].motion (kayıt yoksa "unknown"). m mevcut hareketten farklıysa yeni bölüm
    {m, frame, frame} ve motion_since_frame = frame; aynıysa son bölümün to_frame = frame.
    m == stationary → stationary_frames++.
  - bölüm sayısı max_segments'i aşarsa en eski iki bölüm birleşir: from = ilkinin from, to = ikincinin
    to, state = süresi (to − from + 1) uzun olanın (eşitlikte ilkinin).
  - yol: `ego.merge` varsa ve izin son harita noktası merge.from neslindeyse nokta `apply(T, p)` ile
    taşınır, nesli merge.to olur. `ego.ok` ve `map_xy` varsa: boy s = √(w·h) · scale(frame_to_map)
    (kutu tam çözünürlük) listeye; son nokta aynı nesildeyse yol += ‖map_xy − son nokta‖; son nokta =
    map_xy, nesli = ego.map_gen. Başka nesildeki son noktadan adım eklenmez.
  - path_rel = yol / boyların ortancası (çift sayıda ortadaki iki değerin ortalaması); boy yoksa null.
- Olaylar: `enter` (track_id'li) → zones'a {zone_id, enter_frame: e.frame, exit_frame: null};
  `exit` / `exit_lost` → o bölgenin açık son ziyaretinin exit_frame = e.frame. Kaydı olmayan iz yok sayılır.
- alive = bu update'in `tracks`'inde var. alive false kayıtlar "bitmiş"; bitmiş sayısı ended_keep'i
  aşarsa last_frame'i en küçük (eşitlikte küçük id) olan silinir.
- `get` JSON: {"track_id", "first_frame", "last_frame", "cls_id", "class" (kClassNames adı),
  "motion", "motion_since_frame", "segments": [{"state", "from_frame", "to_frame"}],
  "stationary_frames", "path_rel" | null, "zones": [{"zone_id", "enter_frame", "exit_frame" | null}],
  "alive"}. Bilinmeyen id → nullopt.
- Özet frame'i = istek frame'i (yoksa oturum n'i).

## Bölge gövdesi ayrıştırma (`std::vector<ZoneDef> parse_zone_defs(const nlohmann::json&)`, zones.cpp)
Gövde nesne ve "zones" dizi olmalı; her bölge nesne: "id" (dize), "name" (dize), "kind" (dize), "gen"
(tam sayı), "points" (her biri 2 sayılık dizi olan dizi), "dwell_frames" (isteğe bağlı tam sayı),
"count" (isteğe bağlı dize; line'da yoksa "both"). Başka anahtarlar (ör. "status") yok sayılır. Tür
hatası → std::invalid_argument (Türkçe, bölge sırası / id ile). Değer sınırları `Zones::set`'te.

## Sessions / Session
- `MotionParts` + `EventLog log; TrackSummaries summary;` (cfg.log, cfg.summary).
- `Session` + `int last_frame = 0; bool has_ego = false; int last_gen = 1; bool last_provisional = false;`
  (runtime her hareketli istekte günceller).
- `std::shared_ptr<Session> Sessions::find(const std::string& id)`: varsa döner (son kullanım
  güncellenir, süre dolanlar önce silinir), yoksa nullptr; **oturum açmaz**.
- `Zones::find(const std::string& id) const` → `const ZoneDef*` (yoksa nullptr).

## Uçlar (server.cpp; hepsi `session` zorunlu; yoksa 400; bilinmeyen oturum 404; oturumda hareket
yoksa 400 "bu oturumda hareket kapalı"; işlem oturum kilidi altında)
- `PUT /v1/session/zones?session=`: gövde > body_max_bytes → 413; bozuk JSON / ayrıştırma / doğrulama
  hatası → 400 (`set` hiçbir şeyi değiştirmez); max_gen = ego.generations()'ın en büyük anahtarı (boşsa
  1). 200 {"zones": defs_json()}. Günlük: yukarıdaki sıra.
- `GET /v1/session/zones?session=` → 200 {"zones": defs_json()}.
- `GET /v1/session/events?session=&since=<seq>&format=json|csv` (since vars. 0, tam sayı ≥ 0; format
  vars. json; başka → 400): json → since() gövdesi; csv → `text/csv; charset=utf-8`, csv_since().
- `POST /v1/session/alarms/ack?session=&alarm=<id>`: alarm yoksa 404 {"error"}; 200 {"alarm": id,
  "acked": true}.
- `GET /v1/session/track?session=&id=<tam sayı>`: özet yoksa 404; 200 özet JSON'u.
- `POST /v1/detect?...&focus=<id>`: yanıtta "focus_track" (özet ya da null; focus yoksa null).

## Klip regresyonu (`ClipZones.Regression`, engine/tests/clip_zones_test.cpp)
Release klibi (fixtures/f4, `scenarios.json`), yalnız `clean` ve `long_gap`; senaryo başına hareketli
yeni oturum; ilk kareden önce `parse_zone_defs(fixtures/motion/zones_clip.json)` → `zones.set(.., 1)`;
her adım `frame` ile, `reset_before` clip.cpp'deki gibi. Çıktı: {"clean": {"events": log.since(0)
["events"], "final_counts": son yanıtın zone_counts'u, "frames": [{"frame", "ego": {"ok", "map_gen",
"provisional", "frame_to_map"}, "tracks": [{"track_id", "box", "motion", "map_xy"}], "alarms"}]},
"long_gap": {...}}. Beklenen `fixtures/expected/zones_clip.json` yalnız "events" ve "final_counts"
anahtarlarını içerir; ikisi birebir eşit olmalı. Beklenen dosya yoksa: BAYKUS_EVIDENCE_DIR verilmişse
tam çıktı oraya `zones_clip_candidate.json` adıyla yazılır, test başarısız ("beklenen dosya yok; aday
yazıldı").
