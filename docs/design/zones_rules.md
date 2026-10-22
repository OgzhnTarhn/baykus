# Faz 4 bölgeler, kurallar, alarmlar: kesin kurallar (Görev 14; tasarım docs/design/faz4.md §5.5)

Python referansı yok; davranış yalnız bu kurallar + C++ testleri. Parametreler `ZonesCfg`
(configs/engine/motion.json "zones"): max_zones 16, max_points 32, name_max 40, dwell_max 36000,
enter_obs 3, exit_obs 3, gap_max 5, lost_frames 30, stopped_obs 30, dwell_clear_moving 3,
line_side_obs 3. Kodda sabit yok; hepsi cfg'den.

## Arayüz (engine/include/baykus/zones.hpp, namespace baykus)
- `TrackOut` runtime.hpp'den motion.hpp'ye taşınır (runtime.hpp motion.hpp'yi zaten içerir; döngüsel
  içerme olmasın: sessions.hpp → zones.hpp → motion.hpp).
- `struct ZoneDef { std::string id, name, kind; int gen; std::vector<cv::Point2d> points;
  std::optional<int> dwell_frames; std::string count; };` kind: area | restricted | dwell | line.
- `struct ZoneEvent { int n; int frame; std::string kind; std::string zone_id;
  std::optional<int> track_id; std::optional<std::string> dir; int gen;
  std::optional<std::string> alarm_id; nlohmann::json to_json() const; };`
  to_json: {"n","frame","kind","zone_id","track_id"|null,"dir"|null,"gen","alarm_id"|null}.
- `struct Alarm { std::string id, kind, zone_id; std::vector<int> track_ids; int since_n;
  int since_frame; bool acked; nlohmann::json to_json() const; };`
  to_json: {"id","kind","zone_id","track_ids","since_n","since_frame","acked"}.
- `class Zones`:
  - `explicit Zones(const ZonesCfg&)`
  - `void set(std::vector<ZoneDef> defs, int max_gen, std::vector<ZoneEvent>* cleared = nullptr)`
  - `std::vector<ZoneEvent> update(int n, int frame, const EgoResult& ego,
     const std::vector<TrackOut>& tracks, const std::map<int, TrackMotion>& motion)`
  - `bool ack(const std::string& alarm_id, int n, int frame, std::vector<ZoneEvent>* out)`
  - `std::vector<Alarm> alarms() const` · `nlohmann::json counts() const` ·
    `nlohmann::json defs_json() const` · `void reset_tracks()` · `void reset()`

## Olay türleri (İngilizce kod)
enter, exit, exit_lost, stopped, line_cross, alarm_start, alarm_clear, alarm_ack.
Alarm türleri: intrusion (restricted, bölge düzeyi), dwell_exceeded (dwell, iz düzeyi).
Alarm kimliği: `<zone_id>:<kind>:<since_n>` (intrusion) / `<zone_id>:<kind>:<track_id>:<since_n>`
(dwell_exceeded). Ör. "z2:intrusion:57", "z3:dwell_exceeded:4:120".
Olayın `gen` alanı bölgenin nesli.

## Doğrulama (`set`; ilk hatada std::invalid_argument, Türkçe ileti bölge id'si ya da sırasıyla;
hata varsa HİÇBİR durum değişmez)
- bölge sayısı ≤ max_zones; id `[A-Za-z0-9_-]{1,16}` ve listede benzersiz;
- name 1..name_max **karakter** (UTF-8 kod noktası; "ğ" bir karakter);
- kind dört türden biri; 1 ≤ gen ≤ max_gen; bütün koordinatlar sonlu (NaN / inf hata);
- çokgen türleri: 3..max_points nokta, `is_simple` doğru, `polygon_area` > 1.0;
- line: tam 2 nokta, iki nokta farklı;
- dwell: dwell_frames var ve 1..dwell_max; diğer türlerde dwell_frames yok olmalı;
- line: count ∈ {both, a_to_b, b_to_a}; diğer türlerde count boş olmalı.

## set: değişmeyen bölgenin durumu korunur (Review Focus 4)
Eski listede aynı id ile **kind, gen, points (birebir), dwell_frames, count** aynı olan bölge
"değişmemiş": iz durumları, sayaçlar, açık alarmları ve aktif/askı bilgisi taşınır; yalnız name
güncellenir. Değişen ya da çıkarılan bölgenin açık alarmları için `cleared`'a (null değilse)
alarm_clear olayı (eski sırayla; n / frame = son update'in n / frame'i, hiç update yoksa 0 / 0).
Yeni / değişen bölge boş durumla başlar; aktifliği son görülen ego'dan (son update'te
map_gen == gen ve provisional false; hiç update yoksa askıda).

## Aktiflik ve askı (her update başında, bölge başına)
aktif ⇔ `ego.map_gen == def.gen && !ego.provisional` (ego.ok false olsa bile aktif; o karede bütün
gözlemler `yok` olur). Askıdaysa: bu update'te önce açık alarmları için alarm_clear (n / frame güncel),
sonra bütün iz durumları silinir, bütün sayaçlar (entered_total, exceeded_total, a_to_b, b_to_a)
0; gözlem yapılmaz. Birleşmeyle yeniden aktif olunca boş durumdan devam (olay üretilmez).

## Gözlem (aktif bölge, iz başına, her update)
İz kümesi = bu karedeki `tracks` ∪ bölgede durumu olan izler; **artan track_id sırasıyla** işlenir.
- İz bu karede `tracks`'te yok: `absent`++ (absent 1 olduğunda o karenin n / frame'i saklanır);
  gözlem `yok`. absent ≥ lost_frames ise: çokgen türünde durum `inside` ise exit_lost olayı
  (n / frame = ilk yok karesi); iz–bölge durumu silinir (her türde, olaysız da).
- İz `tracks`'te: absent = 0. `motion`'da kaydı yoksa ya da `map_xy` yoksa gözlem `yok`.
  Değilse nokta p = map_xy; hareket = motion[id].motion (moving | stationary | unknown).
- `yok`: gap++; gap tam gap_max + 1 olduğunda bütün "art arda" sayaçları sıfırlanır (in/out koşusu,
  durağan koşusu, hareketli koşusu, hat taraf koşusu); iz–bölge `state` ve hatta kararlı taraf korunur.
- Gözlem (in/out ya da taraf): gap = 0.

## Çokgen türleri (area, restricted, dwell)
taraf = `point_in_polygon(p)` (kenar üstü içerde) ? in : out. Durum `state` ∈ none | inside | outside
(başta none).
1. Koşu: taraf önceki koşuyla aynıysa run_len++, değilse yeni koşu (run_len 1, ilk gözlemin n / frame'i).
2. run_len ≥ (in → enter_obs, out → exit_obs) ve state ≠ taraf ise geçiş:
   - in: state = inside, enter olayı (n / frame = koşunun ilk gözlemi); kalış başlar: stopped_fired
     false, durağan ve hareketli koşuları 0; restricted ise entered kümesine iz eklenir.
   - out: önceki state inside idiyse exit olayı (koşunun ilk gözlemi); state = outside; durağan ve
     hareketli koşuları 0. none → outside olaysız.
3. Adım 2'den sonra state == inside ise (geçişi onaylayan gözlem dahil) hareket:
   stationary → durağan koşusu++ (1 olunca n / frame saklanır), hareketli koşusu 0;
   moving → durağan 0, hareketli++; unknown → ikisi 0.
   Durağan koşusu ≥ stopped_obs ve !stopped_fired → stopped olayı (n / frame = durağan koşusunun ilk
   gözlemi), stopped_fired = true (kalış başına bir kez; yeni kalış = yeni enter).
İzin bu karedeki hareketi sayaçlar için saklanır (karede yoksa unknown).

## Hat (line), A = points[0], B = points[1]
s = sign(cross(B − A, p − A)) (+1 sol, −1 sağ); s == 0 ise önceki gözlemin tarafı; o da yoksa gözlem
yalnız gap'i 0 yapar, başka etkisi yok. prev_side = s.
- Koşu: s önceki koşuyla aynıysa run_len++, değilse yeni koşu (ilk gözlemin n / frame / konumu).
- Kararlı taraf yoksa: run_len ≥ line_side_obs → kararlı taraf = s, olaysız.
- s == kararlı taraf: last_stable_pos = p.
- s ≠ kararlı taraf ve run_len ≥ line_side_obs: `segments_intersect(last_stable_pos, run_first_pos,
  A, B)` (uçlar dahil) doğruysa geçiş: dir = eski kararlı taraf +1 → "a_to_b", −1 → "b_to_a"; dir
  count'ta varsa (both ya da aynı ad) o yönün sayacı++ ve line_cross olayı (dir, n / frame = koşunun
  ilk gözlemi). Kesişme olsun olmasın kararlı taraf = s, last_stable_pos = p.
- Kararlı taraf belirlendiği gözlemde de last_stable_pos = p.

## Alarmlar (aktif bölgenin iz döngüsünden sonra; önce temizlemeler, sonra açılışlar)
- intrusion (restricted): inside izleri (artan id). Açık alarm varsa track_ids güncellenir; küme
  boşsa alarm_clear. Alarm yoksa ve küme boş değilse açılır (since = güncel n / frame) + alarm_start.
- dwell_exceeded (dwell): açık alarmın izinin durumu yoksa / inside değilse ya da hareketli koşusu ≥
  dwell_clear_moving ise alarm_clear. Sonra inside olup durağan koşusu ≥ dwell_frames olan ve açık
  alarmı olmayan her iz için açılır + alarm_start; exceeded kümesine iz eklenir.
- alarm_start / alarm_clear / alarm_ack olaylarında alarm_id dolu; dwell'de track_id iz, intrusion'da null.
- `alarms()` açık alarmlar, açılış sırasıyla. `ack`: açık alarm yoksa false; acked false ise true
  yapılır ve `out`'a alarm_ack (verilen n / frame); zaten acked ise olaysız true.
- Kare içi olay sırası: bölgeler tanım sırasıyla; bölge içinde iz olayları (artan id), sonra
  alarm_clear'lar, sonra alarm_start'lar. Askı temizlemeleri o bölgenin yerinde.

## counts() (anahtar bölge id; askıdaki bölgede sayılar 0)
{"status": "active"|"suspended", "inside", "moving", "stationary", "unknown", "entered_total",
"exceeded_total", "a_to_b", "b_to_a"}. Çokgen türlerinde inside = state inside iz sayısı; moving /
stationary / unknown bunların son karedeki hareketine göre; hat türünde bu dördü null. entered_total
yalnız restricted'da (entered kümesinin boyu), değilse null; exceeded_total yalnız dwell'de, değilse
null; a_to_b / b_to_a yalnız hatta ve yön count'ta varsa sayı, değilse null.

## defs_json()
Dizi: {"id","name","kind","gen","points":[[x,y],...],"status"} + dwell'de "dwell_frames", hatta "count".

## reset_tracks() / reset()
- reset_tracks (scope=tracks): bütün iz–bölge durumları silinir; tanımlar, aktiflik, sayaçlar ve
  açık alarmlar kalır (sonraki update alarmları kurala göre temizler ve olayını üretir).
- reset (scope=all): her şey silinir (tanımlar, alarmlar, sayaçlar, son n / frame / ego).
