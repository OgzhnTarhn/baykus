#include "baykus/runtime.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

#include "baykus/classes.hpp"
#include "baykus/files.hpp"

namespace baykus {

namespace {

using Clock = std::chrono::steady_clock;
using json = nlohmann::json;

constexpr float kTrackFloor = 0.1f;  // takip katmanı (tasarım §4.1)

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

std::vector<Box> ego_mask_boxes(const std::vector<Det>& layer, double min_score) {
  std::vector<Box> boxes;
  boxes.reserve(layer.size());
  for (const auto& d : layer) {
    if (d.score_raw >= min_score) boxes.push_back(Box{d.x1, d.y1, d.x2, d.y2});
  }
  return boxes;
}

Runtime::Runtime(const Models& models, const TrackerConfig& tracker_cfg, Sessions& sessions)
    : pipeline_(models), tracker_cfg_(tracker_cfg), sessions_(sessions) {}

Response Runtime::process(const Request& req) {
  // Doğrulama oturum açılmadan önce: hatalı istek oturum tablosunu değiştirmez.
  if (!(req.conf >= 0.0 && req.conf <= 1.0)) {
    throw std::invalid_argument("conf [0, 1] aralığında olmalı");
  }
  if (req.gate != "off" && req.gate != "advise" && req.gate != "auto") {
    throw std::invalid_argument("bilinmeyen gate modu: " + req.gate +
                                "; seçenekler: off, advise, auto");
  }
  if (req.track && req.session.empty()) {
    throw std::invalid_argument("takip için session gerekli");
  }
  if (req.tiles != 1 && req.tiles != 4) {
    throw std::invalid_argument("tiles 1 ya da 4 olmalı");
  }

  // Çözme oturumdan önce (S1): bozuk görüntü 400 olur ve oturum hiç açılmaz.
  const auto t_decode = Clock::now();
  Pair pair = decode_pair(req.rgb, req.ir);
  const double decode_ms = ms_since(t_decode);

  Response r;
  std::shared_ptr<Session> s;
  std::unique_lock<std::mutex> session_lock;
  if (!req.session.empty()) {
    s = sessions_.get(req.session, &r.session_new);
    session_lock = std::unique_lock<std::mutex>(s->m);
  }

  FrameResult fr;
  {
    std::lock_guard<std::mutex> infer_lock(infer_mutex_);
    FrozenDetector* frozen = (s && s->motion) ? &s->motion->frozen : nullptr;
    fr = pipeline_.run(std::move(pair), decode_ms, req.gate, req.floor, s ? &s->window : nullptr,
                       frozen, req.tiles);
  }

  // Takip katmanı: ham ≥ 0,1 (conf'tan bağımsız).
  std::vector<Det> layer;
  for (const auto& d : fr.dets_all) {
    if (d.score_raw >= kTrackFloor) layer.push_back(d);
  }

  // Ego (Python `SessionRef.step` sırası); oturum kilidi tutuluyor, çıkarım kilidi dışında.
  std::optional<EgoResult> ego;
  double ego_ms = 0.0;
  if (s && s->motion) {
    ++s->n;
    bool ir_bad = false;
    if (fr.health.contains("ir") && fr.health["ir"].contains("flags")) {
      for (const auto& f : fr.health["ir"]["flags"]) {
        if (f == "lost" || f == "noisy") {
          ir_bad = true;
          break;
        }
      }
    }
    bool ir_frozen = false;
    if (fr.frozen.is_object() && fr.frozen.contains("ir")) {
      const auto& fi = fr.frozen["ir"];
      ir_bad = ir_bad || fi.value("alarm", false);
      ir_frozen = fi.value("frozen", false);
    }
    const std::string source = ir_bad ? "none" : "ir";
    const cv::Mat* gray =
        (source == "none" || ir_frozen || fr.ir_half.empty()) ? nullptr : &fr.ir_half;
    const std::vector<Box> boxes =
        ego_mask_boxes(layer, sessions_.motion_cfg()->ego.mask_min_score);
    const auto t_ego = Clock::now();
    ego = s->motion->ego.update(gray, source, boxes, req.frame);
    ego_ms = ms_since(t_ego);
  }

  // Takip: takip katmanının bütün tespitleri (conf'tan bağımsız), ham skorla.
  std::map<int, int> ids;  // tespit indeksi → API kimliği
  double track_ms = 0.0;
  if (req.track) {
    const auto t = Clock::now();
    for (const auto& [index, id] : s->tracker.update(layer)) {
      if (id < 0) continue;
      const int api = id + s->id_base;
      ids[index] = api;
      if (api > s->max_api_id) s->max_api_id = api;
    }
    track_ms = ms_since(t);
  }

  // Hareket: kimliği olan takip katmanı tespitleri.
  double motion_ms = 0.0;
  if (s && s->motion && ego) {
    const auto t = Clock::now();
    std::vector<TrackObs> obs;
    for (const auto& d : layer) {
      const auto it = ids.find(d.index);
      if (it != ids.end()) obs.push_back(TrackObs{it->second, Box{d.x1, d.y1, d.x2, d.y2}});
    }
    r.motion = s->motion->motion.update(*ego, obs);
    motion_ms = ms_since(t);
  }

  for (const auto& d : layer) {
    const auto it = ids.find(d.index);
    if (it == ids.end()) continue;
    r.tracks.push_back(
        TrackOut{it->second, d.cls_id, Box{d.x1, d.y1, d.x2, d.y2}, d.score_raw, d.score});
  }

  // Bölgeler + günlük + özet: kimlikli takip katmanı ve hareket durumuyla (Faz 4).
  double zones_ms = 0.0;
  if (s && s->motion && ego) {
    const auto t = Clock::now();
    const int frame = req.frame.value_or(s->n);
    EventLog& log = s->motion->log;
    auto map_event = [&](const char* kind, int gen) {
      ZoneEvent e;
      e.n = s->n;
      e.frame = frame;
      e.kind = kind;
      e.gen = gen;
      r.log_events.push_back(log.append(std::move(e), "", ""));
    };
    const EgoResult& e = *ego;
    if (e.merge) {
      // Geçici nesil açıldığı karede birleştiyse önce açılışı yazılır.
      if (!(s->last_provisional && s->last_gen == e.merge->from_gen)) {
        map_event("map_provisional", e.merge->from_gen);
      }
      map_event("map_relocalized", e.merge->to_gen);
      map_event("map_merged", e.merge->to_gen);
    } else if (s->last_provisional && e.map_gen == s->last_gen && !e.provisional) {
      map_event("map_new_generation", e.map_gen);
    } else if (s->last_provisional && e.map_gen != s->last_gen) {
      map_event("map_new_generation", s->last_gen);
      if (e.provisional) map_event("map_provisional", e.map_gen);
    } else if (e.provisional && (!s->has_ego || e.map_gen != s->last_gen)) {
      map_event("map_provisional", e.map_gen);
    }

    r.events = s->motion->zones.update(s->n, frame, *ego, r.tracks, r.motion);
    for (const auto& ev : r.events) {
      const ZoneDef* def = s->motion->zones.find(ev.zone_id);
      r.log_events.push_back(log.append(ev, def ? def->name : "", def ? def->kind : ""));
    }

    s->has_ego = true;
    s->last_gen = e.map_gen;
    s->last_provisional = e.provisional;
    s->last_frame = frame;

    r.alarms = s->motion->zones.alarms();
    r.zone_counts = s->motion->zones.counts();
    s->motion->summary.update(s->n, frame, r.tracks, r.motion, r.events, *ego);
    if (req.focus) r.focus_track = s->motion->summary.get(*req.focus);
    zones_ms = ms_since(t);
  }

  r.health = std::move(fr.health);
  r.ego = ego;
  r.timing_ms = std::move(fr.timing_ms);
  r.timing_ms["track"] = track_ms;
  r.timing_ms["ego"] = ego_ms;
  r.timing_ms["motion"] = motion_ms;
  r.timing_ms["zones"] = zones_ms;
  double total = 0.0;
  for (const char* k :
       {"decode", "health", "pre", "infer", "post", "track", "ego", "motion", "zones"}) {
    total += r.timing_ms.at(k);
  }
  r.timing_ms["total"] = total;
  r.dets = filter_display(fr.dets_all, req.conf);
  for (const auto& d : r.dets) {
    const auto it = ids.find(d.index);
    r.track_ids.push_back(it == ids.end() ? -1 : it->second);
  }
  return r;
}

json response_json(const Response& r) {
  auto motion_fields = [&r](int id, json& motion, json& speed_rel, json& map_xy) {
    motion = "unknown";
    speed_rel = nullptr;
    map_xy = nullptr;
    if (id < 0) return;
    const auto it = r.motion.find(id);
    if (it == r.motion.end()) return;
    motion = it->second.motion;
    if (it->second.speed_rel) speed_rel = *it->second.speed_rel;
    if (it->second.map_xy) map_xy = {it->second.map_xy->x, it->second.map_xy->y};
  };

  json dets = json::array();
  for (std::size_t i = 0; i < r.dets.size(); ++i) {
    const auto& d = r.dets[i];
    const int id = i < r.track_ids.size() ? r.track_ids[i] : -1;
    const bool known = d.cls_id >= 0 && d.cls_id < static_cast<int>(kClassNames.size());
    json motion, speed_rel, map_xy;
    motion_fields(id, motion, speed_rel, map_xy);
    dets.push_back({{"cls", known ? std::string(kClassNames[d.cls_id]) : std::string("unknown")},
                    {"cls_id", d.cls_id},
                    {"score", d.score},
                    {"score_raw", static_cast<double>(d.score_raw)},
                    {"box", {{"x1", d.x1}, {"y1", d.y1}, {"x2", d.x2}, {"y2", d.y2}}},
                    {"track_id", id < 0 ? json(nullptr) : json(id)},
                    {"motion", motion},
                    {"speed_rel", speed_rel},
                    {"map_xy", map_xy}});
  }

  json tracks = json::array();
  for (const auto& t : r.tracks) {
    json motion, speed_rel, map_xy;
    motion_fields(t.track_id, motion, speed_rel, map_xy);
    tracks.push_back(
        {{"track_id", t.track_id},
         {"cls_id", t.cls_id},
         {"box", {{"x1", t.box.x1}, {"y1", t.box.y1}, {"x2", t.box.x2}, {"y2", t.box.y2}}},
         {"score_raw", static_cast<double>(t.score_raw)},
         {"score", t.score},
         {"motion", motion},
         {"speed_rel", speed_rel},
         {"map_xy", map_xy}});
  }

  // Oturumsuz ya da hareketsiz: donmuş kare alanları API'de null (tasarım §5.7). Runtime'ın
  // sağlık sözlüğü Faz 3 referansıyla birebir kalır.
  json health = r.health;
  for (const char* ch : {"rgb", "ir"}) {
    if (health.contains(ch) && !health[ch].contains("frozen")) {
      health[ch]["frozen"] = nullptr;
      health[ch]["diff_mad"] = nullptr;
    }
  }

  json events = r.log_events;
  json alarms = json::array();
  for (const auto& a : r.alarms) alarms.push_back(a.to_json());

  return {{"detections", dets},
          {"tracks", tracks},
          {"events", events},
          {"focus_track", r.focus_track ? *r.focus_track : json(nullptr)},
          {"alarms", alarms},
          {"zone_counts", r.zone_counts},
          {"ego", r.ego ? r.ego->to_json() : json(nullptr)},
          {"health", health},
          {"timing_ms", r.timing_ms},
          {"session_new", r.session_new}};
}

json detect_sequence(Runtime& runtime, const std::string& seq_dir, double floor,
                     const std::string& gate, bool allow_test, int tiles) {
  if (tiles != 1 && tiles != 4) throw std::invalid_argument("tiles 1 ya da 4");
  const std::filesystem::path d(seq_dir);
  const auto index = json::parse(read_bytes((d / "index.json").string()));
  if (index.value("split", "") == "test" && !allow_test) {
    throw std::invalid_argument(seq_dir +
                                ": test dizisi; yalnız son ölçümde --i-know-this-is-test ile");
  }
  json frames = json::array();
  for (const auto& f : index.at("frames")) {
    char name[16];
    std::snprintf(name, sizeof(name), "%06d", f.get<int>());
    Request req;
    req.rgb = read_bytes((d / "rgb" / (std::string(name) + ".jpg")).string());
    req.ir = read_bytes((d / "ir" / (std::string(name) + ".png")).string());
    req.gate = gate;
    req.conf = 0.0;  // bütün katman (kalibre skor ≥ 0)
    req.track = false;
    req.floor = static_cast<float>(floor);
    req.tiles = tiles;
    json dets = json::array();
    for (const auto& det : runtime.process(req).dets) {
      dets.push_back({{"index", det.index},
                      {"cls_id", det.cls_id},
                      {"score_raw", static_cast<double>(det.score_raw)},
                      {"score", det.score},
                      {"box", {det.x1, det.y1, det.x2, det.y2}}});
    }
    frames.push_back({{"frame", f}, {"dets", dets}});
  }
  json rec = {{"seq", index.at("name")},
              {"split", index.at("split")},
              {"floor", floor},
              {"gate", gate},
              {"frames", frames}};
  if (tiles != 1) rec["tiles"] = tiles;
  return rec;
}

}  // namespace baykus
