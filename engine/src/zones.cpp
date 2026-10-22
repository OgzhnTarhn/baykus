#include "baykus/zones.hpp"

#include <cmath>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "baykus/polygon.hpp"

namespace baykus {

namespace {

// UTF-8 kod noktası sayısı: devam baytları (10xxxxxx) sayılmaz.
std::size_t utf8_length(const std::string& s) {
  std::size_t n = 0;
  for (const char c : s) {
    if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) ++n;
  }
  return n;
}

// [A-Za-z0-9_-]{1,16}
bool valid_id(const std::string& id) {
  if (id.empty() || id.size() > 16) return false;
  for (const char c : id) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                    c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

bool is_polygon_kind(const std::string& kind) {
  return kind == "area" || kind == "restricted" || kind == "dwell";
}

// Değişmeyen bölge: kind, gen, points (birebir), dwell_frames, count aynı.
bool same_def(const ZoneDef& a, const ZoneDef& b) {
  if (a.kind != b.kind || a.gen != b.gen || a.dwell_frames != b.dwell_frames ||
      a.count != b.count) {
    return false;
  }
  if (a.points.size() != b.points.size()) return false;
  for (std::size_t i = 0; i < a.points.size(); ++i) {
    if (a.points[i].x != b.points[i].x || a.points[i].y != b.points[i].y) return false;
  }
  return true;
}

}  // namespace

nlohmann::json ZoneEvent::to_json() const {
  nlohmann::json j;
  j["n"] = n;
  j["frame"] = frame;
  j["kind"] = kind;
  j["zone_id"] = zone_id;
  j["track_id"] = track_id ? nlohmann::json(*track_id) : nlohmann::json(nullptr);
  j["dir"] = dir ? nlohmann::json(*dir) : nlohmann::json(nullptr);
  j["gen"] = gen;
  j["alarm_id"] = alarm_id ? nlohmann::json(*alarm_id) : nlohmann::json(nullptr);
  return j;
}

nlohmann::json Alarm::to_json() const {
  nlohmann::json j;
  j["id"] = id;
  j["kind"] = kind;
  j["zone_id"] = zone_id;
  j["track_ids"] = track_ids;
  j["since_n"] = since_n;
  j["since_frame"] = since_frame;
  j["acked"] = acked;
  return j;
}

Zones::Zones(const ZonesCfg& cfg) : cfg_(cfg) {}

void Zones::set(std::vector<ZoneDef> defs, int max_gen, std::vector<ZoneEvent>* cleared) {
  // --- Doğrulama: ilk hatada std::invalid_argument, hiçbir durum değişmez. ---
  if (static_cast<int>(defs.size()) > cfg_.max_zones) {
    throw std::invalid_argument("bölge sayısı en fazla " + std::to_string(cfg_.max_zones) +
                                " olabilir");
  }
  std::set<std::string> ids;
  for (std::size_t i = 0; i < defs.size(); ++i) {
    const ZoneDef& d = defs[i];
    if (!valid_id(d.id)) {
      throw std::invalid_argument("bölge " + std::to_string(i + 1) + ": geçersiz id");
    }
    if (!ids.insert(d.id).second) {
      throw std::invalid_argument("bölge " + d.id + ": id benzersiz değil");
    }
    const std::size_t name_len = utf8_length(d.name);
    if (name_len < 1 || name_len > static_cast<std::size_t>(cfg_.name_max)) {
      throw std::invalid_argument("bölge " + d.id + ": ad uzunluğu 1.." +
                                  std::to_string(cfg_.name_max) + " karakter olmalı");
    }
    const bool polygon = is_polygon_kind(d.kind);
    if (!polygon && d.kind != "line") {
      throw std::invalid_argument("bölge " + d.id + ": geçersiz tür");
    }
    if (d.gen < 1 || d.gen > max_gen) {
      throw std::invalid_argument("bölge " + d.id + ": geçersiz nesil");
    }
    for (const cv::Point2d& p : d.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
        throw std::invalid_argument("bölge " + d.id + ": koordinat sonlu değil");
      }
    }
    if (polygon) {
      if (d.points.size() < 3u || static_cast<int>(d.points.size()) > cfg_.max_points) {
        throw std::invalid_argument("bölge " + d.id + ": nokta sayısı 3.." +
                                    std::to_string(cfg_.max_points) + " olmalı");
      }
      if (!is_simple(d.points)) {
        throw std::invalid_argument("bölge " + d.id + ": kendini kesen çokgen");
      }
      if (!(polygon_area(d.points) > 1.0)) {
        throw std::invalid_argument("bölge " + d.id + ": çokgen alanı çok küçük");
      }
    } else {
      if (d.points.size() != 2u) {
        throw std::invalid_argument("bölge " + d.id + ": hat tam 2 nokta olmalı");
      }
      if (d.points[0].x == d.points[1].x && d.points[0].y == d.points[1].y) {
        throw std::invalid_argument("bölge " + d.id + ": hat noktaları aynı");
      }
    }
    if (d.kind == "dwell") {
      if (!d.dwell_frames || *d.dwell_frames < 1 || *d.dwell_frames > cfg_.dwell_max) {
        throw std::invalid_argument("bölge " + d.id + ": dwell_frames 1.." +
                                    std::to_string(cfg_.dwell_max) + " olmalı");
      }
    } else if (d.dwell_frames) {
      throw std::invalid_argument("bölge " + d.id + ": dwell_frames yalnız dwell türünde olmalı");
    }
    if (d.kind == "line") {
      if (d.count != "both" && d.count != "a_to_b" && d.count != "b_to_a") {
        throw std::invalid_argument("bölge " + d.id + ": geçersiz yön");
      }
    } else if (!d.count.empty()) {
      throw std::invalid_argument("bölge " + d.id + ": count yalnız hat türünde olmalı");
    }
  }

  // --- Yeni durum: değişmeyen bölgenin durumu korunur, diğerleri boş başlar. ---
  std::vector<ZoneState> next;
  next.reserve(defs.size());
  std::set<std::string> unchanged;
  for (const ZoneDef& d : defs) {
    ZoneState zs;
    zs.def = d;
    const ZoneState* old = nullptr;
    for (const ZoneState& o : zones_) {
      if (o.def.id == d.id) {
        old = &o;
        break;
      }
    }
    if (old != nullptr && same_def(old->def, d)) {
      zs = *old;
      zs.def.name = d.name;
      unchanged.insert(d.id);
    } else {
      zs.active = has_ego_ && ego_gen_ == d.gen && !ego_provisional_;
    }
    next.push_back(std::move(zs));
  }

  // --- Değişen / çıkarılan bölgelerin açık alarmları temizlenir (eski sırayla). ---
  std::vector<Alarm> kept;
  kept.reserve(alarms_.size());
  for (const Alarm& a : alarms_) {
    if (unchanged.count(a.zone_id) != 0) {
      kept.push_back(a);
      continue;
    }
    if (cleared != nullptr) {
      ZoneEvent ev;
      ev.n = last_n_;
      ev.frame = last_frame_;
      ev.kind = "alarm_clear";
      ev.zone_id = a.zone_id;
      ev.track_id = std::nullopt;
      if (a.kind == "dwell_exceeded" && !a.track_ids.empty()) {
        ev.track_id = a.track_ids.front();
      }
      ev.dir = std::nullopt;
      ev.gen = 0;
      for (const ZoneState& o : zones_) {
        if (o.def.id == a.zone_id) {
          ev.gen = o.def.gen;
          break;
        }
      }
      ev.alarm_id = a.id;
      cleared->push_back(ev);
    }
  }

  zones_ = std::move(next);
  alarms_ = std::move(kept);
}

bool Zones::ack(const std::string& alarm_id, int n, int frame, std::vector<ZoneEvent>* out) {
  for (Alarm& a : alarms_) {
    if (a.id != alarm_id) continue;
    if (a.acked) return true;
    a.acked = true;
    if (out != nullptr) {
      ZoneEvent ev;
      ev.n = n;
      ev.frame = frame;
      ev.kind = "alarm_ack";
      ev.zone_id = a.zone_id;
      ev.track_id = std::nullopt;
      if (a.kind == "dwell_exceeded" && !a.track_ids.empty()) {
        ev.track_id = a.track_ids.front();
      }
      ev.dir = std::nullopt;
      ev.gen = 0;
      for (const ZoneState& z : zones_) {
        if (z.def.id == a.zone_id) {
          ev.gen = z.def.gen;
          break;
        }
      }
      ev.alarm_id = a.id;
      out->push_back(ev);
    }
    return true;
  }
  return false;
}

std::vector<Alarm> Zones::alarms() const { return alarms_; }

nlohmann::json Zones::counts() const {
  nlohmann::json j = nlohmann::json::object();
  for (const ZoneState& z : zones_) {
    nlohmann::json c;
    c["status"] = z.active ? "active" : "suspended";
    if (is_polygon_kind(z.def.kind)) {
      int inside = 0;
      int moving = 0;
      int stationary = 0;
      int unknown = 0;
      if (z.active) {
        for (const auto& kv : z.tracks) {
          const TrackZone& tz = kv.second;
          if (tz.state != "inside") continue;
          ++inside;
          if (tz.last_motion == "moving") {
            ++moving;
          } else if (tz.last_motion == "stationary") {
            ++stationary;
          } else {
            ++unknown;
          }
        }
      }
      c["inside"] = inside;
      c["moving"] = moving;
      c["stationary"] = stationary;
      c["unknown"] = unknown;
    } else {
      c["inside"] = nullptr;
      c["moving"] = nullptr;
      c["stationary"] = nullptr;
      c["unknown"] = nullptr;
    }
    if (z.def.kind == "restricted") {
      c["entered_total"] = z.active ? static_cast<int>(z.entered.size()) : 0;
    } else {
      c["entered_total"] = nullptr;
    }
    if (z.def.kind == "dwell") {
      c["exceeded_total"] = z.active ? static_cast<int>(z.exceeded.size()) : 0;
    } else {
      c["exceeded_total"] = nullptr;
    }
    if (z.def.kind == "line") {
      const bool ab = z.def.count == "both" || z.def.count == "a_to_b";
      const bool ba = z.def.count == "both" || z.def.count == "b_to_a";
      c["a_to_b"] = ab ? nlohmann::json(z.active ? z.a_to_b : 0) : nlohmann::json(nullptr);
      c["b_to_a"] = ba ? nlohmann::json(z.active ? z.b_to_a : 0) : nlohmann::json(nullptr);
    } else {
      c["a_to_b"] = nullptr;
      c["b_to_a"] = nullptr;
    }
    j[z.def.id] = c;
  }
  return j;
}

nlohmann::json Zones::defs_json() const {
  nlohmann::json arr = nlohmann::json::array();
  for (const ZoneState& z : zones_) {
    nlohmann::json d;
    d["id"] = z.def.id;
    d["name"] = z.def.name;
    d["kind"] = z.def.kind;
    d["gen"] = z.def.gen;
    nlohmann::json pts = nlohmann::json::array();
    for (const cv::Point2d& p : z.def.points) {
      pts.push_back({p.x, p.y});
    }
    d["points"] = pts;
    d["status"] = z.active ? "active" : "suspended";
    if (z.def.kind == "dwell") {
      d["dwell_frames"] = z.def.dwell_frames ? *z.def.dwell_frames : 0;
    }
    if (z.def.kind == "line") {
      d["count"] = z.def.count;
    }
    arr.push_back(d);
  }
  return arr;
}

void Zones::reset_tracks() {
  for (ZoneState& z : zones_) {
    z.tracks.clear();
  }
}

void Zones::reset() {
  zones_.clear();
  alarms_.clear();
  last_n_ = 0;
  last_frame_ = 0;
  has_ego_ = false;
  ego_gen_ = 0;
  ego_provisional_ = false;
}

void Zones::reset_runs_(TrackZone& tz) {
  tz.run_side = 0;
  tz.run_len = 0;
  tz.run_n = 0;
  tz.run_frame = 0;
  tz.run_pos = cv::Point2d{0.0, 0.0};
  tz.stopped_run = 0;
  tz.moving_run = 0;
}

void Zones::suspend_(ZoneState& z, int n, int frame, std::vector<ZoneEvent>& out) {
  // Önce açık alarmlar için alarm_clear (n / frame güncel), açılış sırasıyla.
  for (std::size_t i = 0; i < alarms_.size();) {
    const Alarm& a = alarms_[i];
    if (a.zone_id != z.def.id) {
      ++i;
      continue;
    }
    ZoneEvent ev;
    ev.n = n;
    ev.frame = frame;
    ev.kind = "alarm_clear";
    ev.zone_id = a.zone_id;
    ev.track_id = std::nullopt;
    if (a.kind == "dwell_exceeded" && !a.track_ids.empty()) {
      ev.track_id = a.track_ids.front();
    }
    ev.dir = std::nullopt;
    ev.gen = z.def.gen;
    ev.alarm_id = a.id;
    out.push_back(ev);
    alarms_.erase(alarms_.begin() + static_cast<std::ptrdiff_t>(i));
  }
  // Sonra bütün iz durumları ve sayaçlar silinir.
  z.tracks.clear();
  z.entered.clear();
  z.exceeded.clear();
  z.a_to_b = 0;
  z.b_to_a = 0;
}

std::vector<ZoneEvent> Zones::update(int n, int frame, const EgoResult& ego,
                                     const std::vector<TrackOut>& tracks,
                                     const std::map<int, TrackMotion>& motion) {
  std::vector<ZoneEvent> out;
  last_n_ = n;
  last_frame_ = frame;
  has_ego_ = true;
  ego_gen_ = ego.map_gen;
  ego_provisional_ = ego.provisional;

  // Bu karedeki izler (id → iz).
  std::map<int, const TrackOut*> by_id;
  for (const TrackOut& t : tracks) {
    by_id[t.track_id] = &t;
  }

  for (ZoneState& z : zones_) {
    // Aktiflik: ego.map_gen == def.gen && !ego.provisional (ego.ok false olsa bile).
    if (!(ego.map_gen == z.def.gen && !ego.provisional)) {
      suspend_(z, n, frame, out);
      z.active = false;
      continue;
    }
    z.active = true;

    // İz kümesi = bu karedeki tracks ∪ bölgede durumu olan izler; artan track_id sırasıyla.
    std::set<int> ids;
    for (const auto& kv : by_id) ids.insert(kv.first);
    for (const auto& kv : z.tracks) ids.insert(kv.first);

    for (const int id : ids) {
      TrackZone& tz = z.tracks[id];
      const auto it = by_id.find(id);
      if (it == by_id.end()) {
        // İz bu karede tracks'te yok.
        ++tz.absent;
        if (tz.absent == 1) {
          tz.absent_n = n;
          tz.absent_frame = frame;
        }
        if (tz.absent >= cfg_.lost_frames) {
          if (is_polygon_kind(z.def.kind) && tz.state == "inside") {
            ZoneEvent ev;
            ev.n = tz.absent_n;
            ev.frame = tz.absent_frame;
            ev.kind = "exit_lost";
            ev.zone_id = z.def.id;
            ev.track_id = id;
            ev.dir = std::nullopt;
            ev.gen = z.def.gen;
            ev.alarm_id = std::nullopt;
            out.push_back(ev);
          }
          z.tracks.erase(id);
          continue;
        }
        ++tz.gap;  // listede olmayan iz de `yok` gözlemi
        if (tz.gap == cfg_.gap_max + 1) reset_runs_(tz);
        continue;
      }

      tz.absent = 0;
      const auto mit = motion.find(id);
      if (mit == motion.end() || !mit->second.map_xy) {
        // Gözlem yok: gap++ ve gap_max + 1'de bütün "art arda" sayaçları sıfırlanır.
        tz.last_motion = mit != motion.end() ? mit->second.motion : "unknown";
        ++tz.gap;
        if (tz.gap == cfg_.gap_max + 1) reset_runs_(tz);
        continue;
      }

      tz.gap = 0;
      tz.last_motion = mit->second.motion;
      if (is_polygon_kind(z.def.kind)) {
        observe_polygon_(z, id, tz, *mit->second.map_xy, tz.last_motion, n, frame, out);
      } else {
        observe_line_(z, id, tz, *mit->second.map_xy, n, frame, out);
      }
    }

    reconcile_alarms_(z, n, frame, out);
  }

  return out;
}

void Zones::observe_polygon_(ZoneState& z, int track_id, TrackZone& tz, const cv::Point2d& p,
                             const std::string& motion, int n, int frame,
                             std::vector<ZoneEvent>& out) {
  // taraf: kenar üstü içerde → +1 in, değilse −1 out.
  const int side = point_in_polygon(p, z.def.points) ? 1 : -1;

  // 1. Koşu: aynı tarafta run_len++, değilse yeni koşu (ilk gözlemin n / frame'i).
  if (tz.run_len > 0 && tz.run_side == side) {
    ++tz.run_len;
  } else {
    tz.run_side = side;
    tz.run_len = 1;
    tz.run_n = n;
    tz.run_frame = frame;
  }

  // 2. Geçiş: run_len ≥ (in → enter_obs, out → exit_obs) ve state ≠ taraf.
  const int need = side == 1 ? cfg_.enter_obs : cfg_.exit_obs;
  const std::string target = side == 1 ? "inside" : "outside";
  if (tz.run_len >= need && tz.state != target) {
    if (side == 1) {
      tz.state = "inside";
      ZoneEvent ev;
      ev.n = tz.run_n;
      ev.frame = tz.run_frame;
      ev.kind = "enter";
      ev.zone_id = z.def.id;
      ev.track_id = track_id;
      ev.dir = std::nullopt;
      ev.gen = z.def.gen;
      ev.alarm_id = std::nullopt;
      out.push_back(ev);
      // Kalış başlar.
      tz.stopped_fired = false;
      tz.stopped_run = 0;
      tz.moving_run = 0;
      if (z.def.kind == "restricted") z.entered.insert(track_id);
    } else {
      if (tz.state == "inside") {
        ZoneEvent ev;
        ev.n = tz.run_n;
        ev.frame = tz.run_frame;
        ev.kind = "exit";
        ev.zone_id = z.def.id;
        ev.track_id = track_id;
        ev.dir = std::nullopt;
        ev.gen = z.def.gen;
        ev.alarm_id = std::nullopt;
        out.push_back(ev);
      }
      tz.state = "outside";
      tz.stopped_run = 0;
      tz.moving_run = 0;
    }
  }

  // 3. state == inside ise hareket sayaçları (geçişi onaylayan gözlem dahil).
  if (tz.state == "inside") {
    if (motion == "stationary") {
      ++tz.stopped_run;
      if (tz.stopped_run == 1) {
        tz.stopped_n = n;
        tz.stopped_frame = frame;
      }
      tz.moving_run = 0;
    } else if (motion == "moving") {
      tz.stopped_run = 0;
      ++tz.moving_run;
    } else {
      tz.stopped_run = 0;
      tz.moving_run = 0;
    }
    if (tz.stopped_run >= cfg_.stopped_obs && !tz.stopped_fired) {
      ZoneEvent ev;
      ev.n = tz.stopped_n;
      ev.frame = tz.stopped_frame;
      ev.kind = "stopped";
      ev.zone_id = z.def.id;
      ev.track_id = track_id;
      ev.dir = std::nullopt;
      ev.gen = z.def.gen;
      ev.alarm_id = std::nullopt;
      out.push_back(ev);
      tz.stopped_fired = true;
    }
  }
}

void Zones::observe_line_(ZoneState& z, int track_id, TrackZone& tz, const cv::Point2d& p, int n,
                          int frame, std::vector<ZoneEvent>& out) {
  const cv::Point2d& a = z.def.points[0];
  const cv::Point2d& b = z.def.points[1];

  // s = sign(cross(B − A, p − A)); s == 0 ise önceki gözlemin tarafı.
  const double c = cross(a, b, p);
  int s = c > 0.0 ? 1 : (c < 0.0 ? -1 : 0);
  if (s == 0) {
    if (tz.prev_side == 0) return;  // yalnız gap = 0 (çağıran yaptı), başka etkisi yok.
    s = tz.prev_side;
  }
  tz.prev_side = s;

  // Koşu: aynı tarafta run_len++, değilse yeni koşu (ilk gözlemin n / frame / konumu).
  if (tz.run_len > 0 && tz.run_side == s) {
    ++tz.run_len;
  } else {
    tz.run_side = s;
    tz.run_len = 1;
    tz.run_n = n;
    tz.run_frame = frame;
    tz.run_pos = p;
  }

  // Kararlı taraf yoksa: run_len ≥ line_side_obs → kararlı taraf = s, olaysız.
  if (tz.stable_side == 0) {
    if (tz.run_len >= cfg_.line_side_obs) {
      tz.stable_side = s;
      tz.last_stable_pos = p;
    }
    return;
  }

  if (s == tz.stable_side) {
    tz.last_stable_pos = p;
    return;
  }

  if (tz.run_len >= cfg_.line_side_obs) {
    if (segments_intersect(tz.last_stable_pos, tz.run_pos, a, b)) {
      const std::string dir = tz.stable_side == 1 ? "a_to_b" : "b_to_a";
      if (z.def.count == "both" || z.def.count == dir) {
        if (dir == "a_to_b") {
          ++z.a_to_b;
        } else {
          ++z.b_to_a;
        }
        ZoneEvent ev;
        ev.n = tz.run_n;
        ev.frame = tz.run_frame;
        ev.kind = "line_cross";
        ev.zone_id = z.def.id;
        ev.track_id = track_id;
        ev.dir = dir;
        ev.gen = z.def.gen;
        ev.alarm_id = std::nullopt;
        out.push_back(ev);
      }
    }
    // Kesişme olsun olmasın kararlı taraf = s, last_stable_pos = p.
    tz.stable_side = s;
    tz.last_stable_pos = p;
  }
}

void Zones::reconcile_alarms_(ZoneState& z, int n, int frame, std::vector<ZoneEvent>& out) {
  // --- Önce temizlemeler. ---
  if (z.def.kind == "restricted") {
    // inside izleri (artan id).
    std::vector<int> inside;
    for (const auto& kv : z.tracks) {
      if (kv.second.state == "inside") inside.push_back(kv.first);
    }
    for (std::size_t i = 0; i < alarms_.size();) {
      Alarm& a = alarms_[i];
      if (a.zone_id != z.def.id || a.kind != "intrusion") {
        ++i;
        continue;
      }
      if (inside.empty()) {
        ZoneEvent ev;
        ev.n = n;
        ev.frame = frame;
        ev.kind = "alarm_clear";
        ev.zone_id = z.def.id;
        ev.track_id = std::nullopt;
        ev.dir = std::nullopt;
        ev.gen = z.def.gen;
        ev.alarm_id = a.id;
        out.push_back(ev);
        alarms_.erase(alarms_.begin() + static_cast<std::ptrdiff_t>(i));
        continue;
      }
      a.track_ids = inside;
      ++i;
    }
  } else if (z.def.kind == "dwell") {
    for (std::size_t i = 0; i < alarms_.size();) {
      Alarm& a = alarms_[i];
      if (a.zone_id != z.def.id || a.kind != "dwell_exceeded") {
        ++i;
        continue;
      }
      bool clear = true;
      if (!a.track_ids.empty()) {
        const auto it = z.tracks.find(a.track_ids.front());
        if (it != z.tracks.end() && it->second.state == "inside" &&
            it->second.moving_run < cfg_.dwell_clear_moving) {
          clear = false;
        }
      }
      if (!clear) {
        ++i;
        continue;
      }
      ZoneEvent ev;
      ev.n = n;
      ev.frame = frame;
      ev.kind = "alarm_clear";
      ev.zone_id = z.def.id;
      ev.track_id = a.track_ids.empty() ? std::nullopt : std::optional<int>(a.track_ids.front());
      ev.dir = std::nullopt;
      ev.gen = z.def.gen;
      ev.alarm_id = a.id;
      out.push_back(ev);
      alarms_.erase(alarms_.begin() + static_cast<std::ptrdiff_t>(i));
    }
  }

  // --- Sonra açılışlar. ---
  if (z.def.kind == "restricted") {
    std::vector<int> inside;
    for (const auto& kv : z.tracks) {
      if (kv.second.state == "inside") inside.push_back(kv.first);
    }
    if (inside.empty()) return;
    bool open = false;
    for (const Alarm& a : alarms_) {
      if (a.zone_id == z.def.id && a.kind == "intrusion") {
        open = true;
        break;
      }
    }
    if (open) return;
    Alarm a;
    a.id = z.def.id + ":intrusion:" + std::to_string(n);
    a.kind = "intrusion";
    a.zone_id = z.def.id;
    a.track_ids = inside;
    a.since_n = n;
    a.since_frame = frame;
    a.acked = false;
    alarms_.push_back(a);
    ZoneEvent ev;
    ev.n = n;
    ev.frame = frame;
    ev.kind = "alarm_start";
    ev.zone_id = z.def.id;
    ev.track_id = std::nullopt;
    ev.dir = std::nullopt;
    ev.gen = z.def.gen;
    ev.alarm_id = a.id;
    out.push_back(ev);
  } else if (z.def.kind == "dwell") {
    const int dwell_frames = z.def.dwell_frames ? *z.def.dwell_frames : 0;
    for (const auto& kv : z.tracks) {
      const int id = kv.first;
      const TrackZone& tz = kv.second;
      if (tz.state != "inside" || tz.stopped_run < dwell_frames) continue;
      bool open = false;
      for (const Alarm& a : alarms_) {
        if (a.zone_id == z.def.id && a.kind == "dwell_exceeded" && !a.track_ids.empty() &&
            a.track_ids.front() == id) {
          open = true;
          break;
        }
      }
      if (open) continue;
      Alarm a;
      a.id = z.def.id + ":dwell_exceeded:" + std::to_string(id) + ":" + std::to_string(n);
      a.kind = "dwell_exceeded";
      a.zone_id = z.def.id;
      a.track_ids = {id};
      a.since_n = n;
      a.since_frame = frame;
      a.acked = false;
      alarms_.push_back(a);
      z.exceeded.insert(id);
      ZoneEvent ev;
      ev.n = n;
      ev.frame = frame;
      ev.kind = "alarm_start";
      ev.zone_id = z.def.id;
      ev.track_id = id;
      ev.dir = std::nullopt;
      ev.gen = z.def.gen;
      ev.alarm_id = a.id;
      out.push_back(ev);
    }
  }
}

}  // namespace baykus
