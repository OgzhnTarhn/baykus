#include "baykus/track_summary.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include "baykus/classes.hpp"
#include "baykus/geometry.hpp"

namespace baykus {

namespace {

// İz başına kutu boyu örneği üst sınırı (tasarım §5.12 bellek sınırı).
constexpr std::size_t kSizeSamplesMax = 4096;

// Boyların ortancası (çift sayıda ortadaki iki değerin ortalaması).
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  if (n % 2 == 1) {
    return v[n / 2];
  }
  return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// kClassNames adı; bilinmeyen cls_id → "unknown".
std::string class_name(int cls_id) {
  if (cls_id >= 0 && cls_id < static_cast<int>(kClassNames.size())) {
    return std::string(kClassNames[static_cast<std::size_t>(cls_id)]);
  }
  return "unknown";
}

}  // namespace

TrackSummaries::TrackSummaries(const SummaryCfg& cfg) : cfg_(cfg) {}

void TrackSummaries::reset() { tracks_.clear(); }

void TrackSummaries::update(int n, int frame, const std::vector<TrackOut>& tracks,
                            const std::map<int, TrackMotion>& motion,
                            const std::vector<ZoneEvent>& events, const EgoResult& ego) {
  (void)n;

  // Bu karede görülmeyen kayıtlar "bitmiş" sayılır; birleşmede son noktalar hedef nesle taşınır.
  for (auto& kv : tracks_) {
    Track& rec = kv.second;
    rec.alive = false;
    if (ego.merge && rec.last_point && rec.last_point_gen == ego.merge->from_gen) {
      rec.last_point = apply(ego.merge->transform, *rec.last_point);
      rec.last_point_gen = ego.merge->to_gen;
    }
  }

  for (const auto& t : tracks) {
    const auto mit = motion.find(t.track_id);
    const std::string m = mit != motion.end() ? mit->second.motion : "unknown";

    auto it = tracks_.find(t.track_id);
    if (it == tracks_.end()) {
      // Yeni iz: ilk bölüm bu karenin hareketiyle açılır.
      Track rec;
      rec.first_frame = frame;
      rec.motion = m;
      rec.motion_since_frame = frame;
      rec.segments.push_back(Segment{m, frame, frame});
      it = tracks_.emplace(t.track_id, std::move(rec)).first;
    }
    Track& rec = it->second;
    rec.alive = true;
    rec.last_frame = frame;

    // Sınıf: en çok sayılan cls_id (eşitlikte küçük).
    rec.cls_counts[t.cls_id]++;
    int best = rec.cls_id;
    int best_count = -1;
    for (const auto& kv : rec.cls_counts) {
      if (kv.second > best_count) {
        best_count = kv.second;
        best = kv.first;
      }
    }
    rec.cls_id = best;

    // Hareket bölümleri.
    if (m != rec.motion) {
      rec.motion = m;
      rec.motion_since_frame = frame;
      rec.segments.push_back(Segment{m, frame, frame});
    } else if (!rec.segments.empty()) {
      rec.segments.back().to_frame = frame;
    }
    if (m == "stationary") {
      rec.stationary_frames++;
    }

    // Bölüm sayısı sınırı: en eski iki bölüm birleşir (uzun olanın durumu, eşitlikte ilkinin).
    while (static_cast<int>(rec.segments.size()) > cfg_.max_segments && rec.segments.size() >= 2) {
      const Segment a = rec.segments[0];
      const Segment b = rec.segments[1];
      const int da = a.to_frame - a.from_frame + 1;
      const int db = b.to_frame - b.from_frame + 1;
      Segment merged;
      merged.from_frame = a.from_frame;
      merged.to_frame = b.to_frame;
      merged.state = db > da ? b.state : a.state;
      rec.segments.erase(rec.segments.begin(), rec.segments.begin() + 2);
      rec.segments.insert(rec.segments.begin(), merged);
    }

    // Yol: harita adımları yalnız aynı nesilde birikir.
    if (ego.ok && ego.frame_to_map && mit != motion.end() && mit->second.map_xy) {
      const cv::Point2d p = *mit->second.map_xy;
      const double w = t.box.x2 - t.box.x1;
      const double h = t.box.y2 - t.box.y1;
      // Boy örnekleri: sınır aşılınca eşit aralıkla seyreltilir (ortanca yaklaşık kalır).
      rec.size_seen++;
      if ((rec.size_seen - 1) % rec.size_stride == 0) {
        rec.sizes.push_back(std::sqrt(w * h) * scale(*ego.frame_to_map));
        if (rec.sizes.size() > kSizeSamplesMax) {
          std::size_t keep = 0;
          for (std::size_t i = 0; i < rec.sizes.size(); i += 2) {
            rec.sizes[keep++] = rec.sizes[i];
          }
          rec.sizes.resize(keep);
          rec.size_stride *= 2;
        }
      }
      if (rec.last_point && rec.last_point_gen == ego.map_gen) {
        rec.path += std::hypot(p.x - rec.last_point->x, p.y - rec.last_point->y);
      }
      rec.last_point = p;
      rec.last_point_gen = ego.map_gen;
    }
  }

  // Bölge olayları: giriş açık ziyaret açar, çıkış son açık ziyareti kapatır.
  for (const auto& e : events) {
    if (!e.track_id) {
      continue;
    }
    const auto it = tracks_.find(*e.track_id);
    if (it == tracks_.end()) {
      continue;
    }
    Track& rec = it->second;
    if (e.kind == "enter") {
      rec.zones.push_back(Visit{e.zone_id, e.frame, std::nullopt});
    } else if (e.kind == "exit" || e.kind == "exit_lost") {
      for (auto rit = rec.zones.rbegin(); rit != rec.zones.rend(); ++rit) {
        if (rit->zone_id == e.zone_id && !rit->exit_frame) {
          rit->exit_frame = e.frame;
          break;
        }
      }
    }
  }

  // Bitmiş kayıt sayısı ended_keep'i aşarsa en eski (last_frame, id) silinir.
  int ended = 0;
  for (const auto& kv : tracks_) {
    if (!kv.second.alive) {
      ended++;
    }
  }
  while (ended > cfg_.ended_keep) {
    auto victim = tracks_.end();
    for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
      if (it->second.alive) {
        continue;
      }
      if (victim == tracks_.end() || it->second.last_frame < victim->second.last_frame) {
        victim = it;
      }
    }
    if (victim == tracks_.end()) {
      break;
    }
    tracks_.erase(victim);
    ended--;
  }
}

std::size_t TrackSummaries::size_samples(int id) const {
  const auto it = tracks_.find(id);
  if (it == tracks_.end()) {
    return 0;
  }
  return it->second.sizes.size();
}

std::optional<nlohmann::json> TrackSummaries::get(int id) const {
  const auto it = tracks_.find(id);
  if (it == tracks_.end()) {
    return std::nullopt;
  }
  const Track& rec = it->second;

  nlohmann::json j;
  j["track_id"] = id;
  j["first_frame"] = rec.first_frame;
  j["last_frame"] = rec.last_frame;
  j["cls_id"] = rec.cls_id;
  j["class"] = class_name(rec.cls_id);
  j["motion"] = rec.motion;
  j["motion_since_frame"] = rec.motion_since_frame;

  nlohmann::json segments = nlohmann::json::array();
  for (const auto& s : rec.segments) {
    segments.push_back(
        {{"state", s.state}, {"from_frame", s.from_frame}, {"to_frame", s.to_frame}});
  }
  j["segments"] = std::move(segments);

  j["stationary_frames"] = rec.stationary_frames;
  const double med = rec.sizes.empty() ? 0.0 : median(rec.sizes);
  if (med > 0.0) {
    j["path_rel"] = rec.path / med;
  } else {
    j["path_rel"] = nullptr;
  }

  nlohmann::json zones = nlohmann::json::array();
  for (const auto& v : rec.zones) {
    nlohmann::json z;
    z["zone_id"] = v.zone_id;
    z["enter_frame"] = v.enter_frame;
    if (v.exit_frame) {
      z["exit_frame"] = *v.exit_frame;
    } else {
      z["exit_frame"] = nullptr;
    }
    zones.push_back(std::move(z));
  }
  j["zones"] = std::move(zones);

  j["alive"] = rec.alive;
  return j;
}

}  // namespace baykus
