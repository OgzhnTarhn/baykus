#pragma once

// Faz 4 iz özetleri (docs/design/events_rules.md "TrackSummaries"). Parametreler `SummaryCfg`
// (max_segments, ended_keep).
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "baykus/ego.hpp"
#include "baykus/motion.hpp"
#include "baykus/motion_config.hpp"
#include "baykus/zones.hpp"

namespace baykus {

// İz başına hareket bölümleri, sınıf, yol ve bölge ziyaretleri.
class TrackSummaries {
 public:
  explicit TrackSummaries(const SummaryCfg& cfg);

  void update(int n, int frame, const std::vector<TrackOut>& tracks,
              const std::map<int, TrackMotion>& motion, const std::vector<ZoneEvent>& events,
              const EgoResult& ego);

  std::optional<nlohmann::json> get(int id) const;

  // İz başına tutulan kutu boyu örneği sayısı (test ve tanı için); bilinmeyen id → 0.
  std::size_t size_samples(int id) const;

  void reset();

 private:
  // Hareket bölümü; `state` = moving | stationary | unknown.
  struct Segment {
    std::string state;
    int from_frame;
    int to_frame;
  };

  // Bölge ziyareti; `exit_frame` açık ziyarette boş.
  struct Visit {
    std::string zone_id;
    int enter_frame;
    std::optional<int> exit_frame;
  };

  struct Track {
    int first_frame = 0;
    int last_frame = 0;
    std::map<int, int> cls_counts;  // cls_id → sayı
    int cls_id = 0;
    std::string motion = "unknown";
    int motion_since_frame = 0;
    std::vector<Segment> segments;
    int stationary_frames = 0;
    std::vector<double> sizes;  // kutu boyu örnekleri en çok 4096; aşınca eşit aralıkla
                                // seyreltilir (ortanca yaklaşık, tasarım §5.12 bellek sınırı)
    int size_stride = 1;        // kaç gözlemde bir örnek alınır
    int size_seen = 0;          // boy örneği adayı gözlem sayısı
    double path = 0.0;
    std::optional<cv::Point2d> last_point;  // son harita noktası
    int last_point_gen = 0;
    std::vector<Visit> zones;
    bool alive = false;
  };

  SummaryCfg cfg_;
  std::map<int, Track> tracks_;
};

}  // namespace baykus
