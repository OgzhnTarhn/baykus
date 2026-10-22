#pragma once

#include <deque>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "baykus/ego.hpp"
#include "baykus/geometry.hpp"
#include "baykus/motion_config.hpp"

namespace baykus {

// Python `baykus.motion.state.TrackObs`.
struct TrackObs {
  int track_id;
  Box box;
};

// Python `baykus.motion.state.TrackMotion`; `motion` = moving | stationary | unknown.
struct TrackMotion {
  std::string motion;
  std::optional<double> speed_rel;
  std::optional<cv::Point2d> map_xy;
};

// Python `baykus.motion.state.MotionState`: iz başına harita merkezi/boy geçmişi ve hız kararı.
// `t` = çağrı sayacı (oturum içi istek sırası `n`).
class MotionState {
 public:
  explicit MotionState(const StateCfg& cfg);

  std::map<int, TrackMotion> update(const EgoResult& ego, const std::vector<TrackObs>& tracks);

  void reset();

 private:
  // `n` → (nesil, harita merkezi, harita boyu).
  struct HistEntry {
    int gen;
    cv::Point2d c;
    double s;
  };

  struct Track {
    std::map<int, HistEntry> hist;
    std::string decision = "unknown";
    std::optional<int> last_v_n;
    int last_seen = 0;
  };

  struct EgoEntry {
    int n;
    bool ok;
    int gen;
  };

  // Birleşmede geçmişi `t` ile hedef nesle taşır (merkez `apply(t, c)`, boy `scale(t)·s`).
  void relabel_(int from_gen, int to_gen, const cv::Matx23d& t);
  // `(t − K, t]` bütün kareleri ok ve aynı nesilde mi (deque son K kareyi tutar).
  bool window_ok_(int gen) const;
  // `K + hold` karedir görünmeyen iz: geçmişi kullanılamaz, kararı süresi dolmuş.
  void prune_(int n);

  StateCfg cfg_;
  int n_ = 0;
  std::optional<int> gen_;    // son ok karenin nesli
  std::deque<EgoEntry> ego_;  // son K kare: (n, ok, gen)
  std::map<int, Track> tracks_;
};

}  // namespace baykus
