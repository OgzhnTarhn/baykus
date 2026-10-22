#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace baykus {

// `configs/engine/motion.json` (Python `baykus.motion.config` karşılığı; tasarım §5.7).
// Anahtarlar birebir: eksik ya da fazla anahtar std::invalid_argument (yazım hatası sessizce
// varsayılana düşmesin). `sha256` dosya baytlarının özeti.

struct EgoCfg {
  bool enabled;
  int max_corners;
  double quality;
  double min_distance;
  int block_size;
  int lk_win;
  int lk_levels;
  int relink_levels;
  double fb_max_px;
  double ransac_px;
  int ransac_iters;
  double confidence;
  int refine_iters;
  int ok_min_inliers;
  double ok_min_ratio;
  int key_min_inliers;
  double key_min_ratio;
  int key_max_age;
  double key_max_shift_px;
  double mask_dilate_px;
  int border_px;
  int relink_max_gap;
  int ir_return_frames;
};

struct RelocCfg {
  int orb_features;
  int candidates;
  int max_dframe;
  int window_frames;
  int every;
  double ransac_px;
  int min_inliers;
  double min_ratio;
  int library_max;
};

struct StateCfg {
  int k;
  double stationary_max;
  double moving_min;
  int hold;
};

struct FrozenCfg {
  double mad_max;
  int window;
  int min_frames;
};

struct ZonesCfg {
  int max_zones;
  int max_points;
  int name_max;
  int dwell_max;
  int enter_obs;
  int exit_obs;
  int gap_max;
  int lost_frames;
  int stopped_obs;
  int dwell_clear_moving;
  int line_side_obs;
  int body_max_bytes;
};

struct LogCfg {
  int max_events;
};

struct SummaryCfg {
  int max_segments;
  int ended_keep;
};

struct MotionCfg {
  EgoCfg ego;
  RelocCfg reloc;
  StateCfg motion;
  FrozenCfg frozen;
  ZonesCfg zones;
  LogCfg log;
  SummaryCfg summary;
  std::string sha256;
  nlohmann::json raw;

  nlohmann::json to_json() const { return raw; }

  // Dosyayı okur ve katı şemayla ayrıştırır; hata std::invalid_argument (yol iletide).
  static MotionCfg load(const std::string& path);
};

}  // namespace baykus
