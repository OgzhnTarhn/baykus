#include "baykus/motion_config.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "baykus/files.hpp"

namespace baykus {
namespace {

using json = nlohmann::json;

std::string join(const std::vector<std::string>& items) {
  std::string out;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i) out += ", ";
    out += items[i];
  }
  return out;
}

// `raw` nesnesinin anahtarları tam olarak `keys` olmalı; eksik/fazla → std::invalid_argument.
void check_keys(const std::string& path, const std::string& section, const json& raw,
                const std::vector<std::string>& keys) {
  if (!raw.is_object()) {
    throw std::invalid_argument(path + ": '" + section + "' nesne olmalı");
  }
  std::vector<std::string> missing;
  for (const auto& k : keys) {
    if (!raw.contains(k)) missing.push_back(k);
  }
  std::vector<std::string> extra;
  for (auto it = raw.begin(); it != raw.end(); ++it) {
    if (std::find(keys.begin(), keys.end(), it.key()) == keys.end()) extra.push_back(it.key());
  }
  std::sort(extra.begin(), extra.end());
  if (!missing.empty() || !extra.empty()) {
    throw std::invalid_argument(path + ": '" + section + "' anahtarları [" + join(keys) +
                                "]; eksik [" + join(missing) + "], fazla [" + join(extra) + "]");
  }
}

template <typename T>
T get(const json& raw, const char* key) {
  return raw.at(key).get<T>();
}

EgoCfg parse_ego(const std::string& path, const json& raw) {
  check_keys(
      path, "ego", raw,
      {"enabled",         "max_corners",    "quality",         "min_distance",     "block_size",
       "lk_win",          "lk_levels",      "relink_levels",   "fb_max_px",        "ransac_px",
       "ransac_iters",    "confidence",     "refine_iters",    "ok_min_inliers",   "ok_min_ratio",
       "key_min_inliers", "key_min_ratio",  "key_max_age",     "key_max_shift_px", "mask_dilate_px",
       "border_px",       "relink_max_gap", "ir_return_frames"});
  EgoCfg c;
  c.enabled = get<bool>(raw, "enabled");
  c.max_corners = get<int>(raw, "max_corners");
  c.quality = get<double>(raw, "quality");
  c.min_distance = get<double>(raw, "min_distance");
  c.block_size = get<int>(raw, "block_size");
  c.lk_win = get<int>(raw, "lk_win");
  c.lk_levels = get<int>(raw, "lk_levels");
  c.relink_levels = get<int>(raw, "relink_levels");
  c.fb_max_px = get<double>(raw, "fb_max_px");
  c.ransac_px = get<double>(raw, "ransac_px");
  c.ransac_iters = get<int>(raw, "ransac_iters");
  c.confidence = get<double>(raw, "confidence");
  c.refine_iters = get<int>(raw, "refine_iters");
  c.ok_min_inliers = get<int>(raw, "ok_min_inliers");
  c.ok_min_ratio = get<double>(raw, "ok_min_ratio");
  c.key_min_inliers = get<int>(raw, "key_min_inliers");
  c.key_min_ratio = get<double>(raw, "key_min_ratio");
  c.key_max_age = get<int>(raw, "key_max_age");
  c.key_max_shift_px = get<double>(raw, "key_max_shift_px");
  c.mask_dilate_px = get<double>(raw, "mask_dilate_px");
  c.border_px = get<int>(raw, "border_px");
  c.relink_max_gap = get<int>(raw, "relink_max_gap");
  c.ir_return_frames = get<int>(raw, "ir_return_frames");
  return c;
}

RelocCfg parse_reloc(const std::string& path, const json& raw) {
  check_keys(path, "reloc", raw,
             {"orb_features", "candidates", "max_dframe", "window_frames", "every", "ransac_px",
              "min_inliers", "min_ratio", "library_max"});
  RelocCfg c;
  c.orb_features = get<int>(raw, "orb_features");
  c.candidates = get<int>(raw, "candidates");
  c.max_dframe = get<int>(raw, "max_dframe");
  c.window_frames = get<int>(raw, "window_frames");
  c.every = get<int>(raw, "every");
  c.ransac_px = get<double>(raw, "ransac_px");
  c.min_inliers = get<int>(raw, "min_inliers");
  c.min_ratio = get<double>(raw, "min_ratio");
  c.library_max = get<int>(raw, "library_max");
  return c;
}

StateCfg parse_state(const std::string& path, const json& raw) {
  check_keys(path, "motion", raw, {"k", "stationary_max", "moving_min", "hold"});
  StateCfg c;
  c.k = get<int>(raw, "k");
  c.stationary_max = get<double>(raw, "stationary_max");
  c.moving_min = get<double>(raw, "moving_min");
  c.hold = get<int>(raw, "hold");
  return c;
}

FrozenCfg parse_frozen(const std::string& path, const json& raw) {
  check_keys(path, "frozen", raw, {"mad_max", "window", "min_frames"});
  FrozenCfg c;
  c.mad_max = get<double>(raw, "mad_max");
  c.window = get<int>(raw, "window");
  c.min_frames = get<int>(raw, "min_frames");
  return c;
}

ZonesCfg parse_zones(const std::string& path, const json& raw) {
  check_keys(
      path, "zones", raw,
      {"max_zones", "max_points", "name_max", "dwell_max", "enter_obs", "exit_obs", "gap_max",
       "lost_frames", "stopped_obs", "dwell_clear_moving", "line_side_obs", "body_max_bytes"});
  ZonesCfg c;
  c.max_zones = get<int>(raw, "max_zones");
  c.max_points = get<int>(raw, "max_points");
  c.name_max = get<int>(raw, "name_max");
  c.dwell_max = get<int>(raw, "dwell_max");
  c.enter_obs = get<int>(raw, "enter_obs");
  c.exit_obs = get<int>(raw, "exit_obs");
  c.gap_max = get<int>(raw, "gap_max");
  c.lost_frames = get<int>(raw, "lost_frames");
  c.stopped_obs = get<int>(raw, "stopped_obs");
  c.dwell_clear_moving = get<int>(raw, "dwell_clear_moving");
  c.line_side_obs = get<int>(raw, "line_side_obs");
  c.body_max_bytes = get<int>(raw, "body_max_bytes");
  return c;
}

LogCfg parse_log(const std::string& path, const json& raw) {
  check_keys(path, "log", raw, {"max_events"});
  LogCfg c;
  c.max_events = get<int>(raw, "max_events");
  return c;
}

SummaryCfg parse_summary(const std::string& path, const json& raw) {
  check_keys(path, "summary", raw, {"max_segments", "ended_keep"});
  SummaryCfg c;
  c.max_segments = get<int>(raw, "max_segments");
  c.ended_keep = get<int>(raw, "ended_keep");
  return c;
}

}  // namespace

MotionCfg MotionCfg::load(const std::string& path) {
  const std::string data = read_bytes(path);
  json raw;
  try {
    raw = json::parse(data);
  } catch (const json::exception& e) {
    throw std::invalid_argument(path + ": JSON ayrıştırılamadı: " + e.what());
  }
  if (!raw.is_object()) {
    throw std::invalid_argument(path + ": kök nesne olmalı");
  }
  const std::vector<std::string> sections = {"ego",   "reloc", "motion", "frozen",
                                             "zones", "log",   "summary"};
  std::vector<std::string> missing;
  for (const auto& s : sections) {
    if (!raw.contains(s)) missing.push_back(s);
  }
  std::vector<std::string> extra;
  for (auto it = raw.begin(); it != raw.end(); ++it) {
    if (std::find(sections.begin(), sections.end(), it.key()) == sections.end()) {
      extra.push_back(it.key());
    }
  }
  std::sort(extra.begin(), extra.end());
  if (!missing.empty() || !extra.empty()) {
    throw std::invalid_argument(path + ": bölümler [" + join(sections) + "]; eksik [" +
                                join(missing) + "], fazla [" + join(extra) + "]");
  }

  MotionCfg cfg;
  cfg.ego = parse_ego(path, raw.at("ego"));
  cfg.reloc = parse_reloc(path, raw.at("reloc"));
  cfg.motion = parse_state(path, raw.at("motion"));
  cfg.frozen = parse_frozen(path, raw.at("frozen"));
  cfg.zones = parse_zones(path, raw.at("zones"));
  cfg.log = parse_log(path, raw.at("log"));
  cfg.summary = parse_summary(path, raw.at("summary"));
  cfg.sha256 = sha256_hex(data);
  return cfg;
}

}  // namespace baykus
