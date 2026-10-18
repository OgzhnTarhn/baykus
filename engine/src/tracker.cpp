#include "baykus/tracker.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <stdexcept>

#include "baykus/files.hpp"
#include "baykus/lsap.hpp"

namespace baykus {

namespace {

using json = nlohmann::json;
using Box = std::array<double, 4>;

constexpr const char* kKeys[] = {"lost_track_buffer",          "frame_rate",
                                 "track_activation_threshold", "minimum_consecutive_frames",
                                 "minimum_iou_threshold",      "high_conf_det_threshold"};

Box box_of(const Det& d) { return {d.x1, d.y1, d.x2, d.y2}; }

// supervision.box_iou_batch: köşeler double, oran float32'ye yazılır; payda ≤ 0 ise 0.
float iou_f32(const Box& a, const Box& b) {
  const double iw = std::max(std::min(a[2], b[2]) - std::max(a[0], b[0]), 0.0);
  const double ih = std::max(std::min(a[3], b[3]) - std::max(a[1], b[1]), 0.0);
  const double inter = iw * ih;
  const double norm = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return norm > 0 ? static_cast<float>(inter / norm) : 0.0f;
}

Eigen::MatrixXf iou_matrix(const std::vector<Box>& rows, const std::vector<Box>& cols) {
  Eigen::MatrixXf m(rows.size(), cols.size());
  for (std::size_t i = 0; i < rows.size(); ++i)
    for (std::size_t j = 0; j < cols.size(); ++j) m(i, j) = iou_f32(rows[i], cols[j]);
  return m;
}

json matrix_json(const Eigen::MatrixXf& m) {
  json out = json::array();
  for (Eigen::Index i = 0; i < m.rows(); ++i) {
    json row = json::array();
    for (Eigen::Index j = 0; j < m.cols(); ++j) row.push_back(static_cast<double>(m(i, j)));
    out.push_back(row);
  }
  return out;
}

struct Association {
  std::vector<std::pair<int, int>> matched;
  std::vector<int> unmatched_rows;  // artan
  std::vector<int> unmatched_cols;  // artan
};

// `_get_associated_indices`: önce atama (büyütme), sonra eşik.
Association associate(const Eigen::MatrixXf& sim, float threshold) {
  Association a;
  std::set<int> rows, cols;
  for (int i = 0; i < sim.rows(); ++i) rows.insert(i);
  for (int j = 0; j < sim.cols(); ++j) cols.insert(j);
  if (sim.rows() > 0 && sim.cols() > 0) {
    for (const auto& [r, c] : lsap_maximize(sim.cast<double>())) {
      if (sim(r, c) >= threshold) {
        a.matched.emplace_back(r, c);
        rows.erase(r);
        cols.erase(c);
      }
    }
  }
  a.unmatched_rows.assign(rows.begin(), rows.end());
  a.unmatched_cols.assign(cols.begin(), cols.end());
  return a;
}

json pairs_json(const std::vector<std::pair<int, int>>& pairs) {
  json out = json::array();
  for (const auto& [r, c] : pairs) out.push_back({r, c});
  return out;
}

}  // namespace

TrackerConfig TrackerConfig::load(const std::string& path) {
  const auto j = json::parse(read_bytes(path));
  if (!j.is_object()) throw std::runtime_error(path + ": takip parametreleri nesne olmalı");
  for (const char* k : kKeys) {
    if (!j.contains(k)) throw std::runtime_error(path + ": takip parametresi eksik: " + k);
  }
  if (j.size() != std::size(kKeys)) {
    throw std::runtime_error(path + ": bilinmeyen takip parametresi (yalnız altı anahtar)");
  }
  TrackerConfig c;
  c.lost_track_buffer = j.at("lost_track_buffer").get<int>();
  c.frame_rate = j.at("frame_rate").get<double>();
  c.track_activation_threshold = j.at("track_activation_threshold").get<double>();
  c.minimum_consecutive_frames = j.at("minimum_consecutive_frames").get<int>();
  c.minimum_iou_threshold = j.at("minimum_iou_threshold").get<double>();
  c.high_conf_det_threshold = j.at("high_conf_det_threshold").get<double>();
  if (c.lost_track_buffer < 0) throw std::runtime_error(path + ": lost_track_buffer < 0");
  if (!std::isfinite(c.frame_rate) || c.frame_rate <= 0) {
    throw std::runtime_error(path + ": frame_rate sonlu ve pozitif olmalı");
  }
  return c;
}

json TrackerConfig::to_json() const {
  return {{"lost_track_buffer", lost_track_buffer},
          {"frame_rate", frame_rate},
          {"track_activation_threshold", track_activation_threshold},
          {"minimum_consecutive_frames", minimum_consecutive_frames},
          {"minimum_iou_threshold", minimum_iou_threshold},
          {"high_conf_det_threshold", high_conf_det_threshold}};
}

int TrackerConfig::maximum_frames_without_update() const {
  if (lost_track_buffer == 0) return 0;
  return std::max(1, static_cast<int>(std::ceil(frame_rate / 30.0 * lost_track_buffer)));
}

ByteTrack::ByteTrack(const TrackerConfig& cfg)
    : cfg_(cfg),
      max_frames_(cfg.maximum_frames_without_update()),
      iou_threshold_(static_cast<float>(cfg.minimum_iou_threshold)) {}

void ByteTrack::reset() {
  tracks_.clear();
  next_id_ = 0;
}

json ByteTrack::track_rows() const {
  json rows = json::array();
  for (const auto& t : tracks_) {
    const auto b = t.kf.bbox();
    rows.push_back({b[0], b[1], b[2], b[3], t.tracker_id});
  }
  return rows;
}

std::vector<std::pair<int, int>> ByteTrack::update(const std::vector<Det>& dets, json* dump) {
  const json before = dump != nullptr ? track_rows() : json();
  json dump_dets = json::array();
  if (dump != nullptr) {
    for (const auto& d : dets) {
      dump_dets.push_back({{"index", d.index}, {"score_raw", static_cast<double>(d.score_raw)}});
    }
  }
  if (tracks_.empty() && dets.empty()) {
    if (dump != nullptr) {
      *dump = {{"tracks_before", before}, {"tracks_after", track_rows()}, {"dets", dump_dets},
               {"iou1", json::array()},   {"assign1", json::array()},     {"iou2", json::array()},
               {"assign2", json::array()}};
    }
    return {};
  }

  // Kalman tahmini; kaçırılan karede ardışık güncelleme sayacı sıfırlanır.
  for (auto& t : tracks_) {
    t.kf.predict();
    if (t.time_since_update > 0) t.consecutive = 0;
    ++t.time_since_update;
  }

  std::vector<int> high, low;
  for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
    (static_cast<double>(dets[i].score_raw) >= cfg_.high_conf_det_threshold ? high : low)
        .push_back(i);
  }
  std::vector<Box> predicted, high_boxes, low_boxes;
  for (const auto& t : tracks_) predicted.push_back(t.kf.bbox());
  for (int i : high) high_boxes.push_back(box_of(dets[i]));
  for (int i : low) low_boxes.push_back(box_of(dets[i]));

  std::vector<std::pair<int, int>> out;
  auto on_match = [&](Track& t, int det_i) {
    t.kf.update(box_of(dets[det_i]));
    t.time_since_update = 0;
    ++t.consecutive;
    if (t.consecutive >= cfg_.minimum_consecutive_frames && t.tracker_id == -1) {
      t.tracker_id = next_id_++;
    }
    out.emplace_back(dets[det_i].index, t.tracker_id);
  };

  // 1. aşama: yüksek skorlular bütün izlerle
  const Eigen::MatrixXf iou1 = iou_matrix(predicted, high_boxes);
  const auto a1 = associate(iou1, iou_threshold_);
  for (const auto& [r, c] : a1.matched) on_match(tracks_[r], high[c]);

  // 2. aşama: düşük skorlular kalan izlerle
  std::vector<Box> remaining_boxes;
  for (int r : a1.unmatched_rows) remaining_boxes.push_back(predicted[r]);
  const Eigen::MatrixXf iou2 = iou_matrix(remaining_boxes, low_boxes);
  const auto a2 = associate(iou2, iou_threshold_);
  for (const auto& [r, c] : a2.matched) on_match(tracks_[a1.unmatched_rows[r]], low[c]);
  for (int c : a2.unmatched_cols) out.emplace_back(dets[low[c]].index, -1);

  // Eşleşmeyen yüksek skorlular: -1; etkinleştirme eşiğini geçen yeni iz açar.
  for (int c : a1.unmatched_cols) {
    const int g = high[c];
    out.emplace_back(dets[g].index, -1);
    if (static_cast<double>(dets[g].score_raw) >= cfg_.track_activation_threshold) {
      tracks_.push_back(Track{KalmanBox(box_of(dets[g]))});
    }
  }

  // Yaşayan izler: bütçe içinde ve (kimlikli ya da olgun ya da bu karede güncellenmiş).
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [&](const Track& t) {
                                 const bool mature =
                                     t.tracker_id != -1 ||
                                     t.consecutive >= cfg_.minimum_consecutive_frames;
                                 return !(t.time_since_update <= max_frames_ &&
                                          (mature || t.time_since_update == 0));
                               }),
                tracks_.end());

  if (dump != nullptr) {
    *dump = {{"tracks_before", before},
             {"tracks_after", track_rows()},
             {"dets", dump_dets},
             {"iou1", matrix_json(iou1)},
             {"assign1", pairs_json(a1.matched)},
             {"iou2", matrix_json(iou2)},
             {"assign2", pairs_json(a2.matched)}};
  }
  return out;
}

}  // namespace baykus
