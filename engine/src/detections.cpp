#include "baykus/detections.hpp"

#include <algorithm>
#include <stdexcept>

namespace baykus {

std::vector<Det> decode_rows(const std::vector<float>& out, const Letterbox& lb, float floor) {
  if (out.size() != static_cast<std::size_t>(kRows) * kCols) {
    throw std::invalid_argument("decode_rows: [1, 300, 6] bekleniyor");
  }
  std::vector<Det> dets;
  const double w = lb.width;
  const double h = lb.height;
  for (int i = 0; i < kRows; ++i) {
    const float* r = &out[static_cast<std::size_t>(i) * kCols];
    const float raw = r[4];
    if (raw < floor) {
      continue;
    }
    double x1 = (static_cast<double>(r[0]) - lb.pad_x) / lb.gain;
    double y1 = (static_cast<double>(r[1]) - lb.pad_y) / lb.gain;
    double x2 = (static_cast<double>(r[2]) - lb.pad_x) / lb.gain;
    double y2 = (static_cast<double>(r[3]) - lb.pad_y) / lb.gain;
    x1 = std::min(std::max(x1, 0.0), w);
    x2 = std::min(std::max(x2, 0.0), w);
    y1 = std::min(std::max(y1, 0.0), h);
    y2 = std::min(std::max(y2, 0.0), h);
    if (x2 - x1 <= 0 || y2 - y1 <= 0) {
      continue;
    }
    dets.push_back(Det{i, static_cast<int>(r[5]), raw, static_cast<double>(raw), x1, y1, x2, y2});
  }
  return dets;
}

void order(std::vector<Det>& dets) {
  std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.score_raw != b.score_raw) return a.score_raw > b.score_raw;
    return a.index < b.index;
  });
}

std::vector<Det> filter_display(const std::vector<Det>& dets, double conf) {
  std::vector<Det> out;
  for (const auto& d : dets) {
    if (d.score >= conf) out.push_back(d);
  }
  return out;
}

}  // namespace baykus
