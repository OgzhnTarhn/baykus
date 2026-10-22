#include "baykus/frozen.hpp"

#include <opencv2/imgproc.hpp>

namespace baykus {

FrozenDetector::FrozenDetector(const FrozenCfg& cfg) : cfg_(cfg) { reset(); }

void FrozenDetector::reset() {
  rgb_ = Channel{};
  ir_ = Channel{};
}

nlohmann::json FrozenDetector::channel(Channel& ch, const cv::Mat& img, bool lost) {
  cv::Mat prev = ch.prev;
  ch.prev = img;
  double mad = 0.0;
  bool has_mad = false;
  if (!img.empty() && !prev.empty() && prev.size() == img.size()) {
    cv::Mat diff;
    cv::absdiff(img, prev, diff);
    mad = cv::mean(diff)[0];
    has_mad = true;
  }
  bool frozen = has_mad && !lost && mad <= cfg_.mad_max;
  ch.buf.push_back(frozen);
  while (static_cast<int>(ch.buf.size()) > cfg_.window) ch.buf.pop_front();
  int count = 0;
  for (bool b : ch.buf) count += b ? 1 : 0;
  nlohmann::json out;
  out["frozen"] = frozen;
  out["diff_mad"] = has_mad ? nlohmann::json(mad) : nlohmann::json(nullptr);
  out["alarm"] = count >= cfg_.min_frames;
  return out;
}

nlohmann::json FrozenDetector::update(const cv::Mat& rgb_half, const cv::Mat& ir_half,
                                      bool rgb_lost, bool ir_lost) {
  nlohmann::json out;
  out["rgb"] = channel(rgb_, rgb_half, rgb_lost);
  out["ir"] = channel(ir_, ir_half, ir_lost);
  return out;
}

}  // namespace baykus
