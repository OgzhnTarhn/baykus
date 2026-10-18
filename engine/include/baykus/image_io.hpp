#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace baykus {

// RGB BGR uint8 (H, W, 3), IR gri uint8 (H, W); Python `baykus.engine_ref.compose.decode_pair`.
struct Pair {
  cv::Mat bgr;
  cv::Mat ir;
};

// Kodlanmış baytlar → çift. Çözülemeyen görüntü ya da farklı boyut: std::invalid_argument.
Pair decode_pair(const std::string& rgb_bytes, const std::string& ir_bytes);

// R, G, B, T sırasıyla CV_8UC4 (Python `baykus.data.multichannel.stack_rgbt`).
cv::Mat stack_rgbt(const cv::Mat& bgr, const cv::Mat& ir);

}  // namespace baykus
