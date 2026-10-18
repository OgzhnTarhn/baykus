#include "baykus/preprocess.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace baykus {

namespace {

constexpr int kPad = 114;

// Python round(): yarıda çifte. std::nearbyint geçerli yuvarlama kipini kullanır (FE_TONEAREST).
int py_round(double v) {
  const int mode = std::fegetround();
  std::fesetround(FE_TONEAREST);
  const double r = std::nearbyint(v);
  std::fesetround(mode);
  return static_cast<int>(r);
}

}  // namespace

std::vector<float> letterbox(const cv::Mat& img, int imgsz, Letterbox* lb) {
  if (img.empty() || img.depth() != CV_8U) {
    throw std::invalid_argument("letterbox: uint8 görüntü bekleniyor");
  }
  const int h = img.rows;
  const int w = img.cols;
  const int c = img.channels();
  const double gain = std::min(static_cast<double>(imgsz) / h, static_cast<double>(imgsz) / w);
  const int new_w = py_round(w * gain);
  const int new_h = py_round(h * gain);
  cv::Mat resized = img;
  if (new_w != w || new_h != h) {
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  }
  const double dw = (imgsz - new_w) / 2.0;
  const double dh = (imgsz - new_h) / 2.0;
  const int top = py_round(dh - 0.1);
  const int left = py_round(dw - 0.1);
  cv::Mat canvas(imgsz, imgsz, CV_8UC(c), cv::Scalar::all(kPad));
  resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));

  const std::size_t plane = static_cast<std::size_t>(imgsz) * imgsz;
  std::vector<float> out(plane * c);
  for (int y = 0; y < imgsz; ++y) {
    const uchar* row = canvas.ptr<uchar>(y);
    for (int x = 0; x < imgsz; ++x) {
      for (int k = 0; k < c; ++k) {
        const int src = (c == 3) ? 2 - k : k;  // 3 kanalda BGR → RGB; 4 kanalda sıra korunur
        out[k * plane + static_cast<std::size_t>(y) * imgsz + x] =
            static_cast<float>(row[x * c + src]) / 255.0f;
      }
    }
  }
  if (lb != nullptr) {
    *lb = Letterbox{gain, left, top, w, h};
  }
  return out;
}

}  // namespace baykus
