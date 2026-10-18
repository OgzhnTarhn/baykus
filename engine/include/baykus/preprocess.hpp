#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace baykus {

// Python `baykus.edge.frames.Letterbox` ile aynı alanlar.
struct Letterbox {
  double gain;
  int pad_x;
  int pad_y;
  int width;
  int height;
};

// Ultralytics LetterBox(auto=False) = Python `preprocess`: gain = min(imgsz/h, imgsz/w), yeni boyut
// Python round (yarıda çifte), INTER_LINEAR, dolgu 114, üst/sol = round(d − 0,1), 3 kanalda BGR → RGB,
// 4 kanalda sıra korunur, float32 / 255, CHW. `lb` doldurulur.
std::vector<float> letterbox(const cv::Mat& img, int imgsz, Letterbox* lb);

}  // namespace baykus
