#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "baykus/detections.hpp"

namespace baykus {

// Python `baykus.engine_ref.tiles`: 2 × 2 pencere, sınıfsız NMS eşiği.
constexpr double kTileNmsIou = 0.5;

// (x, y, pw, ph) pencereleri; pw = ceil(w·9/16), ph = ceil(h·9/16). Sıra sol üst, sağ üst,
// sol alt, sağ alt.
std::vector<cv::Rect> tile_windows(int w, int h);

// Parça tespitlerini kare koordinatına taşıyıp sınıfsız NMS ile birleştirir; dönüş `order` sırası.
std::vector<Det> merge_tiles(const std::vector<std::vector<Det>>& per_tile,
                             const std::vector<cv::Rect>& wins);

}  // namespace baykus
