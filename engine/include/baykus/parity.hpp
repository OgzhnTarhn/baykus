#pragma once

// Kabul 1 eşlik kuralları (Faz 3 tasarımı §6.1; ölçümden önce sabit).
#include <array>
#include <string>
#include <vector>

#include "baykus/detections.hpp"

namespace baykus::parity {

constexpr double kRawTol = 1e-4;        // (a) 6 bileşen mutlak fark
constexpr double kRawTieTol = 1e-5;     // (a) ardışık Python skorları bu kadar yakınsa IoU eşlemesi
constexpr double kLayerIoU = 0.99;      // (b) eşleşen kutularda IoU
constexpr double kLayerRawTol = 1e-3;   // (b) ham skor farkı
constexpr double kBoundaryTol = 1e-4;   // (b) katman sınırına yakın satır serbest
constexpr double kSwapRawTol = 1e-4;    // (b) eşit kalibreli komşunun ham farkı
constexpr double kMinRatio = 0.99;      // (b) karşılaştırılan satır oranı

using Row = std::array<float, kCols>;

struct RawCompare {
  bool ok = true;
  int rows = 0;
  int swaps = 0;
  double max_diff = 0.0;
  std::vector<std::string> errors;
};

// (a) Satır satır; ardışık Python skorları kRawTieTol içindeki gruplarda satırlar kutu IoU'suyla eşlenir.
RawCompare compare_raw(const std::vector<Row>& py, const std::vector<Row>& cpp);

struct LayerCompare {
  bool ok = true;
  int py_rows = 0;
  int cpp_rows = 0;
  int compared = 0;
  int free_rows = 0;
  int swaps = 0;
  double ratio = 1.0;
  double max_raw_diff = 0.0;
  double min_iou = 1.0;
  std::vector<std::string> errors;
};

// (b) Pipeline'ın sıralı çıktısı, ORT satır indeksiyle eşlenir; `boundary_raw` katmanın ham sınırı.
LayerCompare compare_layer(const std::vector<Det>& py, const std::vector<Det>& cpp,
                           double boundary_raw);

double iou(double ax1, double ay1, double ax2, double ay2, double bx1, double by1, double bx2,
           double by2);

}  // namespace baykus::parity
