#pragma once

#include <vector>

#include "baykus/preprocess.hpp"

namespace baykus {

// Python `baykus.engine_ref.pipeline.Det`: kutu görüntü pikseli (double), ham skor float32,
// kalibre skor double.
struct Det {
  int index;  // ORT çıktısındaki satır
  int cls_id;
  float score_raw;
  double score;
  double x1, y1, x2, y2;
};

constexpr int kRows = 300;
constexpr int kCols = 6;

// `[1, 300, 6]` → ham ≥ floor satırlar (satır sırasıyla). Geri ölçekleme, kırpma, w/h ≤ 0 atma
// Python `decode_rows` ile aynı (double). `score` = `score_raw` (kalibrasyon ayrı).
std::vector<Det> decode_rows(const std::vector<float>& out, const Letterbox& lb, float floor);

// Kalibre azalan, ham azalan, satır indeksi artan.
void order(std::vector<Det>& dets);

// Gösterim katmanı: kalibre ≥ conf (sıra korunur).
std::vector<Det> filter_display(const std::vector<Det>& dets, double conf);

}  // namespace baykus
