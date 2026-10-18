#pragma once

// Sensör sağlığı (Faz 2 + Faz 3 kare kuralı): Python `baykus.health.features` ve
// `baykus.health.gate` ile aynı ölçüler, durumlar ve kapı kuralı. Hesaplanamayan değer boş optional
// (JSON'da null).
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <utility>

namespace baykus {

struct Features {
  std::optional<double> rgb_std, rgb_noise, rgb_blur, rgb_brightness;
  std::optional<double> ir_std, ir_noise, ir_blur;
  std::optional<double> misalign_px, misalign_response, misalign_dx, misalign_dy;
};

// RGB BGR uint8 (H, W, 3), IR gri uint8 (H, W). Kayma vektörü ve büyüklüğü dahil (vector=True).
Features frame_features(const cv::Mat& bgr, const cv::Mat& ir);

struct Thresholds {
  double rgb_lost, ir_lost, ir_noisy, misaligned, rgb_blurry, ir_blurry, dark;
  std::string sha256;
  static Thresholds load(const std::string& path);
};

struct GateModel {
  double r_min;
  int window;
  int min_frames;
  double tol_px;
  std::string thresholds_sha256;
  std::string sha256;
  static GateModel load(const std::string& path);
};

// (kapı, neden kodu): öncelik kayıp > gürültülü > hizasız; iki kanal birden kapanmaz.
std::pair<std::string, std::optional<std::string>> decide_gate(bool rgb_lost, bool ir_lost,
                                                               bool ir_noisy, bool misaligned);

// Türkçe neden metni (Python `REASONS`); kod yoksa "".
std::string reason_text(const std::optional<std::string>& code);

// Python `assess(f, th, r_min)` sözlüğü: rgb, ir, pair, gate, reason_code, reason.
nlohmann::json assess(const Features& f, const Thresholds& th, std::optional<double> r_min);

}  // namespace baykus
