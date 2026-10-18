#include "baykus/health.hpp"

#include <openssl/evp.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace baykus {

namespace {

std::string read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("okunamadı: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string sha256_hex(const std::string& data) {
  unsigned char d[EVP_MAX_MD_SIZE];
  unsigned int n = 0;
  if (EVP_Digest(data.data(), data.size(), d, &n, EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("sha256 hesaplanamadı");
  }
  std::ostringstream ss;
  for (unsigned int i = 0; i < n; ++i) ss << std::hex << std::setw(2) << std::setfill('0') << int(d[i]);
  return ss.str();
}

cv::Mat gray_of(const cv::Mat& bgr) {
  cv::Mat g;
  cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
  return g;
}

// cv2.meanStdDev(...)[1][0, 0]
double pixel_std(const cv::Mat& m) {
  cv::Scalar mean, std;
  cv::meanStdDev(m, mean, std);
  return std[0];
}

// Immerkær (1996): √(π/2) / (6 (W−2)(H−2)) · Σ |I ∗ M| (kenarlar hariç); filtre float32
double noise_sigma(const cv::Mat& gray) {
  const int h = gray.rows, w = gray.cols;
  if (h < 3 || w < 3) return 0.0;
  static const float kernel[9] = {1, -2, 1, -2, 4, -2, 1, -2, 1};
  const cv::Mat k(3, 3, CV_32F, const_cast<float*>(kernel));
  cv::Mat resp;
  cv::filter2D(gray, resp, CV_32F, k);
  const double l1 = cv::norm(resp(cv::Rect(1, 1, w - 2, h - 2)), cv::NORM_L1);
  return std::sqrt(M_PI / 2) * l1 / (6.0 * (w - 2) * (h - 2));
}

double blur_score(const cv::Mat& gray) {
  cv::Mat lap;
  cv::Laplacian(gray, lap, CV_32F);
  const double s = pixel_std(lap);
  return s * s;
}

double mean_brightness(const cv::Mat& bgr) {
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> ch;
  cv::split(hsv, ch);
  return cv::mean(ch[2])[0];
}

// Yarı çözünürlük (INTER_AREA) → float64 → Sobel x, y (CV_64F) → büyüklük
cv::Mat edges(const cv::Mat& gray) {
  cv::Mat small, f, gx, gy, mag;
  cv::resize(gray, small, cv::Size(gray.cols / 2, gray.rows / 2), 0, 0, cv::INTER_AREA);
  small.convertTo(f, CV_64F);
  cv::Sobel(f, gx, CV_64F, 1, 0);
  cv::Sobel(f, gy, CV_64F, 0, 1);
  cv::magnitude(gx, gy, mag);
  return mag;
}

bool below(const std::optional<double>& v, double t) { return v.has_value() && *v <= t; }
bool above(const std::optional<double>& v, double t) { return v.has_value() && *v >= t; }

nlohmann::json opt(const std::optional<double>& v) {
  return v.has_value() ? nlohmann::json(*v) : nlohmann::json(nullptr);
}

nlohmann::json camera(bool lost, bool noisy, bool dark, bool blurry) {
  // FLAG_ORDER: lost, noisy, dark, blurry; durum ilk bayrak ya da "ok"
  nlohmann::json flags = nlohmann::json::array();
  if (lost) flags.push_back("lost");
  if (noisy) flags.push_back("noisy");
  if (dark) flags.push_back("dark");
  if (blurry) flags.push_back("blurry");
  return {{"state", flags.empty() ? "ok" : flags[0].get<std::string>()}, {"flags", flags}};
}

}  // namespace

Features frame_features(const cv::Mat& bgr, const cv::Mat& ir) {
  Features f;
  const cv::Mat g = gray_of(bgr);
  f.rgb_std = pixel_std(g);
  f.rgb_noise = noise_sigma(g);
  f.rgb_blur = blur_score(g);
  f.rgb_brightness = mean_brightness(bgr);
  f.ir_std = pixel_std(ir);
  f.ir_noise = noise_sigma(ir);
  f.ir_blur = blur_score(ir);
  const cv::Mat a = edges(g), b = edges(ir);
  if (pixel_std(a) == 0 || pixel_std(b) == 0) return f;  // bir kamera sabit: kayma yok
  cv::Mat window;
  cv::createHanningWindow(window, a.size(), CV_64F);
  double response = 0.0;
  const cv::Point2d s = cv::phaseCorrelate(a, b, window, &response);
  const double dx = 2.0 * s.x, dy = 2.0 * s.y;  // yarı çözünürlük → tam çözünürlük
  if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(response)) return f;
  f.misalign_dx = dx;
  f.misalign_dy = dy;
  f.misalign_px = std::hypot(dx, dy);
  f.misalign_response = response;
  return f;
}

Thresholds Thresholds::load(const std::string& path) {
  const std::string bytes = read_bytes(path);
  const auto j = nlohmann::json::parse(bytes);
  const auto& s = j.at("states");
  auto t = [&](const char* name) { return s.at(name).at("threshold").get<double>(); };
  return Thresholds{t("rgb_lost"),   t("ir_lost"),   t("ir_noisy"), t("misaligned"),
                    t("rgb_blurry"), t("ir_blurry"), j.at("dark_brightness").get<double>(),
                    sha256_hex(bytes)};
}

GateModel GateModel::load(const std::string& path) {
  const std::string bytes = read_bytes(path);
  const auto j = nlohmann::json::parse(bytes);
  const auto& t = j.at("temporal");
  return GateModel{j.at("r_min").get<double>(),       t.at("window").get<int>(),
                   t.at("min_frames").get<int>(),     t.at("tol_px").get<double>(),
                   j.at("thresholds_sha256").get<std::string>(), sha256_hex(bytes)};
}

std::pair<std::string, std::optional<std::string>> decide_gate(bool rgb_lost, bool ir_lost,
                                                               bool ir_noisy, bool misaligned) {
  if (rgb_lost && ir_lost) return {"none", "both_lost"};
  if (rgb_lost) return {"rgb_off", "rgb_lost"};
  if (ir_lost) return {"ir_off", "ir_lost"};
  if (ir_noisy) return {"ir_off", "ir_noisy"};
  if (misaligned) return {"ir_off", "misaligned"};
  return {"none", std::nullopt};
}

std::string reason_text(const std::optional<std::string>& code) {
  if (!code) return "";
  if (*code == "rgb_lost") return "RGB kayıp";
  if (*code == "ir_lost") return "IR kayıp";
  if (*code == "ir_noisy") return "IR gürültülü";
  if (*code == "misaligned") return "IR hizasız";
  if (*code == "both_lost") return "iki kamera da kayıp";
  throw std::invalid_argument("bilinmeyen neden kodu: " + *code);
}

nlohmann::json assess(const Features& f, const Thresholds& th, std::optional<double> r_min) {
  const bool rgb_lost = below(f.rgb_std, th.rgb_lost);
  const bool ir_lost = below(f.ir_std, th.ir_lost);
  const bool ir_noisy = !ir_lost && above(f.ir_noise, th.ir_noisy);
  const bool misaligned = !(rgb_lost || ir_lost) && above(f.misalign_px, th.misaligned) &&
                          (!r_min.has_value() || above(f.misalign_response, *r_min));
  nlohmann::json rgb = camera(rgb_lost, false,
                              f.rgb_brightness.has_value() && *f.rgb_brightness < th.dark,
                              below(f.rgb_blur, th.rgb_blurry));
  rgb["std"] = opt(f.rgb_std);
  rgb["noise"] = opt(f.rgb_noise);
  rgb["blur"] = opt(f.rgb_blur);
  rgb["brightness"] = opt(f.rgb_brightness);
  nlohmann::json ir = camera(ir_lost, ir_noisy, false, below(f.ir_blur, th.ir_blurry));
  ir["std"] = opt(f.ir_std);
  ir["noise"] = opt(f.ir_noise);
  ir["blur"] = opt(f.ir_blur);
  const auto [gate, code] = decide_gate(rgb_lost, ir_lost, ir_noisy, misaligned);
  return {
      {"rgb", rgb},
      {"ir", ir},
      {"pair",
       {{"misalign_px", opt(f.misalign_px)},
        {"dx_px", opt(f.misalign_dx)},
        {"dy_px", opt(f.misalign_dy)},
        {"response", opt(f.misalign_response)},
        {"misaligned_frame", misaligned},
        {"misaligned", misaligned}}},
      {"gate", gate},
      {"reason_code", code ? nlohmann::json(*code) : nlohmann::json(nullptr)},
      {"reason", reason_text(code)},
  };
}

}  // namespace baykus
