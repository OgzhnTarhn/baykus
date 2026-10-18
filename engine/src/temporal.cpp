#include "baykus/temporal.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include "baykus/health.hpp"

namespace baykus {

namespace {

using json = nlohmann::json;

bool has_flag(const json& cam, const char* flag) {
  const auto& flags = cam.at("flags");
  return std::find(flags.begin(), flags.end(), flag) != flags.end();
}

// numpy.median: çift sayıda ortadaki iki değerin ortalaması.
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

std::optional<double> number(const json& pair, const char* key) {
  const auto it = pair.find(key);
  if (it == pair.end() || it->is_null()) return std::nullopt;
  return it->get<double>();
}

}  // namespace

MisalignWindow::MisalignWindow(int window, int min_frames, double tol_px)
    : window_(window), min_frames_(min_frames), tol_px_(tol_px) {}

void MisalignWindow::reset() { buf_.clear(); }

json MisalignWindow::update(const json& h) {
  json out = h;
  auto& pair = out.at("pair");
  const bool flagged = pair.at("misaligned_frame").get<bool>();
  const auto dx = number(pair, "dx_px");
  const auto dy = number(pair, "dy_px");
  if (flagged && !(dx && dy)) {
    throw std::invalid_argument("işaretli karede kayma vektörü (dx_px, dy_px) yok");
  }
  buf_.push_back({flagged, dx.value_or(0.0), dy.value_or(0.0)});
  while (static_cast<int>(buf_.size()) > window_) buf_.pop_front();

  std::vector<double> xs, ys;
  for (const auto& e : buf_) {
    if (!e.flagged) continue;
    xs.push_back(e.dx);
    ys.push_back(e.dy);
  }
  int n_c = 0;
  if (!xs.empty()) {
    const double cx = median(xs);
    const double cy = median(ys);
    for (std::size_t i = 0; i < xs.size(); ++i) {
      // np.linalg.norm(axis=1) = sqrt(ex² + ey²) (hypot değil: sınırda aynı yuvarlama)
      const double ex = xs[i] - cx;
      const double ey = ys[i] - cy;
      if (std::sqrt(ex * ex + ey * ey) <= tol_px_) ++n_c;
    }
  }
  const bool rgb_lost = has_flag(out.at("rgb"), "lost");
  const bool ir_lost = has_flag(out.at("ir"), "lost");
  const bool ir_noisy = has_flag(out.at("ir"), "noisy");
  const bool misaligned = n_c >= min_frames_ && !(rgb_lost || ir_lost);
  pair["misaligned"] = misaligned;
  const auto [gate, code] = decide_gate(rgb_lost, ir_lost, ir_noisy, misaligned);
  out["gate"] = gate;
  out["reason_code"] = code ? json(*code) : json(nullptr);
  out["reason"] = reason_text(code);
  return out;
}

}  // namespace baykus
