#include "baykus/calibration.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace baykus {

Calibration::Calibration(std::vector<double> x, std::vector<double> y)
    : x_(std::move(x)), y_(std::move(y)) {
  if (x_.empty() || x_.size() != y_.size()) {
    throw std::invalid_argument("kalibrasyon tablosu boş ya da x/y uzunlukları farklı");
  }
}

Calibration Calibration::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("okunamadı: " + path);
  }
  const auto j = nlohmann::json::parse(in);
  const auto& t = j.at("table");
  return Calibration(t.at("x").get<std::vector<double>>(), t.at("y").get<std::vector<double>>());
}

double Calibration::apply(float raw) const {
  // numpy.searchsorted(x, v, side="left") = ilk x[i] ≥ v; sonra [0, n−1]'e kırpma
  const auto it = std::lower_bound(x_.begin(), x_.end(), static_cast<double>(raw));
  const auto idx = std::min<std::size_t>(static_cast<std::size_t>(it - x_.begin()), x_.size() - 1);
  return y_[idx];
}

double Calibration::display_boundary(double conf) const {
  // Kalibre ≥ conf ⇔ blok indeksi ≥ i* (y azalmaz) ⇔ ham > x[i* − 1]
  const auto it = std::lower_bound(y_.begin(), y_.end(), conf);
  const auto i = static_cast<std::size_t>(it - y_.begin());
  if (i == 0) return -std::numeric_limits<double>::infinity();
  if (i >= y_.size()) return std::numeric_limits<double>::infinity();
  return x_[i - 1];
}

}  // namespace baykus
