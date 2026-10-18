#pragma once

#include <string>
#include <vector>

namespace baykus {

// İzotonik tablo (Faz 2, `calibration.json` "table"): blok i = (x[i−1], x[i]]; en küçüğün altı ilk
// blok, en büyüğün üstü son blok. Python `apply_isotonic` (searchsorted sol + kırpma), double.
class Calibration {
 public:
  Calibration(std::vector<double> x, std::vector<double> y);
  static Calibration load(const std::string& path);

  double apply(float raw) const;
  // Kalibre ≥ conf ⇔ ham > dönen değer (gösterim katmanının ham sınırı; Kabul 1 b).
  double display_boundary(double conf) const;

  const std::vector<double>& x() const { return x_; }
  const std::vector<double>& y() const { return y_; }

 private:
  std::vector<double> x_;
  std::vector<double> y_;
};

}  // namespace baykus
