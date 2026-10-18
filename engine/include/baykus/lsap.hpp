#pragma once

// Dikdörtgen atama: SciPy `linear_sum_assignment(cost, maximize=True)` (aynı C++ kodu,
// third_party/lsap). Dönüş (satır, sütun) çiftleri satır sırasıyla; boş matriste boş.
// NaN ya da büyütmede +inf içeren matris: std::invalid_argument.
#include <Eigen/Core>
#include <utility>
#include <vector>

namespace baykus {

std::vector<std::pair<int, int>> lsap_maximize(const Eigen::MatrixXd& cost);

}  // namespace baykus
