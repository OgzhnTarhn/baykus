#include "baykus/lsap.hpp"

#include <rectangular_lsap.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace baykus {

std::vector<std::pair<int, int>> lsap_maximize(const Eigen::MatrixXd& cost) {
  const auto nr = static_cast<intptr_t>(cost.rows());
  const auto nc = static_cast<intptr_t>(cost.cols());
  if (nr == 0 || nc == 0) return {};
  // SciPy satır öncelikli C dizisi bekler; Eigen sütun öncelikli.
  std::vector<double> c(static_cast<std::size_t>(nr * nc));
  for (intptr_t i = 0; i < nr; ++i)
    for (intptr_t j = 0; j < nc; ++j) c[i * nc + j] = cost(i, j);
  const auto n = static_cast<std::size_t>(std::min(nr, nc));
  std::vector<int64_t> a(n), b(n);
  const int rc =
      solve_rectangular_linear_sum_assignment(nr, nc, c.data(), true, a.data(), b.data());
  if (rc == RECTANGULAR_LSAP_INVALID) {
    throw std::invalid_argument("atama matrisinde geçersiz değer (NaN ya da sonsuz)");
  }
  if (rc == RECTANGULAR_LSAP_INFEASIBLE) throw std::invalid_argument("atama matrisi çözülemez");
  std::vector<std::pair<int, int>> out(n);
  for (std::size_t k = 0; k < n; ++k) out[k] = {static_cast<int>(a[k]), static_cast<int>(b[k])};
  return out;
}

}  // namespace baykus
