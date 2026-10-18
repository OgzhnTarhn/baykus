// Atama çözücüsü: SciPy `linear_sum_assignment(maximize=True)` ile birebir (`lsap_cases.json`:
// 200 rastgele float32 IoU matrisi, yarısında eşit değerler).
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "baykus/lsap.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;

TEST(Lsap, MatchesScipyOnRandomAndTies) {
  BAYKUS_REQUIRE_FILES({bt::fixtures_root() / "index.json"});
  const auto entries = bt::fixtures_of("lsap");
  ASSERT_EQ(entries.size(), 1u);
  const auto rec = bt::read_json(bt::fixtures_root() / entries[0].at("path").get<std::string>());
  const auto& cases = rec.at("cases");
  ASSERT_EQ(cases.size(), 200u);
  for (std::size_t k = 0; k < cases.size(); ++k) {
    SCOPED_TRACE("vaka " + std::to_string(k));
    const auto& m = cases[k].at("iou");
    Eigen::MatrixXd cost(m.size(), m.at(0).size());
    for (Eigen::Index i = 0; i < cost.rows(); ++i)
      for (Eigen::Index j = 0; j < cost.cols(); ++j) cost(i, j) = m[i][j].get<double>();
    const auto got = baykus::lsap_maximize(cost);
    const auto& rows = cases[k].at("rows");
    const auto& cols = cases[k].at("cols");
    ASSERT_EQ(got.size(), rows.size());
    for (std::size_t t = 0; t < got.size(); ++t) {
      EXPECT_EQ(got[t].first, rows[t].get<int>());
      EXPECT_EQ(got[t].second, cols[t].get<int>());
    }
  }
}

TEST(Lsap, EmptyMatrix) {
  EXPECT_TRUE(baykus::lsap_maximize(Eigen::MatrixXd(0, 3)).empty());
  EXPECT_TRUE(baykus::lsap_maximize(Eigen::MatrixXd(2, 0)).empty());
}

TEST(Lsap, TallMatrixSortedByRow) {  // nr > nc: SciPy transpoze eder, sonucu satıra göre sıralar
  Eigen::MatrixXd c(3, 2);
  c << 0.1, 0.9, 0.8, 0.2, 0.5, 0.5;
  const auto got = baykus::lsap_maximize(c);
  ASSERT_EQ(got.size(), 2u);
  EXPECT_EQ(got[0], std::make_pair(0, 1));
  EXPECT_EQ(got[1], std::make_pair(1, 0));
}

TEST(Lsap, NanThrows) {
  Eigen::MatrixXd c(1, 2);
  c << 0.5, std::nan("");
  EXPECT_THROW(baykus::lsap_maximize(c), std::invalid_argument);
}
