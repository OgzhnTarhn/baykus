#include <gtest/gtest.h>

#include <vector>

#include "baykus/calibration.hpp"
#include "baykus/detections.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;

// Kabul 1 c: tablonun her x'inin float32 karşılığı ve iki float32 komşusu → birebir (double).
TEST(Calibration, ExactAtProbes) {
  const auto root = bt::fixtures_root();
  const auto models = bt::models_root();
  BAYKUS_REQUIRE_FILES({root / "expected/calibration_probe.json", models / "calibration.json"});
  const auto cal = baykus::Calibration::load((models / "calibration.json").string());
  const auto probes = bt::read_json(root / "expected/calibration_probe.json");
  ASSERT_EQ(probes.size(), cal.x().size());
  int n = 0;
  for (const auto& p : probes) {
    for (const auto& q : p.at("probes")) {
      const float raw = static_cast<float>(q.at("score").get<double>());
      ASSERT_EQ(static_cast<double>(raw), q.at("score").get<double>());  // float32 tam temsil
      EXPECT_EQ(cal.apply(raw), q.at("calibrated").get<double>());
      ++n;
    }
  }
  EXPECT_EQ(n, 3 * static_cast<int>(cal.x().size()));
}

TEST(Calibration, ClipsOutsideTable) {
  baykus::Calibration cal({0.1, 0.25, 0.3, 1.0}, {0.0, 0.2, 0.3, 0.9});
  EXPECT_EQ(cal.apply(0.0f), 0.0);
  EXPECT_EQ(cal.apply(2.0f), 0.9);
  EXPECT_EQ(cal.apply(0.26f), 0.3);
  EXPECT_EQ(cal.display_boundary(0.25), 0.25);  // kalibre ≥ 0,25 ⇔ ham > 0,25
}

TEST(Order, TiesByRawThenIndex) {
  using baykus::Det;
  std::vector<Det> v{Det{5, 0, 0.5f, 0.3, 0, 0, 1, 1}, Det{1, 0, 0.6f, 0.3, 0, 0, 1, 1},
                     Det{2, 0, 0.5f, 0.3, 0, 0, 1, 1}, Det{0, 0, 0.9f, 0.2, 0, 0, 1, 1}};
  baykus::order(v);
  ASSERT_EQ(v.size(), 4u);
  EXPECT_EQ(v[0].index, 1);
  EXPECT_EQ(v[1].index, 2);
  EXPECT_EQ(v[2].index, 5);
  EXPECT_EQ(v[3].index, 0);
}
