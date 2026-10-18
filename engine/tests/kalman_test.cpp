// İz Kalman'ı: trackers 2.6.1 `ByteTrackTracklet` adımlarıyla (`kalman_steps.json`: rastgele
// predict / update; her adımdan sonra durum ve kovaryans) ≤ 1e-6.
#include <gtest/gtest.h>

#include <array>
#include <string>

#include "baykus/kalman.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;

namespace {

std::array<double, 4> box(const nlohmann::json& j) {
  return {j.at(0).get<double>(), j.at(1).get<double>(), j.at(2).get<double>(),
          j.at(3).get<double>()};
}

}  // namespace

TEST(Kalman, InitialState) {
  const baykus::KalmanBox k({1.0, 2.0, 3.0, 4.0});
  EXPECT_EQ(k.state().head<4>(), Eigen::Vector4d(1, 2, 3, 4));
  EXPECT_TRUE(k.state().tail<4>().isZero());
  EXPECT_EQ(k.covariance(), baykus::KalmanBox::Cov::Identity());
  EXPECT_EQ(k.bbox(), (std::array<double, 4>{1, 2, 3, 4}));
}

TEST(Kalman, MatchesPythonSteps) {
  BAYKUS_REQUIRE_FILES({bt::fixtures_root() / "index.json"});
  const auto entries = bt::fixtures_of("kalman");
  ASSERT_EQ(entries.size(), 1u);
  const auto rec = bt::read_json(bt::fixtures_root() / entries[0].at("path").get<std::string>());
  baykus::KalmanBox k(box(rec.at("init")));
  const auto& steps = rec.at("steps");
  ASSERT_FALSE(steps.empty());
  int updates = 0;
  for (std::size_t s = 0; s < steps.size(); ++s) {
    SCOPED_TRACE("adım " + std::to_string(s));
    const auto& st = steps[s];
    if (st.at("op") == "update") {
      k.update(box(st.at("bbox")));
      ++updates;
    } else {
      k.predict();
    }
    for (int i = 0; i < 8; ++i) {
      EXPECT_NEAR(k.state()(i), st.at("state")[i].get<double>(), 1e-6) << "durum " << i;
      for (int j = 0; j < 8; ++j) {
        EXPECT_NEAR(k.covariance()(i, j), st.at("cov")[i][j].get<double>(), 1e-6)
            << "kovaryans " << i << "," << j;
      }
    }
  }
  EXPECT_GT(updates, 0);
  EXPECT_LT(updates, static_cast<int>(steps.size()));  // iki işlem de sınandı
}
