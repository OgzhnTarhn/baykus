// Kabul 3 (tek kareler): `modules.health` (kapı off, özgün pikseller, r_min ile).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

#include "baykus/health.hpp"
#include "baykus/image_io.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
using json = nlohmann::json;

namespace {

constexpr double kRel = 1e-4;   // std / noise / blur / brightness göreli
constexpr double kPx = 0.5;     // misalign_px, dx_px, dy_px mutlak
constexpr double kResp = 0.02;  // response mutlak

void expect_rel(const json& want, const std::optional<double>& got, const char* name) {
  if (want.is_null()) {
    EXPECT_FALSE(got.has_value()) << name;
    return;
  }
  ASSERT_TRUE(got.has_value()) << name;
  const double w = want.get<double>();
  EXPECT_LE(std::fabs(w - *got), kRel * std::max(std::fabs(w), std::fabs(*got)) + 1e-12)
      << name << " py=" << w << " cpp=" << *got;
}

void expect_abs(const json& want, const std::optional<double>& got, double tol, const char* name) {
  if (want.is_null()) {
    EXPECT_FALSE(got.has_value()) << name;
    return;
  }
  ASSERT_TRUE(got.has_value()) << name;
  EXPECT_LE(std::fabs(want.get<double>() - *got), tol)
      << name << " py=" << want.get<double>() << " cpp=" << *got;
}

}  // namespace

class HealthParity : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto m = bt::models_root();
    BAYKUS_REQUIRE_FILES(
        {bt::fixtures_root() / "index.json", m / "thresholds.json", m / "gate_model.json"});
    th_ = baykus::Thresholds::load((m / "thresholds.json").string());
    gm_ = baykus::GateModel::load((m / "gate_model.json").string());
  }

  template <typename F>
  void each_frame(F f) {
    const auto root = bt::fixtures_root();
    for (const auto& e : bt::fixtures_of("frame")) {
      SCOPED_TRACE(e.at("name").get<std::string>());
      const auto exp = bt::read_json(root / e.at("path").get<std::string>());
      const auto& in = exp.at("inputs");
      const auto p = baykus::decode_pair(bt::read_file(root / in.at("rgb").get<std::string>()),
                                         bt::read_file(root / in.at("ir").get<std::string>()));
      f(baykus::frame_features(p.bgr, p.ir), exp.at("modules").at("health"));
    }
  }

  baykus::Thresholds th_{};
  baykus::GateModel gm_{};
};

TEST_F(HealthParity, FeatureParity) {
  each_frame([](const baykus::Features& f, const json& h) {
    expect_rel(h.at("rgb").at("std"), f.rgb_std, "rgb_std");
    expect_rel(h.at("rgb").at("noise"), f.rgb_noise, "rgb_noise");
    expect_rel(h.at("rgb").at("blur"), f.rgb_blur, "rgb_blur");
    expect_rel(h.at("rgb").at("brightness"), f.rgb_brightness, "rgb_brightness");
    expect_rel(h.at("ir").at("std"), f.ir_std, "ir_std");
    expect_rel(h.at("ir").at("noise"), f.ir_noise, "ir_noise");
    expect_rel(h.at("ir").at("blur"), f.ir_blur, "ir_blur");
    const auto& p = h.at("pair");
    expect_abs(p.at("misalign_px"), f.misalign_px, kPx, "misalign_px");
    expect_abs(p.at("dx_px"), f.misalign_dx, kPx, "dx_px");
    expect_abs(p.at("dy_px"), f.misalign_dy, kPx, "dy_px");
    expect_abs(p.at("response"), f.misalign_response, kResp, "response");
  });
}

TEST_F(HealthParity, StatesGateReasonEqual) {
  each_frame([this](const baykus::Features& f, const json& want) {
    const json got = baykus::assess(f, th_, gm_.r_min);
    for (const char* cam : {"rgb", "ir"}) {
      EXPECT_EQ(got.at(cam).at("state"), want.at(cam).at("state")) << cam;
      EXPECT_EQ(got.at(cam).at("flags"), want.at(cam).at("flags")) << cam;
    }
    EXPECT_EQ(got.at("pair").at("misaligned_frame"), want.at("pair").at("misaligned_frame"));
    EXPECT_EQ(got.at("pair").at("misaligned"), want.at("pair").at("misaligned"));
    EXPECT_EQ(got.at("gate"), want.at("gate"));
    EXPECT_EQ(got.at("reason_code"), want.at("reason_code"));
    EXPECT_EQ(got.at("reason"), want.at("reason"));
  });
}

namespace {

baykus::Thresholds test_thresholds() {
  baykus::Thresholds th{};
  th.rgb_lost = 2;
  th.ir_lost = 2;
  th.ir_noisy = 8;
  th.misaligned = 14;
  th.rgb_blurry = 50;
  th.ir_blurry = 50;
  th.dark = 48.66;
  return th;
}

baykus::Features ok_features() {
  baykus::Features f{};
  f.rgb_std = 40.0;
  f.rgb_noise = 3.0;
  f.rgb_blur = 300.0;
  f.rgb_brightness = 120.0;
  f.ir_std = 30.0;
  f.ir_noise = 3.0;
  f.ir_blur = 200.0;
  f.misalign_px = 3.0;
  f.misalign_response = 0.3;
  return f;
}

}  // namespace

TEST(Health, RMinNullIsPhase2) {
  auto f = ok_features();
  f.misalign_px = 30.0;
  f.misalign_response = 0.05;
  const auto th = test_thresholds();
  const json phase2 = baykus::assess(f, th, std::nullopt);
  EXPECT_EQ(phase2.at("gate"), "ir_off");
  EXPECT_EQ(phase2.at("reason"), "IR hizasız");
  const json guarded = baykus::assess(f, th, 0.2);
  EXPECT_EQ(guarded.at("gate"), "none");
  EXPECT_TRUE(guarded.at("reason_code").is_null());
  EXPECT_EQ(guarded.at("reason"), "");
}

TEST(Health, DecideGatePriority) {
  using baykus::decide_gate;
  EXPECT_EQ(decide_gate(true, true, false, false).first, "none");
  EXPECT_EQ(decide_gate(true, true, false, false).second.value(), "both_lost");
  EXPECT_EQ(decide_gate(true, false, true, true).second.value(), "rgb_lost");
  EXPECT_EQ(decide_gate(false, true, false, true).second.value(), "ir_lost");
  EXPECT_EQ(decide_gate(false, false, true, true).second.value(), "ir_noisy");
  EXPECT_EQ(decide_gate(false, false, false, true).second.value(), "misaligned");
  EXPECT_FALSE(decide_gate(false, false, false, false).second.has_value());
}

TEST(Health, LostCameraSkipsMisalignment) {
  auto f = ok_features();
  f.ir_std = 0.0;
  f.misalign_px = 30.0;
  const json h = baykus::assess(f, test_thresholds(), std::nullopt);
  EXPECT_EQ(h.at("pair").at("misaligned"), false);
  EXPECT_EQ(h.at("reason"), "IR kayıp");
  EXPECT_EQ(h.at("ir").at("state"), "lost");
}
