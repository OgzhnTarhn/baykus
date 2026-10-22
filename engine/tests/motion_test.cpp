#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "baykus/geometry.hpp"
#include "baykus/motion.hpp"
#include "baykus/motion_config.hpp"

namespace {

using baykus::Box;
using baykus::EgoResult;
using baykus::Merge;
using baykus::MotionCfg;
using baykus::MotionState;
using baykus::TrackMotion;
using baykus::TrackObs;

const MotionCfg& cfg() {
  const char* env = std::getenv("BAYKUS_CONFIGS");
  static const MotionCfg c =
      MotionCfg::load(std::string(env && *env ? env : "/configs") + "/engine/motion.json");
  return c;
}

// Python `_ego`: `m` null ise kayıp kare.
EgoResult ego(const std::optional<cv::Matx23d>& m, int gen = 1, bool provisional = false,
              std::optional<Merge> merge = std::nullopt) {
  EgoResult r;
  r.ok = m.has_value();
  r.source = m.has_value() ? "ir" : "none";
  r.map_gen = gen;
  r.provisional = provisional;
  r.frame_to_map = m;
  r.inliers = std::nullopt;
  r.inlier_ratio = std::nullopt;
  r.keyframe = false;
  r.relinked = false;
  r.relocalized = merge.has_value();
  r.merge = merge;
  return r;
}

// Python `_box`.
Box box(double cx, double cy, double size = 10.0) {
  return Box{cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2};
}

const cv::Matx23d S2 = baykus::similarity(0, 0, 0, 2.0);
const cv::Matx23d I2 = baykus::similarity(0, 0, 0, 1.0);

// Python `_run`: kutu her karede `step` px sağa (kare koordinatında); `m` sabit harita dönüşümü.
std::vector<TrackMotion> run(MotionState& st, int n, double step, const cv::Matx23d& m = S2,
                             int tid = 1, double x0 = 50.0) {
  std::vector<TrackMotion> out;
  out.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    out.push_back(st.update(ego(m), {TrackObs{tid, box(x0 + step * i, 40)}}).at(tid));
  }
  return out;
}

}  // namespace

TEST(Motion, SpeedInMapUnits) {
  // Ölçek 2, 1 px/kare, kutu 10: harita 2 px/kare, s = 10·2 = 20 → v = 2·15/15/20 = 0,10.
  MotionState st(cfg().motion);
  const auto out = run(st, 20, 1.0);
  for (int i = 0; i < 15; ++i) {
    EXPECT_EQ(out[i].motion, "unknown");
    EXPECT_FALSE(out[i].speed_rel.has_value());
  }
  ASSERT_TRUE(out[15].speed_rel.has_value());
  EXPECT_NEAR(*out[15].speed_rel, 0.1, 1e-9);  // sınır dahil
  EXPECT_EQ(out[15].motion, "moving");
  ASSERT_TRUE(out[15].map_xy.has_value());
  EXPECT_NEAR(out[15].map_xy->x, 2 * (50 + 15), 1e-9);
  EXPECT_NEAR(out[15].map_xy->y, 2 * 40, 1e-9);
}

TEST(Motion, StationaryBoundaryInclusive) {
  // 0,3 px/kare, ölçek 1, kutu 10 → v = 0,03.
  MotionState st(cfg().motion);
  const auto out = run(st, 20, 0.3, I2);
  ASSERT_TRUE(out[15].speed_rel.has_value());
  EXPECT_NEAR(*out[15].speed_rel, 0.03, 1e-9);
  EXPECT_EQ(out[15].motion, "stationary");
}

TEST(Motion, HysteresisBandKeepsPrevious) {
  MotionState st(cfg().motion);
  const auto out = run(st, 20, 1.0);  // moving
  EXPECT_EQ(out.back().motion, "moving");
  // 0,5 px/kare → harita 1 px/kare, v = 0,05 (bant): karar sürer.
  double x = 50 + 19.0;
  std::vector<TrackMotion> band;
  for (int i = 0; i < 40; ++i) {
    x += 0.5;
    band.push_back(st.update(ego(S2), {TrackObs{1, box(x, 40)}}).at(1));
  }
  for (const auto& o : band) EXPECT_EQ(o.motion, "moving");
  ASSERT_TRUE(band.back().speed_rel.has_value());
  EXPECT_NEAR(*band.back().speed_rel, 0.05, 1e-9);  // pencere tamamen bantta
  // Yeni iz bantta başlarsa unknown.
  MotionState st2(cfg().motion);
  EXPECT_EQ(run(st2, 20, 0.5).back().motion, "unknown");
}

TEST(Motion, NewTrackUnknownFirstK) {
  MotionState st(cfg().motion);
  run(st, 20, 1.0, S2, 1);
  const auto out = run(st, 16, 1.0, S2, 2, 200);
  for (int i = 0; i < 15; ++i) EXPECT_EQ(out[i].motion, "unknown");
  EXPECT_EQ(out[15].motion, "moving");
}

TEST(Motion, HoldThenUnknown) {
  MotionState st(cfg().motion);
  run(st, 20, 1.0);
  std::vector<TrackMotion> held;
  for (int i = 0; i < 17; ++i) {
    held.push_back(st.update(ego(std::nullopt), {TrackObs{1, box(0, 0)}}).at(1));
  }
  for (const auto& o : held) {
    EXPECT_FALSE(o.speed_rel.has_value());
    EXPECT_FALSE(o.map_xy.has_value());
  }
  const int hold = static_cast<int>(cfg().motion.hold);
  for (int i = 0; i < hold; ++i) EXPECT_EQ(held[i].motion, "moving");
  EXPECT_EQ(held[hold].motion, "unknown");
  EXPECT_EQ(held[hold + 1].motion, "unknown");
}

TEST(Motion, EgoGapInWindowBlocksV) {
  MotionState st(cfg().motion);
  run(st, 20, 1.0);
  st.update(ego(std::nullopt), {TrackObs{1, box(0, 0)}});  // tek kayıp karesi (n = 21)
  std::vector<TrackMotion> out;
  for (int i = 0; i < 17; ++i) {
    out.push_back(st.update(ego(S2), {TrackObs{1, box(50 + 20 + i, 40)}}).at(1));
  }
  // n = 22..36: pencere kayıp kareyi içerir ya da t − K kaydı yok → v yok, karar tutulur.
  for (int i = 0; i < 15; ++i) EXPECT_FALSE(out[i].speed_rel.has_value());
  EXPECT_TRUE(out[15].speed_rel.has_value());                       // n = 37: t − K = 22
  for (int i = 0; i < 14; ++i) EXPECT_EQ(out[i].motion, "moving");  // son v n = 20; 20 + 15 = 35
  EXPECT_EQ(out[14].motion, "unknown");                             // n = 36: tutma bitti
  EXPECT_EQ(out[15].motion, "moving");
}

TEST(Motion, TrackAbsentFramesStillMeasured) {
  // İzin aradaki kareleri gerekmez: geçerli ve tam t − K karesinde kaydı yeter (ego penceresi ok).
  MotionState st(cfg().motion);
  run(st, 20, 1.0);
  for (int i = 0; i < 5; ++i) st.update(ego(S2), {});  // iz görünmüyor (n = 21..25)
  const auto o = st.update(ego(S2), {TrackObs{1, box(80, 40)}}).at(1);  // n = 26, t − K = 11
  EXPECT_EQ(o.motion, "moving");
  ASSERT_TRUE(o.speed_rel.has_value());
  EXPECT_NEAR(*o.speed_rel, 2.0 * 20.0 / 15.0 / 20.0, 1e-9);
}

TEST(Motion, NewGenerationResets) {
  MotionState st(cfg().motion);
  run(st, 20, 1.0);
  std::vector<TrackMotion> out;
  for (int i = 0; i < 16; ++i) {
    out.push_back(st.update(ego(S2, 2, true), {TrackObs{1, box(70 + i, 40)}}).at(1));
  }
  for (int i = 0; i < 15; ++i) EXPECT_EQ(out[i].motion, "unknown");
  EXPECT_EQ(out[15].motion, "moving");
  EXPECT_FALSE(out[0].speed_rel.has_value());
}

TEST(Motion, MergeCarriesHistory) {
  MotionState st(cfg().motion);
  const cv::Matx23d m_prov = S2;  // geçici nesil 2'de karenin dönüşümü
  const cv::Matx23d t = baykus::similarity(30, -12, 0.0, 0.5);  // geçici → hedef
  std::vector<TrackMotion> out;
  for (int i = 0; i < 20; ++i) {
    out.push_back(st.update(ego(m_prov, 2, true), {TrackObs{1, box(50 + i, 40)}}).at(1));
  }
  EXPECT_EQ(out.back().motion, "moving");
  ASSERT_TRUE(out.back().speed_rel.has_value());
  const double v_before = *out.back().speed_rel;
  const Merge mg{2, 1, t};
  const auto o =
      st.update(ego(baykus::compose(t, m_prov), 1, false, mg), {TrackObs{1, box(70, 40)}}).at(1);
  // Taşınan geçmişle v kesintisiz: c ve s aynı oranda ölçeklenir → v aynı.
  EXPECT_EQ(o.motion, "moving");
  ASSERT_TRUE(o.speed_rel.has_value());
  EXPECT_NEAR(*o.speed_rel, v_before, 1e-9);
  const cv::Point2d p = baykus::apply(baykus::compose(t, m_prov), cv::Point2d(70.0, 40.0));
  ASSERT_TRUE(o.map_xy.has_value());
  EXPECT_NEAR(o.map_xy->x, p.x, 1e-9);
  EXPECT_NEAR(o.map_xy->y, p.y, 1e-9);
  const auto nxt = st.update(ego(baykus::compose(t, m_prov), 1), {TrackObs{1, box(71, 40)}}).at(1);
  ASSERT_TRUE(nxt.speed_rel.has_value());
  EXPECT_NEAR(*nxt.speed_rel, v_before, 1e-9);
}

TEST(Motion, FrozenFrameHoldsMotion) {
  MotionState st(cfg().motion);
  run(st, 20, 1.0);
  const auto o = st.update(ego(std::nullopt), {TrackObs{1, box(70, 40)}}).at(1);  // donmuş kare
  EXPECT_EQ(o.motion, "moving");
  EXPECT_FALSE(o.speed_rel.has_value());
  EXPECT_FALSE(o.map_xy.has_value());
}

TEST(Motion, ResetClears) {
  MotionState st(cfg().motion);
  run(st, 20, 1.0);
  st.reset();
  EXPECT_EQ(run(st, 1, 1.0)[0].motion, "unknown");
}

TEST(Motion, UntrackedMergeFromOtherGenDoesNotReset) {
  // Birleşme geçici neslin ilk karesindeyse hareket modülü nesil 2'yi hiç görmez: geçmiş sürer.
  MotionState st(cfg().motion);
  run(st, 20, 1.0);
  const Merge mg{2, 1, baykus::invert(I2)};
  const auto o = st.update(ego(S2, 1, false, mg), {TrackObs{1, box(70, 40)}}).at(1);
  EXPECT_EQ(o.motion, "moving");
  ASSERT_TRUE(o.speed_rel.has_value());
  EXPECT_NEAR(*o.speed_rel, 0.1, 1e-9);
}
