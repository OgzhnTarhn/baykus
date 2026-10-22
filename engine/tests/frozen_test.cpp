#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "baykus/frozen.hpp"
#include "baykus/geometry.hpp"
#include "baykus/motion_config.hpp"
#include "fixtures.hpp"

namespace {

namespace fs = std::filesystem;
namespace bt = baykus::test;
using nlohmann::json;

fs::path clip_dir() { return bt::fixtures_root() / "f4" / "clip" / "ir"; }

baykus::MotionCfg load_cfg() {
  const char* c = std::getenv("BAYKUS_CONFIGS");
  return baykus::MotionCfg::load(
      (fs::path(c && *c ? c : "/configs") / "engine" / "motion.json").string());
}

cv::Mat read_half_gray(const fs::path& path) {
  return baykus::half_gray(cv::imread(path.string(), cv::IMREAD_GRAYSCALE));
}

// f4 klip: ardışık iki kare (000001, 000002).
struct Clip {
  cv::Mat a;
  cv::Mat b;
};

Clip load_clip() {
  const fs::path dir = clip_dir();
  return {read_half_gray(dir / "000001.png"), read_half_gray(dir / "000002.png")};
}

void expect_no_diff(const json& ch) {
  EXPECT_FALSE(ch["frozen"].get<bool>());
  EXPECT_TRUE(ch["diff_mad"].is_null());
  EXPECT_FALSE(ch["alarm"].get<bool>());
}

}  // namespace

TEST(Frozen, FirstFrameHasNoDiff) {
  const baykus::MotionCfg cfg = load_cfg();
  BAYKUS_REQUIRE_FILES({clip_dir() / "000001.png", clip_dir() / "000002.png"});
  const Clip clip = load_clip();
  baykus::FrozenDetector fd(cfg.frozen);
  const json out = fd.update(clip.a, clip.a, false, false);
  expect_no_diff(out["rgb"]);
  expect_no_diff(out["ir"]);
}

TEST(Frozen, SameFrameIsFrozen) {
  const baykus::MotionCfg cfg = load_cfg();
  BAYKUS_REQUIRE_FILES({clip_dir() / "000001.png", clip_dir() / "000002.png"});
  const Clip clip = load_clip();
  baykus::FrozenDetector fd(cfg.frozen);
  fd.update(clip.a, clip.a, false, false);
  const json out = fd.update(clip.b, clip.a, false, false);
  EXPECT_TRUE(out["ir"]["frozen"].get<bool>());
  EXPECT_DOUBLE_EQ(out["ir"]["diff_mad"].get<double>(), 0.0);
  EXPECT_FALSE(out["rgb"]["frozen"].get<bool>());
  EXPECT_GT(out["rgb"]["diff_mad"].get<double>(), 0.5);
}

TEST(Frozen, ConsecutiveValFramesNotFrozen) {
  const baykus::MotionCfg cfg = load_cfg();
  BAYKUS_REQUIRE_FILES({clip_dir() / "000001.png", clip_dir() / "000002.png"});
  const Clip clip = load_clip();
  baykus::FrozenDetector fd(cfg.frozen);
  for (int i = 0; i < 15; ++i) {
    const cv::Mat& img = (i % 2 == 0) ? clip.a : clip.b;
    const json out = fd.update(img, img, false, false);
    for (const char* ch : {"rgb", "ir"}) {
      EXPECT_FALSE(out[ch]["frozen"].get<bool>()) << ch << " frame " << i;
      EXPECT_FALSE(out[ch]["alarm"].get<bool>()) << ch << " frame " << i;
    }
  }
}

TEST(Frozen, ThresholdInclusive) {
  const baykus::MotionCfg cfg = load_cfg();
  baykus::FrozenDetector fd(cfg.frozen);
  const cv::Mat a(4, 4, CV_8U, cv::Scalar(100));
  cv::Mat b = a.clone();
  b(cv::Rect(0, 0, 4, 2)) += 1;  // ortalama mutlak fark 0,5
  fd.update(a, a, false, false);
  const json out = fd.update(b, b, false, false);
  EXPECT_DOUBLE_EQ(out["ir"]["diff_mad"].get<double>(), 0.5);
  EXPECT_TRUE(out["ir"]["frozen"].get<bool>());
}

TEST(Frozen, AlarmThreeOfFive) {
  const baykus::MotionCfg cfg = load_cfg();
  BAYKUS_REQUIRE_FILES({clip_dir() / "000001.png", clip_dir() / "000002.png"});
  const Clip clip = load_clip();
  baykus::FrozenDetector fd(cfg.frozen);
  // 3., 5. ve 6. çağrı donmuş (önceki kareyle aynı görüntü).
  const std::vector<bool> use_a = {true,  false, false, true, true, true,
                                   false, true,  false, true, false};
  const std::vector<bool> expected = {false, false, false, false, false, true,
                                      true,  false, false, false, false};
  for (std::size_t i = 0; i < use_a.size(); ++i) {
    const cv::Mat& img = use_a[i] ? clip.a : clip.b;
    const json out = fd.update(img, img, false, false);
    EXPECT_EQ(out["ir"]["alarm"].get<bool>(), expected[i]) << "call " << i;
  }
}

TEST(Frozen, LostIsNotFrozen) {
  const baykus::MotionCfg cfg = load_cfg();
  BAYKUS_REQUIRE_FILES({clip_dir() / "000001.png", clip_dir() / "000002.png"});
  const Clip clip = load_clip();
  baykus::FrozenDetector fd(cfg.frozen);
  const cv::Mat z = cv::Mat::zeros(clip.a.size(), clip.a.type());
  json last;
  for (int n = 0; n < 6; ++n) {
    const cv::Mat& rgb = (n % 2 == 0) ? clip.a : clip.b;
    last = fd.update(rgb, z, false, true);
    EXPECT_FALSE(last["ir"]["frozen"].get<bool>()) << "call " << n;
    EXPECT_FALSE(last["ir"]["alarm"].get<bool>()) << "call " << n;
  }
  EXPECT_DOUBLE_EQ(last["ir"]["diff_mad"].get<double>(), 0.0);
}

TEST(Frozen, MissingImageAndReset) {
  const baykus::MotionCfg cfg = load_cfg();
  BAYKUS_REQUIRE_FILES({clip_dir() / "000001.png", clip_dir() / "000002.png"});
  const Clip clip = load_clip();
  baykus::FrozenDetector fd(cfg.frozen);
  fd.update(clip.a, clip.a, false, false);
  const json out = fd.update(cv::Mat(), clip.a, true, false);
  expect_no_diff(out["rgb"]);
  fd.reset();
  EXPECT_TRUE(fd.update(clip.a, clip.a, false, false)["ir"]["diff_mad"].is_null());
}
