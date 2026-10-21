#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "baykus/ego.hpp"
#include "baykus/geometry.hpp"
#include "baykus/motion_config.hpp"
#include "baykus/reloc.hpp"
#include "fixtures.hpp"

namespace baykus {
namespace {

using json = nlohmann::json;

constexpr int kW = 640;
constexpr int kH = 512;

std::filesystem::path config_path() {
  const char* env = std::getenv("BAYKUS_CONFIGS");
  const std::filesystem::path root = env ? env : "/configs";
  return root / "engine" / "motion.json";
}

std::filesystem::path f4_dir() { return test::fixtures_root() / "f4"; }

std::filesystem::path ir_path(int n) {
  char name[32];
  std::snprintf(name, sizeof(name), "%06d.png", n);
  return f4_dir() / "clip" / "ir" / name;
}

std::filesystem::path mirror_path(int n) {
  char name[32];
  std::snprintf(name, sizeof(name), "%06d.png", n);
  return f4_dir() / "mirror" / "ir" / name;
}

cv::Mat read_ir(int n) { return cv::imread(ir_path(n).string(), cv::IMREAD_GRAYSCALE); }

cv::Mat read_mirror(int n) { return cv::imread(mirror_path(n).string(), cv::IMREAD_GRAYSCALE); }

MotionCfg load_cfg() { return MotionCfg::load(config_path().string()); }

cv::Matx23d mat_from_json(const json& j) {
  return cv::Matx23d(j.at(0).get<double>(), j.at(1).get<double>(), j.at(2).get<double>(),
                     j.at(3).get<double>(), j.at(4).get<double>(), j.at(5).get<double>());
}

Record make_record(int gen, std::optional<int> frame, int n, const std::string& source) {
  Record rec;
  rec.gen = gen;
  rec.m_key = identity();
  rec.frame = frame;
  rec.n = n;
  rec.source = source;
  rec.kps = std::vector<cv::Point2f>();
  rec.desc = cv::Mat(0, 32, CV_8U);
  return rec;
}

// --- Kabul 10 (a): büyük ötelemeler -----------------------------------------------------------

TEST(Reloc, LargeShifts) {
  const auto warps_path = f4_dir() / "warps.json";
  BAYKUS_REQUIRE_FILES({warps_path, ir_path(1)});
  const json doc = test::read_json(warps_path);
  ASSERT_TRUE(doc.contains("warps"));
  const MotionCfg cfg = load_cfg();

  int total = 0;
  int good = 0;
  for (const auto& w : doc.at("warps")) {
    if (w.value("kind", std::string()) != "large") continue;
    const int src_frame = w.at("src_frame").get<int>();
    const cv::Matx23d m = mat_from_json(w.at("matrix"));
    const cv::Mat base = read_ir(src_frame);
    ASSERT_FALSE(base.empty()) << "kaynak kare " << src_frame;
    const auto rel = w.at("path").get<std::string>();
    const cv::Mat warped = cv::imread((f4_dir() / rel).string(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(warped.empty()) << rel;

    Ego ego(cfg, new Relocalizer(cfg));
    ego.update(nullptr, "none", {}, 0);  // ilk kare: kaynak kare
    // Kaynak kareyi doğrudan besle.
    const cv::Mat base_half = half_gray(base);
    ego.update(&base_half, "ir", {}, 1);
    ego.break_continuity();
    const cv::Mat warped_half = half_gray(warped);
    const EgoResult r = ego.update(&warped_half, "ir", {}, 2);

    ++total;
    if (!r.relocalized || !r.merge.has_value() || r.merge->to_gen != 1) {
      continue;
    }
    const double err = corner_error(*r.frame_to_map, invert(m), kW, kH);
    if (err <= 1.0) {
      ++good;
    }
  }

  ASSERT_EQ(total, 30);
  EXPECT_GE(good, static_cast<int>(std::ceil(0.95 * total)));
}

// --- Kabul 10 (c): aynalanmış sahne -----------------------------------------------------------

TEST(Reloc, BlankFrameNoMerge) {
  BAYKUS_REQUIRE_FILES({ir_path(1)});
  const MotionCfg cfg = load_cfg();
  Relocalizer rl(cfg);
  const cv::Mat base = read_ir(1);
  rl.add_keyframe(half_gray(base), {}, 1, identity(), 1, 1, "ir");

  const cv::Mat zero = cv::Mat::zeros(kH / 2, kW / 2, CV_8U);
  const auto mg = rl.try_merge(zero, {}, "ir", 2, identity(), 2);
  EXPECT_FALSE(mg.has_value());
}

TEST(Reloc, OrbCanonicalOrder) {
  BAYKUS_REQUIRE_FILES({ir_path(36)});
  const MotionCfg cfg = load_cfg();
  const cv::Mat g = half_gray(read_ir(36));
  const auto [pts, desc] = orb_features(g, {}, cfg);
  ASSERT_FALSE(pts.empty());
  EXPECT_EQ(desc.rows, static_cast<int>(pts.size()));
  for (std::size_t i = 1; i < pts.size(); ++i) {
    const bool ok =
        pts[i - 1].y < pts[i].y || (pts[i - 1].y == pts[i].y && pts[i - 1].x <= pts[i].x);
    EXPECT_TRUE(ok) << "i=" << i;
  }
  const auto [pts2, desc2] = orb_features(g, {}, cfg);
  ASSERT_EQ(pts.size(), pts2.size());
  for (std::size_t i = 0; i < pts.size(); ++i) {
    EXPECT_FLOAT_EQ(pts[i].x, pts2[i].x);
    EXPECT_FLOAT_EQ(pts[i].y, pts2[i].y);
  }
  ASSERT_EQ(desc.rows, desc2.rows);
  ASSERT_EQ(desc.cols, desc2.cols);
  EXPECT_EQ(cv::norm(desc, desc2, cv::NORM_INF), 0.0);
}

TEST(Reloc, MirroredNeverMerges) {
  BAYKUS_REQUIRE_FILES({ir_path(1), mirror_path(21)});
  const MotionCfg cfg = load_cfg();
  Ego ego(cfg, new Relocalizer(cfg));

  for (int n = 1; n <= 20; ++n) {
    const cv::Mat g = half_gray(read_ir(n));
    ego.update(&g, "ir", {}, n);
  }
  ego.break_continuity();

  bool any_merge = false;
  EgoResult last;
  for (int n = 21; n <= 60; ++n) {
    const cv::Mat g = half_gray(read_mirror(n));
    last = ego.update(&g, "ir", {}, n);
    if (last.merge.has_value() || last.relocalized) {
      any_merge = true;
    }
  }
  EXPECT_FALSE(any_merge);
  EXPECT_GT(last.map_gen, 1);
  EXPECT_FALSE(last.provisional);
}

// --- kütüphane --------------------------------------------------------------------------------

TEST(Reloc, LibraryCapAndCandidates) {
  const MotionCfg cfg = load_cfg();
  Library lib(cfg);

  for (int i = 0; i < 305; ++i) {
    lib.add(make_record(1, i, i, "ir"));
  }
  lib.add(make_record(2, 100, 400, "ir"));   // dışlanan nesil
  lib.add(make_record(1, 100, 401, "rgb"));  // başka kanal

  EXPECT_EQ(lib.size(), static_cast<std::size_t>(cfg.reloc.library_max));
  int min_n = std::numeric_limits<int>::max();
  for (const auto& r : lib.records()) {
    min_n = std::min(min_n, r.n);
  }
  EXPECT_EQ(min_n, 7);  // en eski 7 kayıt çıktı (305 + 2 − 300)

  const auto got = lib.candidates("ir", 100, 2);
  ASSERT_EQ(got.size(), 5u);
  EXPECT_EQ(got[0].frame.value_or(-1), 100);
  EXPECT_EQ(got[1].frame.value_or(-1), 101);
  EXPECT_EQ(got[2].frame.value_or(-1), 99);
  EXPECT_EQ(got[3].frame.value_or(-1), 102);
  EXPECT_EQ(got[4].frame.value_or(-1), 98);

  EXPECT_TRUE(lib.candidates("ir", 304 + 301, 2).empty());
  const auto near = lib.candidates("ir", 304 + 299, 2);
  ASSERT_EQ(near.size(), 2u);
  EXPECT_EQ(near[0].frame.value_or(-1), 304);
  EXPECT_EQ(near[1].frame.value_or(-1), 303);

  // frame yokken en yeni kayıtlar.
  Library lib2(cfg);
  for (int i = 0; i < 10; ++i) {
    lib2.add(make_record(1, (i % 2) ? std::nullopt : std::optional<int>(i), i, "ir"));
  }
  lib2.add(make_record(3, std::nullopt, 50, "ir"));
  const auto recent = lib2.candidates("ir", std::nullopt, 3);
  ASSERT_EQ(recent.size(), 5u);
  EXPECT_EQ(recent[0].n, 9);
  EXPECT_EQ(recent[1].n, 8);
  EXPECT_EQ(recent[2].n, 7);
  EXPECT_EQ(recent[3].n, 6);
  EXPECT_EQ(recent[4].n, 5);
}

}  // namespace
}  // namespace baykus
