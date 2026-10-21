// Kabul 3: ego birim testleri (Python `test_motion_ego.py` karşılığı) + iz eşlik denetimi.
#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
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

namespace bt = baykus::test;
using json = nlohmann::json;

namespace {

std::filesystem::path config_path() {
  const char* env = std::getenv("BAYKUS_CONFIGS");
  const std::filesystem::path root = env ? env : "/configs";
  return root / "engine" / "motion.json";
}

std::filesystem::path ir_path(int n) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%06d.png", n);
  return bt::fixtures_root() / "f4" / "clip" / "ir" / buf;
}

cv::Mat load_half(int n) {
  const cv::Mat img = cv::imread(ir_path(n).string(), cv::IMREAD_GRAYSCALE);
  return baykus::half_gray(img);
}

cv::Matx23d mat_from_json(const json& j) {
  if (j.is_array() && j.size() == 6) {
    return cv::Matx23d(j[0].get<double>(), j[1].get<double>(), j[2].get<double>(),
                       j[3].get<double>(), j[4].get<double>(), j[5].get<double>());
  }
  if (j.is_array() && j.size() == 2) {
    return cv::Matx23d(j[0][0].get<double>(), j[0][1].get<double>(), j[0][2].get<double>(),
                       j[1][0].get<double>(), j[1][1].get<double>(), j[1][2].get<double>());
  }
  if (j.is_array() && j.size() == 3 && j[0].is_array() && j[0].size() == 3) {
    return cv::Matx23d(j[0][0].get<double>(), j[0][1].get<double>(), j[0][2].get<double>(),
                       j[1][0].get<double>(), j[1][1].get<double>(), j[1][2].get<double>());
  }
  throw std::invalid_argument("mat_from_json");
}

}  // namespace

// --- config -----------------------------------------------------------------------------------

TEST(MotionConfig, LoadsAndRejectsUnknownKeys) {
  const auto p = config_path();
  BAYKUS_REQUIRE_FILES({p});
  const auto cfg = baykus::MotionCfg::load(p.string());
  EXPECT_EQ(cfg.sha256.size(), 64u);
  for (const char c : cfg.sha256) {
    EXPECT_TRUE(std::isxdigit(static_cast<unsigned char>(c)));
  }

  std::ifstream in(p);
  ASSERT_TRUE(in.good());
  json raw;
  in >> raw;
  raw["motion"]["typo"] = 1;
  const auto tmp = std::filesystem::temp_directory_path() / "baykus_motion_extra.json";
  {
    std::ofstream out(tmp);
    out << raw.dump();
  }
  EXPECT_THROW(baykus::MotionCfg::load(tmp.string()), std::invalid_argument);
  std::filesystem::remove(tmp);
}

// --- geometry ---------------------------------------------------------------------------------

TEST(Geometry, HalfToFull) {
  using namespace baykus;
  const cv::Matx23d I = identity();
  const cv::Matx23d f = to_full(I);
  EXPECT_NEAR(f(0, 0), 1.0, 1e-12);
  EXPECT_NEAR(f(0, 1), 0.0, 1e-12);
  EXPECT_NEAR(f(0, 2), 0.0, 1e-12);
  EXPECT_NEAR(f(1, 0), 0.0, 1e-12);
  EXPECT_NEAR(f(1, 1), 1.0, 1e-12);
  EXPECT_NEAR(f(1, 2), 0.0, 1e-12);

  // Yarı öteleme (1, 0) → tam (2, 0) (0,5 ofset kuralıyla tutarlı).
  const cv::Matx23d half_t(1.0, 0.0, 1.0, 0.0, 1.0, 0.0);
  const cv::Matx23d full_t = to_full(half_t);
  EXPECT_NEAR(full_t(0, 2), 2.0, 1e-9);
  EXPECT_NEAR(full_t(1, 2), 0.0, 1e-9);

  const cv::Matx23d a = similarity(3.0, -2.0, 10.0, 1.1, 320.0, 256.0);
  const cv::Matx23d c = compose(a, invert(a));
  EXPECT_NEAR(c(0, 0), 1.0, 1e-9);
  EXPECT_NEAR(c(0, 1), 0.0, 1e-9);
  EXPECT_NEAR(c(0, 2), 0.0, 1e-9);
  EXPECT_NEAR(c(1, 0), 0.0, 1e-9);
  EXPECT_NEAR(c(1, 1), 1.0, 1e-9);
  EXPECT_NEAR(c(1, 2), 0.0, 1e-9);

  EXPECT_DOUBLE_EQ(corner_error(a, a, 640.0, 512.0), 0.0);
}

// --- ego --------------------------------------------------------------------------------------

TEST(Ego, FirstFrameIdentity) {
  const auto p = config_path();
  BAYKUS_REQUIRE_FILES({p, ir_path(1)});
  const auto cfg = baykus::MotionCfg::load(p.string());
  baykus::Ego ego(cfg);
  const cv::Mat g = load_half(1);
  const auto r = ego.update(&g, "ir", {}, 1);
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.keyframe);
  EXPECT_EQ(r.map_gen, 1);
  EXPECT_FALSE(r.provisional);
  ASSERT_TRUE(r.frame_to_map.has_value());
  EXPECT_NEAR((*r.frame_to_map)(0, 0), 1.0, 1e-9);
  EXPECT_NEAR((*r.frame_to_map)(0, 1), 0.0, 1e-9);
  EXPECT_NEAR((*r.frame_to_map)(0, 2), 0.0, 1e-9);
  EXPECT_NEAR((*r.frame_to_map)(1, 0), 0.0, 1e-9);
  EXPECT_NEAR((*r.frame_to_map)(1, 1), 1.0, 1e-9);
  EXPECT_NEAR((*r.frame_to_map)(1, 2), 0.0, 1e-9);
  EXPECT_FALSE(r.inliers.has_value());
}

TEST(Ego, SyntheticWarps) {
  const auto p = config_path();
  BAYKUS_REQUIRE_FILES({p});
  const auto cfg = baykus::MotionCfg::load(p.string());

  const auto f4 = bt::fixtures_root() / "f4";
  const auto warps_path = f4 / "warps.json";
  BAYKUS_REQUIRE_FILES({warps_path});
  const auto doc = bt::read_json(warps_path);
  ASSERT_TRUE(doc.contains("warps"));
  const auto& arr = doc.at("warps");

  int total = 0;
  int good = 0;
  for (const auto& w : arr) {
    if (w.value("kind", std::string()) != "small") continue;
    ++total;
    const int src_frame = w.at("src_frame").get<int>();
    const cv::Matx23d mat = mat_from_json(w.at("matrix"));
    const cv::Mat base = cv::imread(ir_path(src_frame).string(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(base.empty()) << "kaynak kare " << src_frame;
    const auto rel = w.at("path").get<std::string>();
    const cv::Mat warped = cv::imread((f4 / rel).string(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(warped.empty()) << rel;
    baykus::Ego ego(cfg);
    const cv::Mat g1 = baykus::half_gray(base);
    const cv::Mat g2 = baykus::half_gray(warped);
    ego.update(&g1, "ir", {}, 1);
    const auto r = ego.update(&g2, "ir", {}, 2);
    const double err =
        r.ok ? baykus::corner_error(*r.frame_to_map, baykus::invert(mat), 640.0, 512.0)
             : std::numeric_limits<double>::infinity();
    if (err <= 0.5) ++good;
  }
  EXPECT_EQ(total, 50);
  EXPECT_GE(good, 48);
}

TEST(Ego, BlankFrameNotOk) {
  const auto p = config_path();
  BAYKUS_REQUIRE_FILES({p, ir_path(1)});
  const auto cfg = baykus::MotionCfg::load(p.string());
  baykus::Ego ego(cfg);
  const cv::Mat z = cv::Mat::zeros(256, 320, CV_8UC1);
  const auto r0 = ego.update(&z, "ir", {}, 1);
  EXPECT_FALSE(r0.ok);
  EXPECT_FALSE(r0.frame_to_map.has_value());
  const cv::Mat g = load_half(1);
  EXPECT_TRUE(ego.update(&g, "ir", {}, 2).ok);
  for (int f = 3; f <= 6; ++f) {
    const auto r = ego.update(&z, "ir", {}, f);
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.frame_to_map.has_value());
  }
}

TEST(Ego, FailedFrameNeverKeyframe) {
  const auto p = config_path();
  BAYKUS_REQUIRE_FILES({p, ir_path(1)});
  const auto cfg = baykus::MotionCfg::load(p.string());
  const cv::Mat base = cv::imread(ir_path(1).string(), cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(base.empty());
  const cv::Mat g1 = baykus::half_gray(base);
  cv::Mat flipped;
  cv::flip(base, flipped, 1);
  const cv::Mat g2 = baykus::half_gray(flipped);
  baykus::Ego ego(cfg);
  const auto r1 = ego.update(&g1, "ir", {}, 1);
  EXPECT_TRUE(r1.keyframe);
  const auto r2 = ego.update(&g2, "ir", {}, 2);
  EXPECT_FALSE(r2.ok);
  EXPECT_FALSE(r2.keyframe);
}

TEST(Ego, ShortGapRelinks) {
  const auto p = config_path();
  BAYKUS_REQUIRE_FILES({p, ir_path(1), ir_path(10), ir_path(16)});
  const auto cfg = baykus::MotionCfg::load(p.string());
  baykus::Ego ego(cfg);
  for (int f = 1; f <= 10; ++f) {
    const cv::Mat g = load_half(f);
    EXPECT_TRUE(ego.update(&g, "ir", {}, f).ok) << "kare " << f;
  }
  for (int f = 11; f <= 15; ++f) {
    const auto r = ego.update(nullptr, "none", {}, f);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.map_gen, 1);
  }
  const cv::Mat g16 = load_half(16);
  const auto r = ego.update(&g16, "ir", {}, 16);
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.relinked);
  EXPECT_EQ(r.map_gen, 1);
}

TEST(Ego, LongGapOpensProvisional) {
  const auto p = config_path();
  BAYKUS_REQUIRE_FILES({p, ir_path(1), ir_path(10), ir_path(23)});
  const auto cfg = baykus::MotionCfg::load(p.string());
  baykus::Ego ego(cfg);
  for (int f = 1; f <= 10; ++f) {
    const cv::Mat g = load_half(f);
    EXPECT_TRUE(ego.update(&g, "ir", {}, f).ok) << "kare " << f;
  }
  for (int f = 11; f <= 22; ++f) {
    EXPECT_FALSE(ego.update(nullptr, "none", {}, f).ok);
  }
  const cv::Mat g23 = load_half(23);
  const auto r = ego.update(&g23, "ir", {}, 23);
  EXPECT_TRUE(r.ok);
  EXPECT_EQ(r.map_gen, 2);
  EXPECT_TRUE(r.provisional);
  EXPECT_TRUE(r.keyframe);
}

TEST(Ego, BreakInsideProvisionalClosesIt) {
  const auto p = config_path();
  BAYKUS_REQUIRE_FILES({p, ir_path(1), ir_path(10), ir_path(23), ir_path(24)});
  const auto cfg = baykus::MotionCfg::load(p.string());
  baykus::Ego ego(cfg);
  for (int f = 1; f <= 10; ++f) {
    const cv::Mat g = load_half(f);
    ego.update(&g, "ir", {}, f);
  }
  for (int f = 11; f <= 22; ++f) {
    ego.update(nullptr, "none", {}, f);
  }
  const cv::Mat g23 = load_half(23);
  const auto r23 = ego.update(&g23, "ir", {}, 23);
  ASSERT_EQ(r23.map_gen, 2);
  ASSERT_TRUE(r23.provisional);

  ego.break_continuity();
  const cv::Mat g24 = load_half(24);
  const auto r = ego.update(&g24, "ir", {}, 24);
  EXPECT_EQ(r.map_gen, 3);
  EXPECT_TRUE(r.provisional);
  const auto& gens = ego.generations();
  ASSERT_TRUE(gens.count(2) == 1u);
  EXPECT_FALSE(gens.at(2).provisional);
}

TEST(Ego, TraceParity) {
  const char* trace_env = std::getenv("BAYKUS_EGO_TRACE");
  if (trace_env == nullptr) {
    GTEST_SKIP() << "BAYKUS_EGO_TRACE yok";
  }
  const auto p = config_path();
  const std::filesystem::path trace_path(trace_env);
  BAYKUS_REQUIRE_FILES({p, trace_path});
  const auto cfg = baykus::MotionCfg::load(p.string());
  const auto trace = bt::read_json(trace_path);
  ASSERT_TRUE(trace.contains("scenarios"));
  const auto f4 = bt::fixtures_root() / "f4";

  bool decisions_equal = true;
  double max_inliers_diff = 0.0;
  double max_ratio_diff = 0.0;
  double max_corner_px = 0.0;
  std::map<std::pair<std::string, int>, std::pair<json, json>> results;

  auto cmp_map = [&](const json& got, const json& want, const std::string& what) {
    if (got.is_null() && want.is_null()) return;
    if (got.is_null() != want.is_null()) {
      decisions_equal = false;
      ADD_FAILURE() << what << ": null uyuşmazlığı";
      return;
    }
    const cv::Matx23d a = mat_from_json(got);
    const cv::Matx23d b = mat_from_json(want);
    const double e = baykus::corner_error(a, b, 640.0, 512.0);
    max_corner_px = std::max(max_corner_px, e);
    EXPECT_LE(e, 0.25) << what;
  };

  for (auto it = trace["scenarios"].begin(); it != trace["scenarios"].end(); ++it) {
    const std::string name = it.key();
    SCOPED_TRACE(name);
    baykus::Relocalizer reloc(cfg);
    baykus::Ego ego(cfg, &reloc);
    for (const auto& step : it.value()) {
      const auto rb = step.at("reset_before");
      if (!rb.is_null()) {
        const auto s = rb.get<std::string>();
        if (s == "all") {
          ego.reset();
        } else if (s == "tracks") {
          ego.break_continuity();
        }
      }
      const auto ir = step.at("ir");
      cv::Mat g;
      const cv::Mat* gp = nullptr;
      if (!ir.is_null()) {
        const auto path = f4 / ir.get<std::string>();
        const cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        g = baykus::half_gray(img);
        gp = &g;
      }
      std::vector<baykus::Box> boxes;
      for (const auto& b : step.at("boxes")) {
        boxes.push_back(baykus::Box{b[0].get<double>(), b[1].get<double>(), b[2].get<double>(),
                                    b[3].get<double>()});
      }
      const auto source = step.at("source").get<std::string>();
      const int frame = step.at("frame").get<int>();
      const auto r = ego.update(gp, source, boxes, frame);
      const json got = r.to_json();
      const json want = step.at("ego");

      EXPECT_EQ(got.at("ok").get<bool>(), want.at("ok").get<bool>());
      EXPECT_EQ(got.at("source").get<std::string>(), want.at("source").get<std::string>());
      EXPECT_EQ(got.at("map_gen").get<int>(), want.at("map_gen").get<int>());
      EXPECT_EQ(got.at("provisional").get<bool>(), want.at("provisional").get<bool>());
      EXPECT_EQ(got.at("keyframe").get<bool>(), want.at("keyframe").get<bool>());
      EXPECT_EQ(got.at("relinked").get<bool>(), want.at("relinked").get<bool>());
      EXPECT_EQ(got.at("relocalized").get<bool>(), want.at("relocalized").get<bool>());

      cmp_map(got.at("frame_to_map"), want.at("frame_to_map"), "frame_to_map");

      const auto& gm = got.at("merge");
      const auto& wm = want.at("merge");
      if (gm.is_null() != wm.is_null()) {
        decisions_equal = false;
        ADD_FAILURE() << "merge null uyuşmazlığı";
      } else if (!gm.is_null()) {
        EXPECT_EQ(gm.at("from").get<int>(), wm.at("from").get<int>());
        EXPECT_EQ(gm.at("to").get<int>(), wm.at("to").get<int>());
        cmp_map(gm.at("transform"), wm.at("transform"), "merge.transform");
      }

      const auto& gi = got.at("inliers");
      const auto& wi = want.at("inliers");
      if (gi.is_null() != wi.is_null()) {
        decisions_equal = false;
      } else if (!gi.is_null()) {
        max_inliers_diff = std::max(max_inliers_diff,
                                    static_cast<double>(std::abs(gi.get<int>() - wi.get<int>())));
      }
      const auto& gr = got.at("inlier_ratio");
      const auto& wr = want.at("inlier_ratio");
      if (gr.is_null() != wr.is_null()) {
        decisions_equal = false;
      } else if (!gr.is_null()) {
        max_ratio_diff = std::max(max_ratio_diff, std::abs(gr.get<double>() - wr.get<double>()));
      }

      results[{name, frame}] = {got, want};
    }
  }

  const char* ev = std::getenv("BAYKUS_EVIDENCE_DIR");
  if (ev != nullptr) {
    json out;
    out["decisions_equal"] = decisions_equal;
    out["max_inliers_diff"] = max_inliers_diff;
    out["max_ratio_diff"] = max_ratio_diff;
    out["max_corner_px"] = max_corner_px;
    json near = json::array();
    const auto margins = f4 / "margins.json";
    if (std::filesystem::exists(margins)) {
      const auto mj = bt::read_json(margins);
      if (mj.contains("near")) {
        for (const auto& n : mj["near"]) {
          const std::string sc = n.at("scenario").get<std::string>();
          const int fr = n.at("frame").get<int>();
          json entry;
          entry["scenario"] = sc;
          entry["frame"] = fr;
          entry["text"] = n.value("text", "");
          const auto key = std::make_pair(sc, fr);
          const auto found = results.find(key);
          if (found != results.end()) {
            entry["py"] = {{"inliers", found->second.second.at("inliers")},
                           {"inlier_ratio", found->second.second.at("inlier_ratio")}};
            entry["cpp"] = {{"inliers", found->second.first.at("inliers")},
                            {"inlier_ratio", found->second.first.at("inlier_ratio")}};
          } else {
            entry["py"] = nullptr;
            entry["cpp"] = nullptr;
          }
          near.push_back(entry);
        }
      }
    }
    out["near"] = near;
    const auto dir = std::filesystem::path(ev);
    std::filesystem::create_directories(dir);
    std::ofstream of(dir / "ego_trace_cpp.json");
    of << out.dump(2);
  }
}

TEST(Ego, Timing) {
  const auto p = config_path();
  BAYKUS_REQUIRE_FILES({p, ir_path(1), ir_path(60)});
  const auto cfg = baykus::MotionCfg::load(p.string());
  baykus::Relocalizer reloc(cfg);
  baykus::Ego ego(cfg, &reloc);
  std::vector<double> ordinary;
  std::vector<double> keyframe;
  std::vector<double> prov;
  for (int f = 1; f <= 60; ++f) {
    if (f == 31) ego.break_continuity();
    const cv::Mat g = load_half(f);
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = ego.update(&g, "ir", {}, f);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (r.provisional) {
      prov.push_back(ms);
    } else if (r.keyframe) {
      keyframe.push_back(ms);
    } else {
      ordinary.push_back(ms);
    }
  }
  auto stats = [](std::vector<double> v) {
    if (v.empty()) return std::pair<double, double>{0.0, 0.0};
    std::sort(v.begin(), v.end());
    return std::pair<double, double>{v[v.size() / 2], v.back()};
  };
  const auto so = stats(ordinary);
  const auto sk = stats(keyframe);
  const auto sp = stats(prov);
  std::cout << "ego timing ms: ordinary median=" << so.first << " max=" << so.second
            << "; keyframe median=" << sk.first << " max=" << sk.second
            << "; provisional median=" << sp.first << " max=" << sp.second << "\n";
  const char* ev = std::getenv("BAYKUS_EVIDENCE_DIR");
  if (ev != nullptr) {
    json out;
    out["ordinary"] = {{"median_ms", so.first}, {"max_ms", so.second}};
    out["keyframe"] = {{"median_ms", sk.first}, {"max_ms", sk.second}};
    out["provisional"] = {{"median_ms", sp.first}, {"max_ms", sp.second}};
    const auto dir = std::filesystem::path(ev);
    std::filesystem::create_directories(dir);
    std::ofstream of(dir / "ego_timing.json");
    of << out.dump(2);
  }
}
