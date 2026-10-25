// Gecikme ölçümü: yüzdelik (H5 kaydıyla aynı yöntem), girdi okuma ve 2 + 2 yinelemelik duman testi.
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "baykus/bench.hpp"
#include "baykus/models.hpp"
#include "baykus/motion_config.hpp"
#include "baykus/runtime.hpp"
#include "baykus/sessions.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
namespace fs = std::filesystem;

TEST(Bench, PercentileMatchesNumpy) {
  // numpy.percentile(range(1, 11), [50, 95]) = [5.5, 9.55]
  const std::vector<double> v = {10, 1, 9, 2, 8, 3, 7, 4, 6, 5};
  EXPECT_DOUBLE_EQ(baykus::percentile(v, 50), 5.5);
  EXPECT_NEAR(baykus::percentile(v, 95), 9.55, 1e-12);
  EXPECT_DOUBLE_EQ(baykus::percentile({4.0}, 95), 4.0);
  EXPECT_THROW(baykus::percentile({}, 50), std::invalid_argument);
  const auto s = baykus::summarize(v);
  EXPECT_DOUBLE_EQ(s.at("mean").get<double>(), 5.5);
  EXPECT_EQ(s.at("n"), 10);
}

TEST(Bench, LoadInputsPairsByStem) {
  const auto d = fs::temp_directory_path() / "baykus_bench_inputs";
  fs::remove_all(d);
  fs::create_directories(d / "rgb" / "images" / "val");
  fs::create_directories(d / "ir" / "images" / "val");
  std::ofstream(d / "images.json")
      << R"([{"stem": "b", "split": "val"}, {"stem": "a", "split": "val"}])";
  for (const char* s : {"a", "b"}) {
    std::ofstream(d / "rgb" / "images" / "val" / (std::string(s) + ".jpg")) << "rgb-" << s;
    std::ofstream(d / "ir" / "images" / "val" / (std::string(s) + ".png")) << "ir-" << s;
  }
  const auto frames = baykus::load_bench_inputs(d.string());
  ASSERT_EQ(frames.size(), 2u);
  EXPECT_EQ(frames[0].rgb, "rgb-b");  // images.json sırası
  EXPECT_EQ(frames[1].ir, "ir-a");
  fs::remove(d / "ir" / "images" / "val" / "a.png");
  EXPECT_THROW(baykus::load_bench_inputs(d.string()), std::runtime_error);
}

TEST(Bench, SmokeTwoIters) {
  const auto m = bt::models_root();
  const char* c = std::getenv("BAYKUS_CONFIGS");
  const auto configs = fs::path(c && *c ? c : "/configs") / "engine";
  const auto clip = bt::fixtures_root() / "m3ot" / "clip" / "sequences" / "clip";
  BAYKUS_REQUIRE_FILES({clip / "rgb" / "000001.jpg", m / "e3c.onnx", configs / "models.json"});
  const auto models = baykus::load_models(m.string(), (configs / "models.json").string());
  const baykus::TrackerConfig cfg;
  baykus::Sessions sessions(cfg, models.gate.window, models.gate.min_frames, models.gate.tol_px);
  baykus::Runtime runtime(models, cfg, sessions);
  std::vector<baykus::BenchFrame> frames;
  for (const char* f : {"000001", "000002", "000003"}) {
    frames.push_back({bt::read_file(clip / "rgb" / (std::string(f) + ".jpg")),
                      bt::read_file(clip / "ir" / (std::string(f) + ".png"))});
  }
  const auto r = baykus::bench_memory(runtime, frames, 2, 2, 0.25);
  for (const char* k : {"decode", "health", "pre", "infer", "post", "track", "total"}) {
    ASSERT_TRUE(r.at("stages").contains(k)) << k;
    EXPECT_EQ(r.at("stages").at(k).at("n"), 2) << k;
  }
  EXPECT_GT(r.at("fps").get<double>(), 0.0);
  // her yinelemede total ≥ infer → sıralı değerler ve yüzdelikler de
  EXPECT_GE(r.at("stages").at("total").at("p95").get<double>(),
            r.at("stages").at("infer").at("p95").get<double>());
}

TEST(Bench, SequenceInputsInOrder) {
  const auto d = fs::temp_directory_path() / "baykus_bench_sequence";
  fs::remove_all(d);
  fs::create_directories(d / "rgb");
  fs::create_directories(d / "ir");
  std::ofstream(d / "index.json") << R"({"frames": [5, 3, 9]})";
  for (const int f : {5, 3, 9}) {
    const auto stem = std::string(6 - std::to_string(f).size(), '0') + std::to_string(f);
    std::ofstream(d / "rgb" / (stem + ".jpg")) << "rgb-" << f;
    std::ofstream(d / "ir" / (stem + ".png")) << "ir-" << f;
  }
  const auto frames = baykus::load_sequence(d.string(), 2);
  ASSERT_EQ(frames.size(), 2u);
  EXPECT_EQ(frames[0].frame, 5);  // index.json sırası
  EXPECT_EQ(frames[0].rgb, "rgb-5");
  EXPECT_EQ(frames[0].ir, "ir-5");
  EXPECT_EQ(frames[1].frame, 3);
  EXPECT_EQ(frames[1].rgb, "rgb-3");
  EXPECT_EQ(frames[1].ir, "ir-3");
  EXPECT_THROW(baykus::load_sequence(d.string(), 4), std::runtime_error);
  fs::remove(d / "ir" / "000003.png");
  EXPECT_THROW(baykus::load_sequence(d.string(), 3), std::runtime_error);
  fs::remove_all(d);
}

TEST(Bench, NoEgoDisablesStage) {
  const auto m = bt::models_root();
  const char* c = std::getenv("BAYKUS_CONFIGS");
  const auto configs = fs::path(c && *c ? c : "/configs") / "engine";
  const auto clip = bt::fixtures_root() / "m3ot" / "clip" / "sequences" / "clip";
  BAYKUS_REQUIRE_FILES({clip / "rgb" / "000001.jpg", clip / "ir" / "000001.png", m / "e3c.onnx",
                        configs / "models.json", configs / "motion.json"});
  const auto models = baykus::load_models(m.string(), (configs / "models.json").string());
  const auto frames = baykus::load_sequence(clip.string(), 15);
  const baykus::TrackerConfig cfg;
  baykus::SequenceOptions opt;
  opt.warmup = 2;
  opt.iters = 4;
  {
    baykus::Sessions sessions(cfg, models.gate.window, models.gate.min_frames, models.gate.tol_px);
    baykus::Runtime runtime(models, cfg, sessions);
    const auto r = baykus::bench_sequence(runtime, sessions, frames, opt);
    EXPECT_DOUBLE_EQ(r.at("stages").at("ego").at("p95").get<double>(), 0.0);
    EXPECT_TRUE(r.at("keyframe_p95").is_null());
    EXPECT_TRUE(r.at("reloc_attempt_p95").is_null());
    EXPECT_EQ(r.at("frames_sent").get<std::vector<int>>(), (std::vector<int>{1, 2, 3, 4, 5, 6}));
  }
  {
    baykus::Sessions sessions(cfg, baykus::MotionCfg::load((configs / "motion.json").string()),
                              models.gate.window, models.gate.min_frames, models.gate.tol_px);
    baykus::Runtime runtime(models, cfg, sessions);
    const auto r = baykus::bench_sequence(runtime, sessions, frames, opt);
    EXPECT_GT(r.at("stages").at("ego").at("mean").get<double>(), 0.0);
  }
}

TEST(Bench, SeekEvery) {
  const auto m = bt::models_root();
  const char* c = std::getenv("BAYKUS_CONFIGS");
  const auto configs = fs::path(c && *c ? c : "/configs") / "engine";
  const auto clip = bt::fixtures_root() / "m3ot" / "clip" / "sequences" / "clip";
  BAYKUS_REQUIRE_FILES({clip / "rgb" / "000001.jpg", clip / "ir" / "000001.png", m / "e3c.onnx",
                        configs / "models.json", configs / "motion.json"});
  const auto models = baykus::load_models(m.string(), (configs / "models.json").string());
  const auto frames = baykus::load_sequence(clip.string(), 15);
  const baykus::TrackerConfig cfg;
  baykus::Sessions sessions(cfg, baykus::MotionCfg::load((configs / "motion.json").string()),
                            models.gate.window, models.gate.min_frames, models.gate.tol_px);
  baykus::Runtime runtime(models, cfg, sessions);
  baykus::SequenceOptions opt;
  opt.warmup = 0;
  opt.iters = 12;
  opt.seek_every = 5;
  const auto r = baykus::bench_sequence(runtime, sessions, frames, opt);
  EXPECT_EQ(r.at("seeks").get<int>(), 2);
  EXPECT_EQ(r.at("frames_sent").get<std::vector<int>>(),
            (std::vector<int>{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2}));
  EXPECT_TRUE(r.at("seek_p95").is_number());
}
