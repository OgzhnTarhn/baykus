// Kabul 1 b (pipeline düzeyi): `pipeline` bloğu (kapı auto, oturumsuz): kapı kararı, uygulandı mı,
// letterbox sha256 ve üç katmanın pipeline'ın kendi sıralı çıktısından karşılaştırılması.
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "baykus/models.hpp"
#include "baykus/parity.hpp"
#include "baykus/pipeline.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
namespace bp = baykus::parity;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

fs::path manifest_path() {
  const char* c = std::getenv("BAYKUS_CONFIGS");
  return fs::path(c && *c ? c : "/configs") / "engine" / "models.json";
}

struct Input {
  std::string name;
  json expected;
  std::string rgb;
  std::string ir;
};

std::vector<Input> frame_inputs() {
  const auto root = bt::fixtures_root();
  std::vector<Input> out;
  for (const auto& e : bt::fixtures_of("frame")) {
    auto exp = bt::read_json(root / e.at("path").get<std::string>());
    const auto& in = exp.at("inputs");
    out.push_back({e.at("name").get<std::string>(), exp,
                   bt::read_file(root / in.at("rgb").get<std::string>()),
                   bt::read_file(root / in.at("ir").get<std::string>())});
  }
  return out;
}

}  // namespace

class PipelineParity : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto m = bt::models_root();
    BAYKUS_REQUIRE_FILES({bt::fixtures_root() / "index.json", m / "e3c.onnx", m / "thresholds.json",
                          m / "calibration.json", m / "gate_model.json", manifest_path()});
  }
  baykus::Models models() const {
    return baykus::load_models(bt::models_root().string(), manifest_path().string());
  }
};

TEST_F(PipelineParity, FrameParity) {
  const auto m = models();
  baykus::Pipeline pipe(m, /*hash_tensor=*/true);
  const double display_boundary = m.calibration.display_boundary(0.25);
  const auto inputs = frame_inputs();
  ASSERT_FALSE(inputs.empty());
  for (const auto& f : inputs) {
    SCOPED_TRACE(f.name);
    const auto r = pipe.run(f.rgb, f.ir, "auto", 0.001f, nullptr);
    const auto& want = f.expected.at("pipeline");
    EXPECT_EQ(r.health.at("gate"), want.at("gate"));
    EXPECT_EQ(r.health.at("applied"), want.at("applied"));
    EXPECT_EQ(r.tensor_sha256, want.at("letterbox_sha256").get<std::string>());
    std::vector<baykus::Det> track;
    for (const auto& d : r.dets_all) {
      if (d.score_raw >= 0.1f) track.push_back(d);
    }
    const auto display = baykus::filter_display(track, 0.25);
    const struct {
      const char* name;
      const std::vector<baykus::Det>& got;
      double boundary;
    } layers[] = {
        {"eval", r.dets_all, 0.001}, {"track", track, 0.1}, {"display", display, display_boundary}};
    for (const auto& l : layers) {
      const auto c =
          bp::compare_layer(bt::dets_from_json(want.at("dets").at(l.name)), l.got, l.boundary);
      EXPECT_TRUE(c.ok) << l.name << ": " << (c.errors.empty() ? "" : c.errors.front());
    }
  }
}

TEST_F(PipelineParity, TimingStagesAndNoHashByDefault) {
  baykus::Pipeline pipe(models());
  const auto f = frame_inputs().at(0);
  const auto r = pipe.run(f.rgb, f.ir, "advise", 0.1f, nullptr);
  for (const char* k : {"decode", "health", "pre", "infer", "post"}) {
    ASSERT_TRUE(r.timing_ms.count(k)) << k;
    EXPECT_GE(r.timing_ms.at(k), 0.0) << k;
  }
  EXPECT_GT(r.timing_ms.at("infer"), 0.0);
  EXPECT_TRUE(r.tensor_sha256.empty());
}

TEST_F(PipelineParity, GateOffMeasuresButDecidesNone) {
  baykus::Pipeline pipe(models());
  for (const auto& f : frame_inputs()) {
    if (f.name != "black_ir") continue;
    const auto& mod = f.expected.at("modules").at("health");
    ASSERT_NE(mod.at("gate"), "none");  // fikstür kanalı kapatan bir kare olmalı
    const auto r = pipe.run(f.rgb, f.ir, "off", 0.1f, nullptr);
    EXPECT_EQ(r.health.at("gate"), "none");
    EXPECT_EQ(r.health.at("applied"), false);
    EXPECT_EQ(r.health.at("reason_code"), mod.at("reason_code"));  // ölçüldüğü gibi kalır
    return;
  }
  FAIL() << "black_ir fikstürü yok";
}

TEST_F(PipelineParity, UnknownGateModeThrows) {
  baykus::Pipeline pipe(models());
  const auto f = frame_inputs().at(0);
  EXPECT_THROW(pipe.run(f.rgb, f.ir, "bogus", 0.1f, nullptr), std::invalid_argument);
}
