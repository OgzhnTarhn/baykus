// Kabul 3 a: zaman kuralı uçtan uca. Video klibinin üç değişkeni kodlanmış baytlardan, oturumlu
// (tek MisalignWindow), kapı `auto` ile pipeline'dan geçer; her karede sağlık sözlüğü Python'un
// `clip_<değişken>.json`'ındakiyle Kabul 3 toleranslarında, karar alanları birebir. Alarmlar bu
// alanların deterministik işlevi (alarm hesabı Python ↔ UI arasında sınanır).
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "baykus/models.hpp"
#include "baykus/parity.hpp"
#include "baykus/pipeline.hpp"
#include "baykus/temporal.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
namespace bp = baykus::parity;
namespace fs = std::filesystem;

namespace {

fs::path manifest_path() {
  const char* c = std::getenv("BAYKUS_CONFIGS");
  return fs::path(c && *c ? c : "/configs") / "engine" / "models.json";
}

}  // namespace

TEST(Clip, EndToEnd) {
  const auto m = bt::models_root();
  BAYKUS_REQUIRE_FILES({bt::fixtures_root() / "index.json", m / "e3c.onnx", m / "thresholds.json",
                        m / "calibration.json", m / "gate_model.json", manifest_path()});
  const auto models = baykus::load_models(m.string(), manifest_path().string());
  baykus::Pipeline pipe(models);
  const auto root = bt::fixtures_root();
  const auto clips = bt::fixtures_of("clip");
  ASSERT_EQ(clips.size(), 3u);
  for (const auto& c : clips) {
    const auto exp = bt::read_json(root / c.at("path").get<std::string>());
    SCOPED_TRACE(exp.at("variant").get<std::string>());
    baykus::MisalignWindow w(models.gate.window, models.gate.min_frames, models.gate.tol_px);
    const auto& inputs = exp.at("inputs");
    const auto& frames = exp.at("frames");
    ASSERT_EQ(inputs.size(), frames.size());
    ASSERT_FALSE(frames.empty());
    for (std::size_t i = 0; i < frames.size(); ++i) {
      SCOPED_TRACE("kare " + std::to_string(frames[i].at("frame").get<int>()));
      const auto r = pipe.run(bt::read_file(root / inputs[i].at("rgb").get<std::string>()),
                              bt::read_file(root / inputs[i].at("ir").get<std::string>()),
                              exp.at("gate").get<std::string>(), 0.1f, &w);
      const auto h = bp::compare_health(frames[i].at("health"), r.health);
      EXPECT_TRUE(h.ok) << (h.errors.empty() ? "" : h.errors.front());
    }
  }
}
