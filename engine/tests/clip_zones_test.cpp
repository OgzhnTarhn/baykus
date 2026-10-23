// Kabul 7 klip regresyonu (docs/design/events_rules.md "Klip regresyonu"): Release klibinin `clean`
// ve `long_gap` senaryoları sabit bölge kümesiyle (fixtures/motion/zones_clip.json) Runtime'dan
// geçer; günlük ve son sayaçlar fixtures/expected/zones_clip.json ile birebir. Aday (kare kare ego
// / iz / alarm dökümüyle) her koşuda BAYKUS_EVIDENCE_DIR'e yazılır; beklenen dosya yoksa test
// düşer.
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "baykus/files.hpp"
#include "baykus/models.hpp"
#include "baykus/motion_config.hpp"
#include "baykus/runtime.hpp"
#include "baykus/sessions.hpp"
#include "baykus/zones.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

fs::path configs() {
  const char* c = std::getenv("BAYKUS_CONFIGS");
  return fs::path(c && *c ? c : "/configs") / "engine";
}

// Bir senaryoyu yeni oturumda oynatır: {"events", "final_counts", "frames"}.
json run_scenario(baykus::Runtime& runtime, baykus::Sessions& sessions, const fs::path& f4,
                  const std::string& name, const json& steps, const json& zones_body) {
  bool created = false;
  sessions.get(name, &created)->motion->zones.set(baykus::parse_zone_defs(zones_body), 1);
  json frames = json::array();
  json last_counts = json::object();
  for (const auto& step : steps) {
    const json reset = step.value("reset_before", json(nullptr));
    if (reset == "all") sessions.reset(name, baykus::Scope::All);
    if (reset == "tracks") sessions.reset(name, baykus::Scope::Tracks);
    baykus::Request req;
    req.rgb = baykus::read_bytes((f4 / step.at("rgb").get<std::string>()).string());
    req.ir = baykus::read_bytes((f4 / step.at("ir").get<std::string>()).string());
    req.gate = "advise";
    req.session = name;
    req.conf = 0.25;
    req.track = true;
    req.frame = step.at("frame").get<int>();
    const auto j = baykus::response_json(runtime.process(req));
    json tracks = json::array();
    for (const auto& t : j.at("tracks")) {
      tracks.push_back({{"track_id", t.at("track_id")},
                        {"box", t.at("box")},
                        {"motion", t.at("motion")},
                        {"map_xy", t.at("map_xy")}});
    }
    const auto& e = j.at("ego");
    frames.push_back({{"frame", req.frame.value()},
                      {"ego",
                       {{"ok", e.at("ok")},
                        {"map_gen", e.at("map_gen")},
                        {"provisional", e.at("provisional")},
                        {"frame_to_map", e.at("frame_to_map")}}},
                      {"tracks", tracks},
                      {"alarms", j.at("alarms")}});
    last_counts = j.at("zone_counts");
  }
  const auto s = sessions.find(name);
  return {{"events", s->motion->log.since(0).at("events")},
          {"final_counts", last_counts},
          {"frames", frames}};
}

}  // namespace

TEST(ClipZones, Regression) {
  const auto m = bt::models_root();
  const auto f4 = bt::fixtures_root() / "f4";
  const auto zones_path = bt::fixtures_root() / "motion" / "zones_clip.json";
  const auto expected_path = bt::fixtures_root() / "expected" / "zones_clip.json";
  BAYKUS_REQUIRE_FILES({f4 / "scenarios.json", zones_path, m / "e3c.onnx",
                        configs() / "models.json", configs() / "tracker.json",
                        configs() / "motion.json"});
  const auto models = baykus::load_models(m.string(), (configs() / "models.json").string());
  const auto tracker_cfg = baykus::TrackerConfig::load((configs() / "tracker.json").string());
  const auto motion = baykus::MotionCfg::load((configs() / "motion.json").string());
  baykus::Sessions sessions(tracker_cfg, motion, models.gate.window, models.gate.min_frames,
                            models.gate.tol_px);
  baykus::Runtime runtime(models, tracker_cfg, sessions);

  const auto scenarios = bt::read_json(f4 / "scenarios.json");
  const auto zones_body = bt::read_json(zones_path);
  json got;
  for (const char* name : {"clean", "long_gap"}) {
    got[name] = run_scenario(runtime, sessions, f4, name, scenarios.at(name), zones_body);
  }

  // Aday her koşuda yazılır (kanıt klasörü verilmişse): beklenen yoksa ya da değişince gözden
  // geçirilir.
  const char* dir = std::getenv("BAYKUS_EVIDENCE_DIR");
  if (dir != nullptr && *dir != '\0') {
    std::ofstream(fs::path(dir) / "zones_clip_candidate.json") << got.dump(1) << "\n";
  }
  if (!fs::exists(expected_path)) {
    ADD_FAILURE() << "beklenen dosya yok: " << expected_path;
    return;
  }
  const auto expected = bt::read_json(expected_path);
  for (const char* name : {"clean", "long_gap"}) {
    SCOPED_TRACE(name);
    EXPECT_EQ(got.at(name).at("events"), expected.at(name).at("events"));
    EXPECT_EQ(got.at(name).at("final_counts"), expected.at(name).at("final_counts"));
  }
}
