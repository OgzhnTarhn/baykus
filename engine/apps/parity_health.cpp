// Kabul 3 kanıt dosyası: tek kare fikstürlerde sağlık (modül düzeyi), zaman kuralı birim
// senaryoları (b) ve klibin uçtan uca üç değişkeni (a) → results/engine/parity_health.json (Faz 3
// plan Görev 21). Kullanım: baykus_parity_health <çıktı.json>   (BAYKUS_FIXTURES, BAYKUS_MODELS,
// BAYKUS_CONFIGS)
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "../tests/fixtures.hpp"
#include "baykus/health.hpp"
#include "baykus/image_io.hpp"
#include "baykus/models.hpp"
#include "baykus/parity.hpp"
#include "baykus/pipeline.hpp"
#include "baykus/temporal.hpp"
#include "baykus/version.hpp"

namespace bt = baykus::test;
namespace bp = baykus::parity;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Totals {
  int frames = 0;
  int ok = 0;
  double max_rel = 0.0;
  double max_px = 0.0;
  double max_resp = 0.0;
  json errors = json::array();

  void add(const std::string& where, const bp::HealthCompare& c) {
    ++frames;
    ok += c.ok ? 1 : 0;
    max_rel = std::max(max_rel, c.max_rel);
    max_px = std::max(max_px, c.max_px);
    max_resp = std::max(max_resp, c.max_resp);
    for (const auto& e : c.errors) errors.push_back(where + ": " + e);
  }
  bool passed() const { return frames > 0 && ok == frames; }
  json to_json() const {
    return {{"frames", frames},   {"equal", ok},      {"passed", passed()},
            {"max_rel", max_rel}, {"max_px", max_px}, {"max_response", max_resp},
            {"errors", errors}};
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "kullanım: baykus_parity_health <çıktı.json>\n";
    return 2;
  }
  const auto root = bt::fixtures_root();
  const char* c = std::getenv("BAYKUS_CONFIGS");
  const auto manifest = fs::path(c && *c ? c : "/configs") / "engine" / "models.json";
  const auto models = baykus::load_models(bt::models_root().string(), manifest.string());

  // Tek kareler: kapı off, özgün pikseller, oturumsuz (`modules.health`).
  Totals single;
  for (const auto& e : bt::fixtures_of("frame")) {
    const auto exp = bt::read_json(root / e.at("path").get<std::string>());
    const auto& in = exp.at("inputs");
    const auto p = baykus::decode_pair(bt::read_file(root / in.at("rgb").get<std::string>()),
                                       bt::read_file(root / in.at("ir").get<std::string>()));
    const json got =
        baykus::assess(baykus::frame_features(p.bgr, p.ir), models.thresholds, models.gate.r_min);
    single.add(e.at("name").get<std::string>(),
               bp::compare_health(exp.at("modules").at("health"), got));
  }

  // (b) Zaman kuralı birim senaryoları: her karede misaligned, kapı, neden aynı.
  json scenarios = json::array();
  bool unit_ok = true;
  for (const auto& e : bt::fixtures_of("temporal_unit")) {
    for (const auto& sc : bt::read_json(root / e.at("path").get<std::string>())) {
      baykus::MisalignWindow w(models.gate.window, models.gate.min_frames, models.gate.tol_px);
      int frames = 0, equal = 0, alarms = 0;
      for (const auto& st : sc.at("steps")) {
        if (st.contains("reset")) {
          w.reset();
          continue;
        }
        const json got = w.update(st.at("input"));
        const auto& want = st.at("expected");
        ++frames;
        alarms += want.at("misaligned").get<bool>() ? 1 : 0;
        equal += got.at("pair").at("misaligned") == want.at("misaligned") &&
                 got.at("gate") == want.at("gate") &&
                 got.at("reason_code") == want.at("reason_code") &&
                 got.at("reason") == want.at("reason");
      }
      unit_ok = unit_ok && frames > 0 && equal == frames;
      scenarios.push_back({{"name", sc.at("name")},
                           {"frames", frames},
                           {"equal", equal},
                           {"misaligned_expected", alarms}});
    }
  }
  unit_ok = unit_ok && !scenarios.empty();

  // (a) Klip uçtan uca: kodlanmış baytlar, oturumlu, kapı auto.
  baykus::Pipeline pipe(models);
  json clips = json::object();
  bool clip_ok = true;
  int n_clips = 0;
  for (const auto& e : bt::fixtures_of("clip")) {
    const auto exp = bt::read_json(root / e.at("path").get<std::string>());
    baykus::MisalignWindow w(models.gate.window, models.gate.min_frames, models.gate.tol_px);
    Totals t;
    int flagged = 0;
    const auto& inputs = exp.at("inputs");
    const auto& frames = exp.at("frames");
    for (std::size_t i = 0; i < frames.size(); ++i) {
      const auto r = pipe.run(bt::read_file(root / inputs[i].at("rgb").get<std::string>()),
                              bt::read_file(root / inputs[i].at("ir").get<std::string>()),
                              exp.at("gate").get<std::string>(), 0.1f, &w);
      const auto& want = frames[i].at("health");
      flagged += want.at("pair").at("misaligned_frame").get<bool>() ? 1 : 0;
      t.add("kare " + std::to_string(frames[i].at("frame").get<int>()),
            bp::compare_health(want, r.health));
    }
    auto j = t.to_json();
    j["misaligned_frame_count"] = flagged;  // |S| > 0 olmayan klip alarm yolunu sınamaz
    clips[exp.at("variant").get<std::string>()] = j;
    clip_ok = clip_ok && t.passed();
    ++n_clips;
  }
  clip_ok = clip_ok && n_clips == 3;

  const bool all_ok = single.passed() && unit_ok && clip_ok;
  const json rec = {
      {"kabul", "3 (tek kare + zaman kuralı a uçtan uca, b birim)"},
      {"passed", all_ok},
      {"opencv_version", baykus::opencv_version()},
      {"onnxruntime_version", baykus::onnxruntime_version()},
      {"tolerances",
       {{"relative", bp::kHealthRel}, {"px", bp::kHealthPx}, {"response", bp::kHealthResp}}},
      {"temporal",
       {{"window", models.gate.window},
        {"min_frames", models.gate.min_frames},
        {"tol_px", models.gate.tol_px}}},
      {"single_frames", single.to_json()},
      {"temporal_unit", {{"passed", unit_ok}, {"scenarios", scenarios}}},
      {"clip", {{"passed", clip_ok}, {"gate", "auto"}, {"variants", clips}}},
  };
  std::ofstream(argv[1]) << rec.dump(2) << "\n";
  std::cout << argv[1] << ": tek kare " << single.ok << "/" << single.frames << ", birim "
            << scenarios.size() << " senaryo, klip " << n_clips << " değişken; Kabul 3 "
            << (all_ok ? "geçti" : "GEÇMEDİ") << "\n";
  return all_ok ? 0 : 1;
}
