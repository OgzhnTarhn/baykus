// baykus_clip: senaryo dosyasındaki adımları Runtime üzerinden oynatır (Faz 4 klip aracı).
// Kullanım: baykus_clip --models DIR --manifest M [--motion F] [--tracker F] [--tar T] --out F.json
#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "baykus/files.hpp"
#include "baykus/models.hpp"
#include "baykus/motion_config.hpp"
#include "baykus/runtime.hpp"
#include "baykus/sessions.hpp"

namespace {

int usage() {
  std::cerr << "kullanım: baykus_clip --models DIR --manifest M [--motion F] [--tracker F] "
               "[--tar T] --out F.json\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string models_dir, manifest, out;
  std::string motion_path = "/configs/engine/motion.json";
  std::string tracker_path = "/configs/engine/tracker.json";
  std::string tar_dir = "/fixtures/f4";
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const bool has_value = i + 1 < argc;
      if (a == "--models" && has_value) {
        models_dir = argv[++i];
      } else if (a == "--manifest" && has_value) {
        manifest = argv[++i];
      } else if (a == "--motion" && has_value) {
        motion_path = argv[++i];
      } else if (a == "--tracker" && has_value) {
        tracker_path = argv[++i];
      } else if (a == "--tar" && has_value) {
        tar_dir = argv[++i];
      } else if (a == "--out" && has_value) {
        out = argv[++i];
      } else {
        return usage();
      }
    }
    if (models_dir.empty() || manifest.empty() || out.empty()) return usage();

    const auto models = baykus::load_models(models_dir, manifest);
    const auto motion = baykus::MotionCfg::load(motion_path);
    const auto tracker_cfg = baykus::TrackerConfig::load(tracker_path);
    baykus::Sessions sessions(tracker_cfg, motion, models.gate.window, models.gate.min_frames,
                              models.gate.tol_px);
    baykus::Runtime runtime(models, tracker_cfg, sessions);

    const auto scenarios = nlohmann::json::parse(baykus::read_bytes(tar_dir + "/scenarios.json"));
    nlohmann::json out_json;
    out_json["gate"] = "advise";
    out_json["motion_sha256"] = motion.sha256;
    auto& out_scenarios = out_json["scenarios"];
    for (auto it = scenarios.begin(); it != scenarios.end(); ++it) {
      const std::string name = it.key();
      auto& records = out_scenarios[name];
      records = nlohmann::json::array();
      for (const auto& step : it.value()) {
        const nlohmann::json reset_before =
            step.contains("reset_before") ? step["reset_before"] : nlohmann::json(nullptr);
        if (reset_before.is_string()) {
          const auto scope = reset_before.get<std::string>();
          if (scope == "all") {
            sessions.reset(name, baykus::Scope::All);
          } else if (scope == "tracks") {
            sessions.reset(name, baykus::Scope::Tracks);
          }
        }
        baykus::Request req;
        req.rgb = baykus::read_bytes(tar_dir + "/" + step["rgb"].get<std::string>());
        req.ir = baykus::read_bytes(tar_dir + "/" + step["ir"].get<std::string>());
        req.gate = "advise";
        req.session = name;
        req.conf = 0.25;
        req.track = true;
        req.frame = step["frame"].get<int>();
        const auto resp = runtime.process(req);
        auto rec = baykus::response_json(resp);
        rec["frame"] = step["frame"];
        rec["reset_before"] = reset_before;
        records.push_back(std::move(rec));
      }
    }
    std::ofstream f(out);
    f << out_json.dump();
    std::cout << out << "\n";
  } catch (const std::exception& e) {
    std::cerr << "hata: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
