// Bir dizi klasöründen takipli çıktı (Faz 3 tasarımı §4.1 `baykus_sequence`): Python
// `baykus-engine sequence` ile aynı biçim (.json.gz), `Runtime` üzerinden, oturum "seq".
// Kullanım: baykus_sequence --models DIR --manifest M [--motion F] [--tracker F]
//                           --seq-dir D --out F.json.gz [--i-know-this-is-test]
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "baykus/files.hpp"
#include "baykus/models.hpp"
#include "baykus/motion.hpp"
#include "baykus/runtime.hpp"
#include "baykus/sessions.hpp"
#include "baykus/tracker.hpp"

namespace {

int usage() {
  std::cerr << "kullanım: baykus_sequence --models DIR --manifest M [--motion F] [--tracker F] "
               "--seq-dir D --out F.json.gz [--i-know-this-is-test]\n";
  return 2;
}

std::string read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("dosya okunamadı: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::string models_dir, manifest, motion = "/configs/engine/motion.json";
  std::string tracker = "/configs/engine/tracker.json";
  std::string seq_dir, out;
  bool allow_test = false;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const bool has_value = i + 1 < argc;
      if (a == "--models" && has_value) {
        models_dir = argv[++i];
      } else if (a == "--manifest" && has_value) {
        manifest = argv[++i];
      } else if (a == "--motion" && has_value) {
        motion = argv[++i];
      } else if (a == "--tracker" && has_value) {
        tracker = argv[++i];
      } else if (a == "--seq-dir" && has_value) {
        seq_dir = argv[++i];
      } else if (a == "--out" && has_value) {
        out = argv[++i];
      } else if (a == "--i-know-this-is-test") {
        allow_test = true;
      } else {
        return usage();
      }
    }
    if (models_dir.empty() || manifest.empty() || seq_dir.empty() || out.empty()) return usage();

    nlohmann::json index;
    {
      std::ifstream in(seq_dir + "/index.json");
      if (!in) throw std::runtime_error("index.json okunamadı: " + seq_dir);
      in >> index;
    }
    const std::string split = index.value("split", "");
    if (split == "test" && !allow_test) {
      throw std::invalid_argument("test dizisi: --i-know-this-is-test gerekli");
    }

    const auto models = baykus::load_models(models_dir, manifest);
    const baykus::TrackerConfig tracker_cfg = baykus::TrackerConfig::load(tracker);
    const baykus::MotionCfg motion_cfg = baykus::MotionCfg::load(motion);
    baykus::Sessions sessions(tracker_cfg, motion_cfg, models.gate.window, models.gate.min_frames,
                              models.gate.tol_px);
    baykus::Runtime runtime(models, tracker_cfg, sessions);

    nlohmann::json frames = nlohmann::json::array();
    for (const auto& f : index.at("frames")) {
      const int frame = f.is_number_integer() ? f.get<int>() : f.at("frame").get<int>();
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%06d", frame);
      const std::string stem = buf;

      baykus::Request req;
      req.rgb = read_bytes(seq_dir + "/rgb/" + stem + ".jpg");
      req.ir = read_bytes(seq_dir + "/ir/" + stem + ".png");
      req.gate = "advise";
      req.session = "seq";
      req.conf = 0.25;
      req.track = true;
      req.frame = frame;

      const baykus::Response r = runtime.process(req);

      const nlohmann::json rj = baykus::response_json(r);
      nlohmann::json tracks = nlohmann::json::array();
      for (const auto& t : rj["tracks"]) {
        nlohmann::json o;
        o["track_id"] = t["track_id"];
        o["box"] = t["box"];
        o["motion"] = t["motion"];
        o["speed_rel"] = t["speed_rel"];
        o["map_xy"] = t["map_xy"];
        tracks.push_back(o);
      }

      nlohmann::json fo;
      fo["frame"] = frame;
      fo["ego"] = r.ego ? r.ego->to_json() : nlohmann::json();
      fo["tracks"] = tracks;
      frames.push_back(fo);
    }

    nlohmann::json rec;
    rec["seq"] = index.value("name", "");
    rec["split"] = split;
    rec["motion_sha256"] = motion_cfg.sha256;
    rec["frames"] = frames;
    baykus::write_gzip(out, rec.dump());
    std::cout << out << " (" << frames.size() << " kare)\n";
  } catch (const std::exception& e) {
    std::cerr << "hata: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
