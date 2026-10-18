// Bir dizi klasöründen tespit dosyası (Faz 3 tasarımı §4.1 `baykus_detect`; plan Görev 24): Python
// `baykus-engine detect` ile aynı biçim (.json.gz), `Runtime` üzerinden, takip kapalı. M3OT
// ölçümü (Görev 11 `measure`) ve takip girdisi bunu okur.
// Kullanım: baykus_detect --models DIR --manifest M --seq-dir D --out F.json.gz [--floor 0.001]
//                         [--gate off|advise|auto] [--i-know-this-is-test]
#include <exception>
#include <iostream>
#include <string>

#include "baykus/files.hpp"
#include "baykus/models.hpp"
#include "baykus/runtime.hpp"
#include "baykus/sessions.hpp"

namespace {

int usage() {
  std::cerr << "kullanım: baykus_detect --models DIR --manifest M --seq-dir D --out F.json.gz "
               "[--floor 0.001] [--gate advise] [--i-know-this-is-test]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string models_dir, manifest, seq_dir, out, gate = "advise";
  double floor = 0.001;
  bool allow_test = false;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const bool has_value = i + 1 < argc;
      if (a == "--models" && has_value) {
        models_dir = argv[++i];
      } else if (a == "--manifest" && has_value) {
        manifest = argv[++i];
      } else if (a == "--seq-dir" && has_value) {
        seq_dir = argv[++i];
      } else if (a == "--out" && has_value) {
        out = argv[++i];
      } else if (a == "--floor" && has_value) {
        floor = std::stod(argv[++i]);
      } else if (a == "--gate" && has_value) {
        gate = argv[++i];
      } else if (a == "--i-know-this-is-test") {
        allow_test = true;
      } else {
        return usage();
      }
    }
    if (models_dir.empty() || manifest.empty() || seq_dir.empty() || out.empty()) return usage();
    const auto models = baykus::load_models(models_dir, manifest);
    const baykus::TrackerConfig cfg;  // takip kapalı; oturum açılmaz
    baykus::Sessions sessions(cfg, models.gate.window, models.gate.min_frames, models.gate.tol_px);
    baykus::Runtime runtime(models, cfg, sessions);
    const auto rec = baykus::detect_sequence(runtime, seq_dir, floor, gate, allow_test);
    baykus::write_gzip(out, rec.dump());
    std::cout << out << " (" << rec.at("frames").size() << " kare)\n";
  } catch (const std::exception& e) {
    std::cerr << "hata: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
