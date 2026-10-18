// Tespit dosyasından takip (Faz 3 tasarımı §4.1 `baykus_track`; plan Görev 23). Çıktı Python
// `baykus-track run` ile aynı biçimde .json.gz: `baykus-track compare` iki dosyayı karşılaştırır.
// Kullanım: baykus_track --dets F.json.gz [--config C] --out O.json.gz [--dump N ...]
//                        [--i-know-this-is-test]
#include <cstdlib>
#include <exception>
#include <iostream>
#include <set>
#include <string>

#include "baykus/files.hpp"
#include "baykus/tracker.hpp"

namespace {

int usage() {
  std::cerr << "kullanım: baykus_track --dets F.json.gz [--config C] --out O.json.gz "
               "[--dump N ...] [--i-know-this-is-test]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dets, out, config = "/etc/baykus/tracker.json";
  std::set<int> dump;
  bool allow_test = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--dets" && i + 1 < argc) {
      dets = argv[++i];
    } else if (a == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (a == "--config" && i + 1 < argc) {
      config = argv[++i];
    } else if (a == "--dump") {
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        dump.insert(std::atoi(argv[++i]));
      }
    } else if (a == "--i-know-this-is-test") {
      allow_test = true;
    } else {
      return usage();
    }
  }
  if (dets.empty() || out.empty()) return usage();
  try {
    const auto cfg = baykus::TrackerConfig::load(config);
    const auto rec =
        baykus::track_file(nlohmann::json::parse(baykus::read_gzip(dets)), cfg, dump, allow_test);
    baykus::write_gzip(out, rec.dump());
    std::cout << out << " (" << rec.at("sequences").size() << " dizi)\n";
  } catch (const std::exception& e) {
    std::cerr << "hata: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
