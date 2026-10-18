// Motor sunucusu (Faz 3 tasarımı §4.1 `baykus_server`; plan Görev 25). Açılışta modeller manifestle
// denetlenir; eksik ya da farklıysa nedeni yazılır ve çıkış 1 (motor açılmaz). SIGINT/SIGTERM →
// düzgün durma (docker stop).
// Kullanım: baykus_server [--models /models] [--manifest /etc/baykus/models.json]
//                         [--tracker /etc/baykus/tracker.json] [--host 0.0.0.0] [--port 8080]
#include <csignal>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

#include "baykus/models.hpp"
#include "baykus/runtime.hpp"
#include "baykus/server.hpp"
#include "baykus/sessions.hpp"
#include "baykus/tracker.hpp"

namespace {

int usage() {
  std::cerr << "kullanım: baykus_server [--models DIR] [--manifest M] [--tracker T] [--host H] "
               "[--port P]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string models_dir = "/models", manifest = "/etc/baykus/models.json",
              tracker = "/etc/baykus/tracker.json", host = "0.0.0.0";
  int port = 8080;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has_value = i + 1 < argc;
    if (a == "--models" && has_value) {
      models_dir = argv[++i];
    } else if (a == "--manifest" && has_value) {
      manifest = argv[++i];
    } else if (a == "--tracker" && has_value) {
      tracker = argv[++i];
    } else if (a == "--host" && has_value) {
      host = argv[++i];
    } else if (a == "--port" && has_value) {
      port = std::atoi(argv[++i]);
    } else {
      return usage();
    }
  }
  // Sinyaller bütün iş parçacıklarında engellenir; yalnız bekleyen iş parçacığı alır.
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);
  try {
    const auto models = baykus::load_models(models_dir, manifest);
    const auto cfg = baykus::TrackerConfig::load(tracker);
    baykus::Sessions sessions(cfg, models.gate.window, models.gate.min_frames, models.gate.tol_px);
    baykus::Runtime runtime(models, cfg, sessions);
    baykus::Server server(models, cfg, runtime, sessions);
    if (!server.bind(host, port)) {
      std::cerr << "bağlanamadı: " << host << ":" << port << "\n";
      return 1;
    }
    std::thread waiter([&server, &signals] {
      int sig = 0;
      sigwait(&signals, &sig);
      std::cerr << "sinyal " << sig << ": duruyor\n";
      server.stop();
    });
    waiter.detach();
    std::cerr << "baykus_server " << host << ":" << port << " (model e3c, "
              << models.manifest.value("release", "") << ")\n";
    server.listen_after_bind();
  } catch (const std::exception& e) {
    std::cerr << "motor açılmadı: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
