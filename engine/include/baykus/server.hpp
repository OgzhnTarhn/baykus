#pragma once

// HTTP API (Faz 3 tasarımı §4.1; spec §15.3; Faz 4 uçları docs/design/events_rules.md 'Uçlar'):
// GET /health, GET /v1/model, POST /v1/detect, POST /v1/session/reset, PUT/GET /v1/session/zones,
// GET /v1/session/events, POST /v1/session/alarms/ack, GET /v1/session/track. Hatalar JSON
// `{"error"}`: geçersiz istek 400, parça / gövde > sınır 413, bilinmeyen yol / oturum / alarm / iz
// 404, iç hata 500; tek hatalı istek sunucuyu düşürmez.
#include <cstddef>
#include <memory>
#include <string>

#include "baykus/models.hpp"
#include "baykus/runtime.hpp"
#include "baykus/sessions.hpp"
#include "baykus/tracker.hpp"

namespace httplib {
class Server;
}

namespace baykus {

constexpr std::size_t kMaxPartBytes = 10u * 1024u * 1024u;

class Server {
 public:
  Server(const Models& models, const TrackerConfig& tracker_cfg, Runtime& runtime,
         Sessions& sessions);
  ~Server();

  // Boş bir porta bağlanır ve portu döndürür (testler); bağlanamazsa -1.
  int bind_any_port(const std::string& host);
  bool bind(const std::string& host, int port);
  // Bağlandıktan sonra istekleri işler; stop() çağrılana kadar döner.
  bool listen_after_bind();
  void stop();
  void wait_until_ready() const;

 private:
  std::unique_ptr<httplib::Server> http_;
};

}  // namespace baykus
