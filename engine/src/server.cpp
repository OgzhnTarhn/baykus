#include "baykus/server.hpp"

#include <httplib.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

#include "baykus/classes.hpp"
#include "baykus/image_io.hpp"

namespace baykus {

namespace {

using json = nlohmann::json;

// 413 için ayrı tür: geçersiz istek (400) ile karışmasın.
struct TooLarge : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// 404 için ayrı tür: bilinmeyen oturum / alarm / iz özeti.
struct NotFound : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void send_json(httplib::Response& res, int status, const json& body) {
  res.status = status;
  // Geçersiz UTF-8 (ör. bozuk gövdeyi aktaran ayrıştırma iletisi) dökümü düşürmesin.
  res.set_content(body.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
}

void send_error(httplib::Response& res, int status, const std::string& message) {
  send_json(res, status, {{"error", message}});
}

// Parça dosya (filename'li) ya da düz alan olarak gelebilir.
std::optional<std::string> part(const httplib::Request& req, const std::string& key) {
  if (req.form.has_file(key)) return req.form.get_file(key).content;
  if (req.form.has_field(key)) return req.form.get_field(key);
  return std::nullopt;
}

double parse_conf(const std::string& s) {
  std::size_t used = 0;
  double v = 0.0;
  try {
    v = std::stod(s, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used != s.size() || s.empty() || !std::isfinite(v)) {
    throw std::invalid_argument("conf sayı olmalı: " + s);
  }
  return v;
}

bool parse_bool(const std::string& name, const std::string& s) {
  if (s == "true" || s == "1") return true;
  if (s == "false" || s == "0") return false;
  throw std::invalid_argument(name + " true ya da false olmalı: " + s);
}

int parse_int(const std::string& name, const std::string& s) {
  std::size_t used = 0;
  int v = 0;
  try {
    v = std::stoi(s, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used != s.size() || s.empty()) {
    throw std::invalid_argument(name + " tam sayı olmalı: " + s);
  }
  return v;
}

long parse_long(const std::string& name, const std::string& s) {
  std::size_t used = 0;
  long v = 0;
  try {
    v = std::stol(s, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used != s.size() || s.empty()) {
    throw std::invalid_argument(name + " tam sayı olmalı: " + s);
  }
  return v;
}

// İstek → yanıt; beklenen hatalar türüne göre koda çevrilir, sunucu ayakta kalır.
template <typename F>
void guarded(httplib::Response& res, F f) {
  try {
    f();
  } catch (const TooLarge& e) {
    send_error(res, 413, e.what());
  } catch (const NotFound& e) {
    send_error(res, 404, e.what());
  } catch (const ImageTooLarge& e) {
    send_error(res, 413, e.what());
  } catch (const std::invalid_argument& e) {
    send_error(res, 400, e.what());
  } catch (const std::exception& e) {
    std::cerr << "iç hata: " << e.what() << "\n";
    send_error(res, 500, "iç hata");
  }
}

// Oturum bulunamazsa 404; `session` yoksa 400; oturumda hareket kapalıysa 400. Oturum açmaz.
std::shared_ptr<Session> motion_session(Sessions& sessions, const httplib::Request& req) {
  const std::string id = req.has_param("session") ? req.get_param_value("session") : "";
  if (id.empty()) throw std::invalid_argument("session zorunlu");
  auto s = sessions.find(id);
  if (!s) throw NotFound("oturum yok: " + id);
  if (!s->motion) throw std::invalid_argument("bu oturumda hareket kapalı");
  return s;
}

}  // namespace

Server::Server(const Models& models, const TrackerConfig& tracker_cfg, Runtime& runtime,
               Sessions& sessions)
    : http_(std::make_unique<httplib::Server>()) {
  const auto& files = models.manifest.at("files");
  json classes = json::array();
  for (const auto name : kClassNames) classes.push_back(std::string(name));
  const json model_info = {
      {"name", "e3c"},
      {"family", "yolo26"},
      {"mode", "early_fusion"},
      {"precision", "fp32"},
      {"input_size", {runtime.imgsz(), runtime.imgsz()}},
      {"channels", runtime.channels()},
      {"classes", classes},
      {"sha256",
       {{"model", files.at("e3c.onnx").at("sha256")},
        {"thresholds", files.at("thresholds.json").at("sha256")},
        {"calibration", files.at("calibration.json").at("sha256")},
        {"gate_model", files.at("gate_model.json").at("sha256")}}},
      {"tracker", tracker_cfg.to_json()},
      {"tiles", {{"modes", {1, 4}}, {"default", 1}, {"window_ratio", {9, 16}}, {"nms_iou", 0.5}}},
  };

  // İki parça + çok parçalı başlıklar; tek parça sınırı işleyicide (413 iletisi parça adıyla).
  http_->set_payload_max_length(2 * kMaxPartBytes + 1024 * 1024);

  http_->Get("/health", [](const httplib::Request&, httplib::Response& res) {
    send_json(res, 200, {{"status", "ok"}, {"model_loaded", true}});
  });

  http_->Get("/v1/model", [model_info, &sessions](const httplib::Request&, httplib::Response& res) {
    json body = model_info;
    if (sessions.motion_cfg()) {
      body["motion"] = sessions.motion_cfg()->to_json();
      body["motion_sha256"] = sessions.motion_cfg()->sha256;
    } else {
      body["motion"] = nullptr;
      body["motion_sha256"] = nullptr;
    }
    send_json(res, 200, body);
  });

  http_->Post("/v1/detect", [&runtime](const httplib::Request& req, httplib::Response& res) {
    guarded(res, [&] {
      if (!req.is_multipart_form_data()) {
        throw std::invalid_argument("multipart/form-data bekleniyor (rgb ve ir parçaları)");
      }
      const auto rgb = part(req, "rgb");
      const auto ir = part(req, "ir");
      for (const auto& [name, p] : {std::pair{"rgb", &rgb}, std::pair{"ir", &ir}}) {
        if (p->has_value() && (*p)->size() > kMaxPartBytes) {
          throw TooLarge(std::string(name) + " parçası 10 MB'tan büyük");
        }
      }
      if (!rgb || !ir) throw std::invalid_argument("rgb ve ir parçaları zorunlu");
      Request r;
      r.rgb = *rgb;
      r.ir = *ir;
      if (req.has_param("conf")) r.conf = parse_conf(req.get_param_value("conf"));
      r.track = req.has_param("track") && parse_bool("track", req.get_param_value("track"));
      if (req.has_param("gate")) r.gate = req.get_param_value("gate");
      if (req.has_param("session")) r.session = req.get_param_value("session");
      if (req.has_param("frame")) {
        r.frame = parse_int("frame", req.get_param_value("frame"));
        if (*r.frame < 0) throw std::invalid_argument("frame 0 ya da daha büyük olmalı");
      }
      if (req.has_param("focus")) r.focus = parse_int("focus", req.get_param_value("focus"));
      if (req.has_param("tiles")) r.tiles = parse_int("tiles", req.get_param_value("tiles"));
      const auto out = runtime.process(r);
      send_json(res, 200, response_json(out));
    });
  });

  http_->Post(
      "/v1/session/reset", [&sessions](const httplib::Request& req, httplib::Response& res) {
        guarded(res, [&] {
          const std::string id = req.has_param("session") ? req.get_param_value("session") : "";
          if (id.empty()) throw std::invalid_argument("session zorunlu");
          Scope scope = Scope::All;
          std::string scope_name = "all";
          if (req.has_param("scope")) {
            scope_name = req.get_param_value("scope");
            if (scope_name == "all") {
              scope = Scope::All;
            } else if (scope_name == "tracks") {
              scope = Scope::Tracks;
            } else {
              throw std::invalid_argument("bilinmeyen scope: " + scope_name +
                                          "; seçenekler: all, tracks");
            }
          }
          sessions.reset(id, scope);
          send_json(res, 200, {{"session", id}, {"reset", true}, {"scope", scope_name}});
        });
      });

  http_->Put("/v1/session/zones", [&sessions](const httplib::Request& req, httplib::Response& res) {
    guarded(res, [&] {
      auto s = motion_session(sessions, req);
      std::lock_guard<std::mutex> lock(s->m);
      auto& mp = *s->motion;
      const std::size_t max_bytes = sessions.motion_cfg()->zones.body_max_bytes;
      if (req.body.size() > max_bytes) {
        throw TooLarge("gövde " + std::to_string(max_bytes) + " bayttan büyük");
      }
      json body;
      try {
        body = json::parse(req.body);
      } catch (const json::exception& e) {
        throw std::invalid_argument(std::string("gövde JSON değil: ") + e.what());
      }
      auto defs = parse_zone_defs(body);
      check_zone_gens(defs, mp.ego.generations());
      int max_gen = 1;
      const auto& gens = mp.ego.generations();
      if (!gens.empty()) {
        max_gen = std::max_element(gens.begin(), gens.end(), [](const auto& a, const auto& b) {
                    return a.first < b.first;
                  })->first;
      }
      // Eski tanımların ad / türü: alarm_clear kayıtları eski tanımla yazılır.
      std::map<std::string, std::pair<std::string, std::string>> old;
      for (const auto& z : mp.zones.defs_json()) {
        old[z.at("id").get<std::string>()] = {z.value("name", std::string()),
                                              z.value("kind", std::string())};
      }
      std::vector<ZoneEvent> cleared;
      mp.zones.set(std::move(defs), max_gen, &cleared);
      for (const auto& e : cleared) {
        const auto it = old.find(e.zone_id);
        if (it == old.end()) {
          mp.log.append(e, "", "");
        } else {
          mp.log.append(e, it->second.first, it->second.second);
        }
      }
      ZoneEvent updated;
      updated.n = s->n;
      updated.frame = s->last_frame;
      updated.kind = "zones_updated";
      updated.gen = s->has_ego ? s->last_gen : 1;
      mp.log.append(updated, "", "");
      send_json(res, 200, {{"zones", mp.zones.defs_json()}});
    });
  });

  http_->Get("/v1/session/zones", [&sessions](const httplib::Request& req, httplib::Response& res) {
    guarded(res, [&] {
      auto s = motion_session(sessions, req);
      std::lock_guard<std::mutex> lock(s->m);
      send_json(res, 200, {{"zones", s->motion->zones.defs_json()}});
    });
  });

  http_->Get(
      "/v1/session/events", [&sessions](const httplib::Request& req, httplib::Response& res) {
        guarded(res, [&] {
          auto s = motion_session(sessions, req);
          long since = 0;
          if (req.has_param("since")) {
            since = parse_long("since", req.get_param_value("since"));
            if (since < 0) throw std::invalid_argument("since 0 ya da daha büyük olmalı");
          }
          std::string format = "json";
          if (req.has_param("format")) format = req.get_param_value("format");
          if (format != "json" && format != "csv") {
            throw std::invalid_argument("bilinmeyen format: " + format + "; seçenekler: json, csv");
          }
          std::lock_guard<std::mutex> lock(s->m);
          if (format == "csv") {
            res.status = 200;
            res.set_content(s->motion->log.csv_since(since), "text/csv; charset=utf-8");
          } else {
            send_json(res, 200, s->motion->log.since(since));
          }
        });
      });

  http_->Post(
      "/v1/session/alarms/ack", [&sessions](const httplib::Request& req, httplib::Response& res) {
        guarded(res, [&] {
          auto s = motion_session(sessions, req);
          const std::string alarm = req.has_param("alarm") ? req.get_param_value("alarm") : "";
          if (alarm.empty()) throw std::invalid_argument("alarm zorunlu");
          std::lock_guard<std::mutex> lock(s->m);
          auto& mp = *s->motion;
          std::vector<ZoneEvent> out;
          if (!mp.zones.ack(alarm, s->n, s->last_frame, &out)) {
            throw NotFound("alarm yok: " + alarm);
          }
          for (const auto& e : out) {
            const ZoneDef* z = mp.zones.find(e.zone_id);
            mp.log.append(e, z != nullptr ? z->name : "", z != nullptr ? z->kind : "");
          }
          send_json(res, 200, {{"alarm", alarm}, {"acked", true}});
        });
      });

  http_->Get("/v1/session/track", [&sessions](const httplib::Request& req, httplib::Response& res) {
    guarded(res, [&] {
      auto s = motion_session(sessions, req);
      if (!req.has_param("id")) throw std::invalid_argument("id zorunlu");
      const int id = parse_int("id", req.get_param_value("id"));
      std::lock_guard<std::mutex> lock(s->m);
      const auto summary = s->motion->summary.get(id);
      if (!summary) throw NotFound("iz özeti yok: " + std::to_string(id));
      send_json(res, 200, *summary);
    });
  });

  // httplib'in kendi hataları (404, 413, ...) da JSON gövdeyle
  http_->set_error_handler([](const httplib::Request&, httplib::Response& res) {
    if (!res.body.empty()) return httplib::Server::HandlerResponse::Unhandled;
    send_error(res, res.status, httplib::status_message(res.status));
    return httplib::Server::HandlerResponse::Handled;
  });
  http_->set_exception_handler([](const httplib::Request&, httplib::Response& res,
                                  std::exception_ptr) { send_error(res, 500, "iç hata"); });
}

Server::~Server() = default;

int Server::bind_any_port(const std::string& host) { return http_->bind_to_any_port(host); }

bool Server::bind(const std::string& host, int port) { return http_->bind_to_port(host, port); }

bool Server::listen_after_bind() { return http_->listen_after_bind(); }

void Server::stop() { http_->stop(); }

void Server::wait_until_ready() const { http_->wait_until_ready(); }

}  // namespace baykus
