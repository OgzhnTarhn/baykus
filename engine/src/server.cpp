#include "baykus/server.hpp"

#include <httplib.h>

#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>

#include "baykus/classes.hpp"
#include "baykus/image_io.hpp"

namespace baykus {

namespace {

using json = nlohmann::json;

// 413 için ayrı tür: geçersiz istek (400) ile karışmasın.
struct TooLarge : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void send_json(httplib::Response& res, int status, const json& body) {
  res.status = status;
  res.set_content(body.dump(), "application/json");
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

json detections_json(const Response& r) {
  json out = json::array();
  for (std::size_t i = 0; i < r.dets.size(); ++i) {
    const auto& d = r.dets[i];
    const bool known = d.cls_id >= 0 && d.cls_id < static_cast<int>(kClassNames.size());
    out.push_back({{"cls", known ? std::string(kClassNames[d.cls_id]) : std::string("unknown")},
                   {"cls_id", d.cls_id},
                   {"score", d.score},
                   {"score_raw", static_cast<double>(d.score_raw)},
                   {"box", {{"x1", d.x1}, {"y1", d.y1}, {"x2", d.x2}, {"y2", d.y2}}},
                   {"track_id", r.track_ids[i] < 0 ? json(nullptr) : json(r.track_ids[i])}});
  }
  return out;
}

// İstek → yanıt; beklenen hatalar türüne göre koda çevrilir, sunucu ayakta kalır.
template <typename F>
void guarded(httplib::Response& res, F f) {
  try {
    f();
  } catch (const TooLarge& e) {
    send_error(res, 413, e.what());
  } catch (const ImageTooLarge& e) {
    send_error(res, 413, e.what());
  } catch (const std::invalid_argument& e) {
    send_error(res, 400, e.what());
  } catch (const std::exception& e) {
    std::cerr << "iç hata: " << e.what() << "\n";
    send_error(res, 500, "iç hata");
  }
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
  };

  // İki parça + çok parçalı başlıklar; tek parça sınırı işleyicide (413 iletisi parça adıyla).
  http_->set_payload_max_length(2 * kMaxPartBytes + 1024 * 1024);

  http_->Get("/health", [](const httplib::Request&, httplib::Response& res) {
    send_json(res, 200, {{"status", "ok"}, {"model_loaded", true}});
  });

  http_->Get("/v1/model", [model_info](const httplib::Request&, httplib::Response& res) {
    send_json(res, 200, model_info);
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
      const auto out = runtime.process(r);
      send_json(res, 200,
                {{"detections", detections_json(out)},
                 {"health", out.health},
                 {"timing_ms", out.timing_ms},
                 {"session_new", out.session_new}});
    });
  });

  http_->Post(
      "/v1/session/reset", [&sessions](const httplib::Request& req, httplib::Response& res) {
        guarded(res, [&] {
          const std::string id = req.has_param("session") ? req.get_param_value("session") : "";
          if (id.empty()) throw std::invalid_argument("session zorunlu");
          sessions.reset(id);
          send_json(res, 200, {{"session", id}, {"reset", true}});
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
