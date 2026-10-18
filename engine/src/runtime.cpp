#include "baykus/runtime.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

#include "baykus/files.hpp"

namespace baykus {

namespace {

using Clock = std::chrono::steady_clock;
using json = nlohmann::json;

constexpr float kTrackFloor = 0.1f;  // takip katmanı (tasarım §4.1)

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

Runtime::Runtime(const Models& models, const TrackerConfig& tracker_cfg, Sessions& sessions)
    : pipeline_(models), tracker_cfg_(tracker_cfg), sessions_(sessions) {}

Response Runtime::process(const Request& req) {
  // Doğrulama oturum açılmadan önce: hatalı istek oturum tablosunu değiştirmez.
  if (!(req.conf >= 0.0 && req.conf <= 1.0)) {
    throw std::invalid_argument("conf [0, 1] aralığında olmalı");
  }
  if (req.gate != "off" && req.gate != "advise" && req.gate != "auto") {
    throw std::invalid_argument("bilinmeyen gate modu: " + req.gate +
                                "; seçenekler: off, advise, auto");
  }
  if (req.track && req.session.empty()) {
    throw std::invalid_argument("takip için session gerekli");
  }

  Response r;
  std::shared_ptr<Session> s;
  std::unique_lock<std::mutex> session_lock;
  if (!req.session.empty()) {
    s = sessions_.get(req.session, &r.session_new);
    session_lock = std::unique_lock<std::mutex>(s->m);
  }

  FrameResult fr;
  {
    std::lock_guard<std::mutex> infer_lock(infer_mutex_);
    fr = pipeline_.run(req.rgb, req.ir, req.gate, req.floor, s ? &s->window : nullptr);
  }

  // Takip: takip katmanının bütün tespitleri (conf'tan bağımsız), ham skorla.
  std::map<int, int> ids;
  double track_ms = 0.0;
  if (req.track) {
    const auto t = Clock::now();
    std::vector<Det> in;
    for (const auto& d : fr.dets_all) {
      if (d.score_raw >= kTrackFloor) in.push_back(d);
    }
    for (const auto& [index, id] : s->tracker.update(in)) ids[index] = id;
    track_ms = ms_since(t);
  }

  r.health = std::move(fr.health);
  r.timing_ms = std::move(fr.timing_ms);
  r.timing_ms["track"] = track_ms;
  double total = 0.0;
  for (const char* k : {"decode", "health", "pre", "infer", "post", "track"}) {
    total += r.timing_ms.at(k);
  }
  r.timing_ms["total"] = total;
  r.dets = filter_display(fr.dets_all, req.conf);
  for (const auto& d : r.dets) {
    const auto it = ids.find(d.index);
    r.track_ids.push_back(it == ids.end() ? -1 : it->second);
  }
  return r;
}

json detect_sequence(Runtime& runtime, const std::string& seq_dir, double floor,
                     const std::string& gate, bool allow_test) {
  const std::filesystem::path d(seq_dir);
  const auto index = json::parse(read_bytes((d / "index.json").string()));
  if (index.value("split", "") == "test" && !allow_test) {
    throw std::invalid_argument(seq_dir +
                                ": test dizisi; yalnız son ölçümde --i-know-this-is-test ile");
  }
  json frames = json::array();
  for (const auto& f : index.at("frames")) {
    char name[16];
    std::snprintf(name, sizeof(name), "%06d", f.get<int>());
    Request req;
    req.rgb = read_bytes((d / "rgb" / (std::string(name) + ".jpg")).string());
    req.ir = read_bytes((d / "ir" / (std::string(name) + ".png")).string());
    req.gate = gate;
    req.conf = 0.0;  // bütün katman (kalibre skor ≥ 0)
    req.track = false;
    req.floor = static_cast<float>(floor);
    json dets = json::array();
    for (const auto& det : runtime.process(req).dets) {
      dets.push_back({{"index", det.index},
                      {"cls_id", det.cls_id},
                      {"score_raw", static_cast<double>(det.score_raw)},
                      {"score", det.score},
                      {"box", {det.x1, det.y1, det.x2, det.y2}}});
    }
    frames.push_back({{"frame", f}, {"dets", dets}});
  }
  return {{"seq", index.at("name")},
          {"split", index.at("split")},
          {"floor", floor},
          {"gate", gate},
          {"frames", frames}};
}

}  // namespace baykus
