// Gecikme ölçümü (Faz 3 tasarımı §6.4, Kabul 5; plan Görev 27). Bellek içi kip `Runtime`'ı doğrudan
// çağırır (motor p95 = decode + health + pre + infer + post + track, HTTP hariç; eşik 100 ms).
// `--http` kipi aynı kareleri sunucuya gönderir ve gidiş-dönüş süresini de yazar (eşik yok).
// Faz 4 dizi kipi (plan Görev 17): `--sequence DIR` ardışık kareleri `index.json` sırasıyla işler;
// ego (`--motion`, `--no-ego`), IR düşmesi (`--ir-drop`), seek (`--seek-every`) ve bölgeler
// (`--zones`) senaryoları aşama p95'lerine ek olarak yazılır. `--http` dizi kipinde kullanılamaz.
// Kullanım: baykus_bench --inputs DIR --out F.json [--models /models]
//             [--manifest /etc/baykus/models.json] [--tracker /etc/baykus/tracker.json]
//             [--warmup 50] [--iters 500] [--http http://engine:8080] [--hardware laptop-docker]
//             [--meta anahtar=değer ...]
//          baykus_bench --sequence DIR --out F.json [--motion /etc/baykus/motion.json] [--no-ego]
//             [--ir-drop FROM:TO] [--seek-every N] [--zones F.json] [--models /models]
//             [--manifest M] [--tracker T] [--warmup N] [--iters N] [--hardware H] [--meta k=v ...]
#include <httplib.h>

#include <chrono>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "baykus/bench.hpp"
#include "baykus/cpu.hpp"
#include "baykus/files.hpp"
#include "baykus/models.hpp"
#include "baykus/motion_config.hpp"
#include "baykus/runtime.hpp"
#include "baykus/sessions.hpp"
#include "baykus/tracker.hpp"
#include "baykus/version.hpp"

namespace {

using json = nlohmann::json;

constexpr double kKabul5Ms = 100.0;
constexpr double kConf = 0.25;
constexpr const char* kStages[] = {"decode", "health", "pre", "infer", "post", "track", "total"};

int usage() {
  std::cerr << "kullanım: baykus_bench --inputs DIR --out F.json [--models DIR] [--manifest M] "
               "[--tracker T] [--warmup N] [--iters N] [--http URL] [--hardware H] "
               "[--meta k=v ...]\n"
               "       baykus_bench --sequence DIR --out F.json [--motion F] [--no-ego] "
               "[--ir-drop FROM:TO] [--seek-every N] [--zones F.json] [--models DIR] "
               "[--manifest M] [--tracker T] [--warmup N] [--iters N] [--hardware H] "
               "[--meta k=v ...]\n";
  return 2;
}

// "FROM:TO" → {from, to}; biçim bozuksa std::invalid_argument.
std::pair<int, int> parse_ir_drop(const std::string& s) {
  const auto colon = s.find(':');
  if (colon == std::string::npos) throw std::invalid_argument("--ir-drop FROM:TO olmalı: " + s);
  const int from = std::stoi(s.substr(0, colon));
  const int to = std::stoi(s.substr(colon + 1));
  if (from > to) throw std::invalid_argument("--ir-drop FROM <= TO olmalı: " + s);
  return {from, to};
}

// /proc/self/status satırı (kB) → MB; yoksa null
json proc_status_mb(const std::string& key) {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind(key + ":", 0) == 0) return std::stod(line.substr(key.size() + 1)) / 1000.0;
  }
  return nullptr;
}

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
  }
  return "";
}

std::string utc_now() {
  const std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

json bench_http(const std::string& url, const std::vector<baykus::BenchFrame>& frames, int warmup,
                int iters, json* model_info) {
  httplib::Client client(url);
  client.set_read_timeout(60, 0);
  const auto m = client.Get("/v1/model");
  if (!m || m->status != 200) throw std::runtime_error(url + "/v1/model yanıt vermedi");
  *model_info = json::parse(m->body);
  std::map<std::string, std::vector<double>> times;
  const std::string path = "/v1/detect?gate=advise&track=true&session=bench&conf=0.25";
  for (int i = 0; i < warmup + iters; ++i) {
    const auto& f = frames[static_cast<std::size_t>(i) % frames.size()];
    const httplib::UploadFormDataItems items = {{"rgb", f.rgb, "rgb.jpg", "image/jpeg"},
                                                {"ir", f.ir, "ir.png", "image/png"}};
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = client.Post(path, items);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!r || r->status != 200) {
      throw std::runtime_error("istek başarısız: " + (r ? std::to_string(r->status) : "bağlantı"));
    }
    if (i < warmup) continue;
    const auto timing = json::parse(r->body).at("timing_ms");
    for (const char* k : kStages) times[k].push_back(timing.at(k).get<double>());
    times["http_total"].push_back(ms);
  }
  json stages = json::object();
  for (const char* k : kStages) stages[k] = baykus::summarize(times[k]);
  return {{"stages", stages},
          {"http_total", baykus::summarize(times["http_total"])},
          {"fps", 1000.0 / baykus::summarize(times["http_total"]).at("mean").get<double>()}};
}

}  // namespace

int main(int argc, char** argv) {
  std::string inputs, out, url, hardware = "laptop-docker", models_dir = "/models",
                                manifest = "/etc/baykus/models.json",
                                tracker = "/etc/baykus/tracker.json";
  std::string sequence, motion = "/etc/baykus/motion.json", zones_file;
  bool no_ego = false;
  int seek_every = 0;
  std::optional<std::pair<int, int>> ir_drop;
  int warmup = 50, iters = 500;
  json meta = json::object();
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const bool v = i + 1 < argc;
      if (a == "--inputs" && v) {
        inputs = argv[++i];
      } else if (a == "--sequence" && v) {
        sequence = argv[++i];
      } else if (a == "--motion" && v) {
        motion = argv[++i];
      } else if (a == "--no-ego") {
        no_ego = true;
      } else if (a == "--ir-drop" && v) {
        ir_drop = parse_ir_drop(argv[++i]);
      } else if (a == "--seek-every" && v) {
        seek_every = std::stoi(argv[++i]);
      } else if (a == "--zones" && v) {
        zones_file = argv[++i];
      } else if (a == "--out" && v) {
        out = argv[++i];
      } else if (a == "--models" && v) {
        models_dir = argv[++i];
      } else if (a == "--manifest" && v) {
        manifest = argv[++i];
      } else if (a == "--tracker" && v) {
        tracker = argv[++i];
      } else if (a == "--warmup" && v) {
        warmup = std::stoi(argv[++i]);
      } else if (a == "--iters" && v) {
        iters = std::stoi(argv[++i]);
      } else if (a == "--http" && v) {
        url = argv[++i];
      } else if (a == "--hardware" && v) {
        hardware = argv[++i];
      } else if (a == "--meta" && v) {
        const std::string kv = argv[++i];
        const auto eq = kv.find('=');
        if (eq == std::string::npos) return usage();
        meta[kv.substr(0, eq)] = kv.substr(eq + 1);
      } else {
        return usage();
      }
    }
    // Tam olarak biri: --inputs ya da --sequence.
    if (inputs.empty() == sequence.empty()) return usage();
    if (out.empty() || iters <= 0 || warmup < 0 || seek_every < 0) return usage();
    if (!sequence.empty() && !url.empty()) return usage();  // dizi kipi HTTP ile ölçülmez
    // Dizi kipine özgü bayraklar `--inputs` kipinde sessizce yutulmaz.
    if (sequence.empty() && (no_ego || ir_drop || seek_every > 0 || !zones_file.empty())) {
      return usage();
    }
    if (!sequence.empty()) {
      const auto frames = baykus::load_sequence(sequence, warmup + iters);
      const auto index = json::parse(baykus::read_bytes(sequence + "/index.json"));
      const auto models = baykus::load_models(models_dir, manifest);
      const auto cfg = baykus::TrackerConfig::load(tracker);
      std::optional<baykus::MotionCfg> motion_cfg;
      if (!no_ego) motion_cfg = baykus::MotionCfg::load(motion);
      std::unique_ptr<baykus::Sessions> sessions;
      if (motion_cfg) {
        sessions = std::make_unique<baykus::Sessions>(cfg, *motion_cfg, models.gate.window,
                                                      models.gate.min_frames, models.gate.tol_px);
      } else {
        sessions = std::make_unique<baykus::Sessions>(cfg, models.gate.window,
                                                      models.gate.min_frames, models.gate.tol_px);
      }
      baykus::Runtime runtime(models, cfg, *sessions);
      baykus::SequenceOptions opt;
      opt.warmup = warmup;
      opt.iters = iters;
      opt.conf = kConf;
      opt.seek_every = seek_every;
      opt.ir_drop = ir_drop;
      if (!zones_file.empty()) opt.zones = json::parse(baykus::read_bytes(zones_file));
      const auto r = baykus::bench_sequence(runtime, *sessions, frames, opt);
      const json ir_drop_str =
          ir_drop ? json(std::to_string(ir_drop->first) + ":" + std::to_string(ir_drop->second))
                  : json(nullptr);
      json rec = {{"pipeline", "e3c"},
                  {"variant", "onnx_fp32_engine"},
                  {"hardware", hardware},
                  {"mode", "memory"},
                  {"protocol",
                   {{"inputs", "sequence"},
                    {"seq", index.at("name")},
                    {"warmup", warmup},
                    {"runs", iters},
                    {"conf", kConf},
                    {"gate", "advise"},
                    {"track", true},
                    {"session", "single"},
                    {"provider", "cpu"},
                    {"batch", 1},
                    {"ego", !no_ego},
                    {"ir_drop", ir_drop_str},
                    {"seek_every", seek_every},
                    {"zones", zones_file.empty() ? json(nullptr) : json(zones_file)}}},
                  {"threads", baykus::cpu_cfg_json(
                                  baykus::cpu_cfg_from_env())}};  // null = kütüphane varsayılanı
      rec["stages"] = r.at("stages");
      rec["fps"] = r.at("fps");
      rec["peak_rss_mb"] = proc_status_mb("VmHWM");
      const double p95 = r.at("stages").at("total").at("p95").get<double>();
      rec["kabul5"] = {
          {"threshold_ms", kKabul5Ms}, {"total_p95_ms", p95}, {"passed", p95 <= kKabul5Ms}};
      rec["models"] = {
          {{"name", "e3c"}, {"sha256", models.manifest.at("files").at("e3c.onnx").at("sha256")}}};
      rec["keyframe_p95"] = r.at("keyframe_p95");
      rec["relink_p95"] = r.at("relink_p95");
      rec["reloc_attempt_p95"] = r.at("reloc_attempt_p95");
      rec["ir_drop_p95"] = r.at("ir_drop_p95");
      rec["seek_p95"] = r.at("seek_p95");
      rec["seeks"] = r.at("seeks");
      json system = {{"nproc", std::thread::hardware_concurrency()},
                     {"cpu_model", cpu_model()},
                     {"opencv", baykus::opencv_version()},
                     {"onnxruntime", baykus::onnxruntime_version()}};
      for (const auto& [k, val] : meta.items()) system[k] = val;
      rec["system"] = system;
      rec["created_at"] = utc_now();
      std::ofstream(out) << rec.dump(1) << "\n";
      const auto& total = rec.at("stages").at("total");
      std::cout << out << ": total p50 " << total.at("p50").get<double>() << " ms, p95 "
                << total.at("p95").get<double>() << " ms\n";
      return 0;
    }
    const auto frames = baykus::load_bench_inputs(inputs);
    json rec = {{"pipeline", "e3c"},
                {"variant", "onnx_fp32_engine"},
                {"hardware", hardware},
                {"mode", url.empty() ? "memory" : "http"},
                {"protocol",
                 {{"warmup", warmup},
                  {"runs", iters},
                  {"images", frames.size()},
                  {"conf", kConf},
                  {"gate", "advise"},
                  {"track", true},
                  {"session", "single"},
                  {"provider", "cpu"},
                  {"batch", 1}}},
                {"threads", baykus::cpu_cfg_json(
                                baykus::cpu_cfg_from_env())}};  // null = kütüphane varsayılanı
    json system = {{"nproc", std::thread::hardware_concurrency()}, {"cpu_model", cpu_model()}};
    if (url.empty()) {
      const auto models = baykus::load_models(models_dir, manifest);
      const auto cfg = baykus::TrackerConfig::load(tracker);
      baykus::Sessions sessions(cfg, models.gate.window, models.gate.min_frames,
                                models.gate.tol_px);
      baykus::Runtime runtime(models, cfg, sessions);
      const auto r = baykus::bench_memory(runtime, frames, warmup, iters, kConf);
      rec["stages"] = r.at("stages");
      rec["fps"] = r.at("fps");
      rec["peak_rss_mb"] = proc_status_mb("VmHWM");
      const double p95 = r.at("stages").at("total").at("p95").get<double>();
      rec["kabul5"] = {
          {"threshold_ms", kKabul5Ms}, {"total_p95_ms", p95}, {"passed", p95 <= kKabul5Ms}};
      rec["models"] = {
          {{"name", "e3c"}, {"sha256", models.manifest.at("files").at("e3c.onnx").at("sha256")}}};
      system["opencv"] = baykus::opencv_version();
      system["onnxruntime"] = baykus::onnxruntime_version();
    } else {
      json model_info;
      const auto r = bench_http(url, frames, warmup, iters, &model_info);
      rec["url"] = url;
      rec["stages"] = r.at("stages");
      rec["http_total"] = r.at("http_total");
      rec["fps"] = r.at("fps");
      rec["peak_rss_mb"] = nullptr;  // sunucu ayrı süreç; istemcinin belleği anlamsız
      rec["kabul5"] = nullptr;       // Kabul 5 bellek içi kipte (HTTP hariç)
      rec["models"] = {{{"name", "e3c"}, {"sha256", model_info.at("sha256").at("model")}}};
    }
    for (const auto& [k, v] : meta.items()) system[k] = v;
    rec["system"] = system;
    rec["created_at"] = utc_now();
    std::ofstream(out) << rec.dump(1) << "\n";
    const auto& total = rec.at("stages").at("total");
    std::cout << out << ": total p50 " << total.at("p50").get<double>() << " ms, p95 "
              << total.at("p95").get<double>() << " ms";
    if (!url.empty())
      std::cout << ", http p95 " << rec.at("http_total").at("p95").get<double>() << " ms";
    std::cout << "\n";
  } catch (const std::exception& e) {
    std::cerr << "hata: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
