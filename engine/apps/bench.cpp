// Gecikme ölçümü (Faz 3 tasarımı §6.4, Kabul 5; plan Görev 27). Bellek içi kip `Runtime`'ı doğrudan
// çağırır (motor p95 = decode + health + pre + infer + post + track, HTTP hariç; eşik 100 ms).
// `--http` kipi aynı kareleri sunucuya gönderir ve gidiş-dönüş süresini de yazar (eşik yok).
// Kullanım: baykus_bench --inputs DIR --out F.json [--models /models]
//             [--manifest /etc/baykus/models.json] [--tracker /etc/baykus/tracker.json]
//             [--warmup 50] [--iters 500] [--http http://engine:8080] [--hardware laptop-docker]
//             [--meta anahtar=değer ...]
#include <httplib.h>

#include <chrono>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "baykus/bench.hpp"
#include "baykus/files.hpp"
#include "baykus/models.hpp"
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
               "[--meta k=v ...]\n";
  return 2;
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
  int warmup = 50, iters = 500;
  json meta = json::object();
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const bool v = i + 1 < argc;
      if (a == "--inputs" && v) {
        inputs = argv[++i];
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
    if (inputs.empty() || out.empty() || iters <= 0 || warmup < 0) return usage();
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
                {"threads", nullptr}};  // ORT varsayılanı (Python OrtBackend gibi)
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
