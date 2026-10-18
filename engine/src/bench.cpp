#include "baykus/bench.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <numeric>
#include <stdexcept>

#include "baykus/files.hpp"

namespace baykus {

namespace {

using json = nlohmann::json;

constexpr const char* kStages[] = {"decode", "health", "pre", "infer", "post", "track", "total"};

}  // namespace

double percentile(std::vector<double> v, double q) {
  if (v.empty()) throw std::invalid_argument("boş dizide yüzdelik yok");
  std::sort(v.begin(), v.end());
  const double idx = q / 100.0 * static_cast<double>(v.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(idx));
  const std::size_t hi = std::min(lo + 1, v.size() - 1);
  const double t = idx - static_cast<double>(lo);
  const double a = v[lo];
  const double b = v[hi];
  // numpy `_lerp`: t ≥ 0,5 iken üst uçtan (yuvarlama aynı olsun)
  return t >= 0.5 ? b - (b - a) * (1.0 - t) : a + (b - a) * t;
}

json summarize(const std::vector<double>& v) {
  const double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
  return {{"p50", percentile(v, 50)}, {"p95", percentile(v, 95)}, {"mean", mean}, {"n", v.size()}};
}

std::vector<BenchFrame> load_bench_inputs(const std::string& dir) {
  const std::filesystem::path d(dir);
  std::vector<BenchFrame> out;
  for (const auto& e : json::parse(read_bytes((d / "images.json").string()))) {
    const std::string stem = e.at("stem").get<std::string>();
    const std::string split = e.at("split").get<std::string>();
    out.push_back({read_bytes((d / "rgb" / "images" / split / (stem + ".jpg")).string()),
                   read_bytes((d / "ir" / "images" / split / (stem + ".png")).string())});
  }
  if (out.empty()) throw std::runtime_error(dir + ": bench girdisi yok");
  return out;
}

json bench_memory(Runtime& runtime, const std::vector<BenchFrame>& frames, int warmup, int iters,
                  double conf) {
  std::map<std::string, std::vector<double>> times;
  for (int i = 0; i < warmup + iters; ++i) {
    const auto& f = frames[static_cast<std::size_t>(i) % frames.size()];
    Request req;
    req.rgb = f.rgb;
    req.ir = f.ir;
    req.gate = "advise";
    req.session = "bench";
    req.conf = conf;
    req.track = true;
    const auto r = runtime.process(req);
    if (i < warmup) continue;
    for (const char* k : kStages) times[k].push_back(r.timing_ms.at(k));
  }
  json stages = json::object();
  for (const char* k : kStages) stages[k] = summarize(times[k]);
  return {{"stages", stages}, {"fps", 1000.0 / stages.at("total").at("mean").get<double>()}};
}

}  // namespace baykus
