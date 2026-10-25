#include "baykus/bench.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <utility>

#include "baykus/files.hpp"
#include "baykus/zones.hpp"

namespace baykus {

namespace {

using json = nlohmann::json;

constexpr const char* kStages[] = {"decode", "health", "pre", "infer", "post", "track", "total"};

// Faz 4 dizi kipi: ego, motion ve zones aşamaları da ölçülür (hareket kapalıysa 0).
constexpr const char* kSeqStages[] = {"decode", "health", "pre",    "infer", "post",
                                      "track",  "ego",    "motion", "zones", "total"};

// Boş liste → null; doluysa total p95'i.
json p95_or_null(const std::vector<double>& v) {
  return v.empty() ? json(nullptr) : json(percentile(v, 95));
}

// Aynı boyutta tamamen sıfır PNG (IR düşmesi senaryosu); boyut başına bir kez kodlanır.
std::string zero_png(const std::string& ir, std::map<std::pair<int, int>, std::string>& cache) {
  const cv::Mat buf(1, static_cast<int>(ir.size()), CV_8UC1, const_cast<char*>(ir.data()));
  const cv::Mat img = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
  if (img.empty()) throw std::runtime_error("IR düşmesi: kare çözülemedi");
  const std::pair<int, int> key(img.cols, img.rows);
  if (const auto it = cache.find(key); it != cache.end()) return it->second;
  const cv::Mat zeros(img.size(), CV_8UC1, cv::Scalar(0));
  std::vector<unsigned char> out;
  if (!cv::imencode(".png", zeros, out)) throw std::runtime_error("IR düşmesi: PNG kodlanamadı");
  return cache.emplace(key, std::string(out.begin(), out.end())).first->second;
}

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

std::vector<SequenceFrame> load_sequence(const std::string& dir, int count) {
  const std::filesystem::path d(dir);
  const auto index = json::parse(read_bytes((d / "index.json").string()));
  const auto nums = index.at("frames");
  if (count < 0 || static_cast<std::size_t>(count) > nums.size()) {
    throw std::runtime_error(dir + ": " + std::to_string(count) + " kare istendi, dizide " +
                             std::to_string(nums.size()) + " var");
  }
  std::vector<SequenceFrame> out;
  for (int i = 0; i < count; ++i) {
    const int n = nums[static_cast<std::size_t>(i)].get<int>();
    char name[16];
    std::snprintf(name, sizeof(name), "%06d", n);
    out.push_back({n, read_bytes((d / "rgb" / (std::string(name) + ".jpg")).string()),
                   read_bytes((d / "ir" / (std::string(name) + ".png")).string())});
  }
  return out;
}

json bench_sequence(Runtime& runtime, Sessions& sessions, const std::vector<SequenceFrame>& frames,
                    const SequenceOptions& opt) {
  if (frames.empty()) throw std::invalid_argument("dizi boş");
  if (opt.iters <= 0) throw std::invalid_argument("iters pozitif olmalı");
  // Bölgeler oturumun hareket parçalarına yazılır: hareket kapalıysa bölge de yok.
  if (opt.zones) {
    bool created = false;
    const auto session = sessions.get("bench", &created);
    if (!session->motion) throw std::invalid_argument("bölgeler hareket açıkken");
    session->motion->zones.set(parse_zone_defs(*opt.zones), 1);
  }

  std::map<std::string, std::vector<double>> times;
  std::vector<double> keyframe, relink, reloc_attempt, ir_drop, seek;
  std::vector<int> frames_sent;
  std::map<std::pair<int, int>, std::string> zero_cache;
  int seeks = 0;
  int measured = 0;
  bool seek_next = false;
  std::size_t pos = 0;
  for (int i = 0; i < opt.warmup + opt.iters; ++i) {
    const auto& f = frames[pos];
    Request req;
    req.rgb = f.rgb;
    req.ir = f.ir;
    if (opt.ir_drop && f.frame >= opt.ir_drop->first && f.frame <= opt.ir_drop->second) {
      req.ir = zero_png(f.ir, zero_cache);
    }
    req.gate = "advise";
    req.session = "bench";
    req.conf = opt.conf;
    req.track = true;
    req.frame = f.frame;
    const auto r = runtime.process(req);
    frames_sent.push_back(f.frame);
    pos = (pos + 1) % frames.size();
    if (i < opt.warmup) continue;

    ++measured;
    const double total = r.timing_ms.count("total") ? r.timing_ms.at("total") : 0.0;
    for (const char* k : kSeqStages) {
      const auto it = r.timing_ms.find(k);
      times[k].push_back(it == r.timing_ms.end() ? 0.0 : it->second);
    }
    if (seek_next) {
      seek.push_back(total);
      seek_next = false;
    }
    if (opt.ir_drop && f.frame >= opt.ir_drop->first && f.frame <= opt.ir_drop->second) {
      ir_drop.push_back(total);
    }
    // Kare türleri ayrık değil: yeniden konumlanan kare anahtar kare de olabilir.
    if (r.ego) {
      if (r.ego->keyframe) keyframe.push_back(total);
      if (r.ego->relinked) relink.push_back(total);
      if (r.ego->provisional || r.ego->relocalized) reloc_attempt.push_back(total);
    }
    // Seek: ölçülen kare sayısı `seek_every`nin katı ve ölçülecek kare kaldıysa geri sar.
    if (opt.seek_every > 0 && measured % opt.seek_every == 0 && measured < opt.iters) {
      pos = pos > 50 ? pos - 50 : 0;
      sessions.reset("bench", Scope::Tracks);
      ++seeks;
      seek_next = true;
    }
  }

  json stages = json::object();
  for (const char* k : kSeqStages) stages[k] = summarize(times[k]);
  return {{"stages", stages},
          {"fps", 1000.0 / stages.at("total").at("mean").get<double>()},
          {"frames_sent", frames_sent},
          {"seeks", seeks},
          {"keyframe_p95", p95_or_null(keyframe)},
          {"relink_p95", p95_or_null(relink)},
          {"reloc_attempt_p95", p95_or_null(reloc_attempt)},
          {"ir_drop_p95", p95_or_null(ir_drop)},
          {"seek_p95", p95_or_null(seek)}};
}

}  // namespace baykus
