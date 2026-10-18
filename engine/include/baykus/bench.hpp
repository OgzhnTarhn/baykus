#pragma once

// Gecikme ölçümü (Faz 3 tasarımı §6.4, Kabul 5): `Runtime` üzerinden, bellekteki baytlarla,
// gate=advise, takip açık, tek oturum, kareler döngüde. Aşama başına p50/p95/ortalama (H5 bench
// kaydıyla aynı: NumPy `percentile` doğrusal).
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "baykus/runtime.hpp"

namespace baykus {

// numpy.percentile(v, q) (method="linear"); boş vektör: std::invalid_argument.
double percentile(std::vector<double> v, double q);
// {"p50", "p95", "mean", "n"}
nlohmann::json summarize(const std::vector<double>& v);

struct BenchFrame {
  std::string rgb;
  std::string ir;
};

// `data/edge/bench_inputs` düzeni: images.json sırasıyla rgb/images/<split>/<stem>.jpg ve
// ir/images/<split>/<stem>.png. Eksik dosya: std::runtime_error.
std::vector<BenchFrame> load_bench_inputs(const std::string& dir);

// Isınma + ölçüm; dönüş {"stages": {decode, health, pre, infer, post, track, total}, "fps"}.
// fps = 1000 / ortalama total.
nlohmann::json bench_memory(Runtime& runtime, const std::vector<BenchFrame>& frames, int warmup,
                            int iters, double conf);

}  // namespace baykus
