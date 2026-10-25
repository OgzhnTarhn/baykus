#pragma once

// Gecikme ölçümü (Faz 3 tasarımı §6.4, Kabul 5): `Runtime` üzerinden, bellekteki baytlarla,
// gate=advise, takip açık, tek oturum, kareler döngüde. Aşama başına p50/p95/ortalama (H5 bench
// kaydıyla aynı: NumPy `percentile` doğrusal).
// Faz 4 dizi kipi (plan Görev 17): ardışık kareler, ego kare türleri, IR düşmesi, seek, bölgeler.
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "baykus/runtime.hpp"
#include "baykus/sessions.hpp"

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

// Dizi karesi: `frame` = `index.json`'daki kare numarası (ego'ya geçer).
struct SequenceFrame {
  int frame;
  std::string rgb;
  std::string ir;
};

// `<dizi>/{index.json, rgb/NNNNNN.jpg, ir/NNNNNN.png}`: `index.json` sırasıyla ilk `count` kare.
// `count` kareden fazlaysa ya da dosya yoksa std::runtime_error.
std::vector<SequenceFrame> load_sequence(const std::string& dir, int count);

struct SequenceOptions {
  int warmup = 0;
  int iters = 1;
  double conf = 0.25;
  int seek_every = 0;                          // 0 = kapalı; N ölçülen karede bir geri 50 kare
  std::optional<std::pair<int, int>> ir_drop;  // [from, to] kare numarası, dahil
  std::optional<nlohmann::json> zones;         // bölge gövdesi (hareket açıkken)
};

// Ardışık karelerde ölçüm (plan Görev 17): tek oturum "bench", gate=advise, takip açık. Dönüş
// {"stages", "fps", "frames_sent", "seeks", "keyframe_p95", "relink_p95", "reloc_attempt_p95",
// "ir_drop_p95", "seek_p95"}; boş liste → null. Bölge verilip oturumda hareket yoksa
// std::invalid_argument.
nlohmann::json bench_sequence(Runtime& runtime, Sessions& sessions,
                              const std::vector<SequenceFrame>& frames, const SequenceOptions& opt);

}  // namespace baykus
