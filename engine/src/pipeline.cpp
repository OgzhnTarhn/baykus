#include "baykus/pipeline.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

#include "baykus/files.hpp"
#include "baykus/health.hpp"
#include "baykus/image_io.hpp"
#include "baykus/preprocess.hpp"

namespace baykus {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

Pipeline::Pipeline(const Models& models, bool hash_tensor)
    : models_(models), detector_(models.onnx_path), hash_tensor_(hash_tensor) {}

FrameResult Pipeline::run(const std::string& rgb, const std::string& ir,
                          const std::string& gate_mode, float floor, MisalignWindow* window) {
  if (gate_mode != "off" && gate_mode != "advise" && gate_mode != "auto") {
    throw std::invalid_argument("bilinmeyen gate modu: " + gate_mode +
                                "; seçenekler: off, advise, auto");
  }
  FrameResult r;
  auto t = Clock::now();
  Pair p = decode_pair(rgb, ir);
  r.timing_ms["decode"] = ms_since(t);

  t = Clock::now();
  nlohmann::json health =
      assess(frame_features(p.bgr, p.ir), models_.thresholds, models_.gate.r_min);
  if (window != nullptr) health = window->update(health);
  if (gate_mode == "off") health["gate"] = "none";  // ölçülür; yalnız kapı kararı none sayılır
  const std::string gate = health.at("gate").get<std::string>();
  const bool applied = gate_mode == "auto" && gate != "none";
  health["applied"] = applied;
  if (applied && gate == "ir_off") p.ir = cv::Mat::zeros(p.ir.size(), p.ir.type());
  if (applied && gate == "rgb_off") p.bgr = cv::Mat::zeros(p.bgr.size(), p.bgr.type());
  r.timing_ms["health"] = ms_since(t);

  t = Clock::now();
  Letterbox lb{};
  const auto x = letterbox(stack_rgbt(p.bgr, p.ir), detector_.imgsz(), &lb);
  r.timing_ms["pre"] = ms_since(t);
  if (hash_tensor_) r.tensor_sha256 = sha256_hex(x.data(), x.size() * sizeof(float));

  t = Clock::now();
  const auto out = detector_.infer(x);
  r.timing_ms["infer"] = ms_since(t);

  t = Clock::now();
  r.dets_all = decode_rows(out, lb, floor);
  for (auto& d : r.dets_all) d.score = models_.calibration.apply(d.score_raw);
  order(r.dets_all);
  r.timing_ms["post"] = ms_since(t);
  r.health = std::move(health);
  return r;
}

}  // namespace baykus
