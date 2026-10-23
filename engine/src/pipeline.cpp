#include "baykus/pipeline.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

#include "baykus/files.hpp"
#include "baykus/geometry.hpp"
#include "baykus/health.hpp"
#include "baykus/image_io.hpp"
#include "baykus/preprocess.hpp"
#include "baykus/tiles.hpp"

namespace baykus {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

bool has_lost(const nlohmann::json& flags) {
  for (const auto& f : flags) {
    if (f.get<std::string>() == "lost") return true;
  }
  return false;
}

}  // namespace

Pipeline::Pipeline(const Models& models, bool hash_tensor)
    : models_(models), detector_(models.onnx_path), hash_tensor_(hash_tensor) {}

FrameResult Pipeline::run(const std::string& rgb, const std::string& ir,
                          const std::string& gate_mode, float floor, MisalignWindow* window,
                          FrozenDetector* frozen, int tiles) {
  const auto t = Clock::now();
  Pair p = decode_pair(rgb, ir);
  return run(std::move(p), ms_since(t), gate_mode, floor, window, frozen, tiles);
}

FrameResult Pipeline::run(Pair p, double decode_ms, const std::string& gate_mode, float floor,
                          MisalignWindow* window, FrozenDetector* frozen, int tiles) {
  if (gate_mode != "off" && gate_mode != "advise" && gate_mode != "auto") {
    throw std::invalid_argument("bilinmeyen gate modu: " + gate_mode +
                                "; seçenekler: off, advise, auto");
  }
  if (tiles != 1 && tiles != 4) {
    throw std::invalid_argument("bilinmeyen tiles modu: " + std::to_string(tiles) +
                                "; seçenekler: 1, 4");
  }
  FrameResult r;
  r.timing_ms["decode"] = decode_ms;

  auto t = Clock::now();
  nlohmann::json health =
      assess(frame_features(p.bgr, p.ir), models_.thresholds, models_.gate.r_min);
  if (window != nullptr) health = window->update(health);
  if (frozen != nullptr) {
    r.rgb_half = half_gray(p.bgr);
    r.ir_half = half_gray(p.ir);
    const nlohmann::json fz =
        frozen->update(r.rgb_half, r.ir_half, has_lost(health.at("rgb").at("flags")),
                       has_lost(health.at("ir").at("flags")));
    for (const char* ch : {"rgb", "ir"}) {
      health[ch]["frozen"] = fz[ch]["frozen"];
      health[ch]["diff_mad"] = fz[ch]["diff_mad"];
    }
    r.frozen = fz;
  }
  if (gate_mode == "off") health["gate"] = "none";  // ölçülür; yalnız kapı kararı none sayılır
  const std::string gate = health.at("gate").get<std::string>();
  const bool applied = gate_mode == "auto" && gate != "none";
  health["applied"] = applied;
  if (applied && gate == "ir_off") p.ir = cv::Mat::zeros(p.ir.size(), p.ir.type());
  if (applied && gate == "rgb_off") p.bgr = cv::Mat::zeros(p.bgr.size(), p.bgr.type());
  r.timing_ms["health"] = ms_since(t);

  if (tiles == 1) {
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
  } else {
    const std::vector<cv::Rect> wins = tile_windows(p.bgr.cols, p.bgr.rows);
    std::vector<std::vector<Det>> per_tile;
    std::vector<float> tensors;  // hash_tensor_ açıksa dört tensörün baytları sırayla
    double pre_ms = 0.0, infer_ms = 0.0, post_ms = 0.0;
    for (const auto& rect : wins) {
      auto tp = Clock::now();
      const cv::Mat rgb_c = p.bgr(rect).clone();
      const cv::Mat ir_c = p.ir(rect).clone();
      Letterbox lb{};
      const auto x = letterbox(stack_rgbt(rgb_c, ir_c), detector_.imgsz(), &lb);
      pre_ms += ms_since(tp);
      if (hash_tensor_) tensors.insert(tensors.end(), x.data(), x.data() + x.size());

      tp = Clock::now();
      const auto out = detector_.infer(x);
      infer_ms += ms_since(tp);

      tp = Clock::now();
      std::vector<Det> dets = decode_rows(out, lb, floor);
      for (auto& d : dets) d.score = models_.calibration.apply(d.score_raw);
      post_ms += ms_since(tp);
      per_tile.push_back(std::move(dets));
    }
    r.timing_ms["pre"] = pre_ms;
    r.timing_ms["infer"] = infer_ms;
    r.timing_ms["post"] = post_ms;
    if (hash_tensor_) {
      r.tensor_sha256 = sha256_hex(tensors.data(), tensors.size() * sizeof(float));
    }
    r.dets_all = merge_tiles(per_tile, wins);
  }
  r.health = std::move(health);
  return r;
}

}  // namespace baykus
