#include "baykus/detector.hpp"

#include <opencv2/core.hpp>

#include <stdexcept>

#include "baykus/cpu.hpp"
#include "baykus/detections.hpp"

namespace baykus {

namespace {

// Gecikme ayarı (plan F4 Görev 17): BAYKUS_* değişkenleri yoksa davranış bugünküyle aynı.
Ort::SessionOptions options(int threads) {
  const CpuCfg cpu = cpu_cfg_from_env();
  if (cpu.cv_threads) cv::setNumThreads(*cpu.cv_threads);
  Ort::SessionOptions opts;
  if (threads > 0) {
    opts.SetIntraOpNumThreads(threads);
  } else if (cpu.ort_threads) {
    opts.SetIntraOpNumThreads(*cpu.ort_threads);
  }
  if (cpu.ort_spinning) {
    opts.AddConfigEntry("session.intra_op.allow_spinning", *cpu.ort_spinning ? "1" : "0");
  }
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);  // Python varsayılanı
  return opts;
}

// NFR-1: telemetri olayları kapalı (imajda ayrıca ORT_DISABLE_TELEMETRY=1; ortam kurulurken okunur)
Ort::Env make_env() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "baykus");
  env.DisableTelemetryEvents();
  return env;
}

}  // namespace

Detector::Detector(const std::string& onnx_path, int threads)
    : env_(make_env()), session_(env_, onnx_path.c_str(), options(threads)) {
  Ort::AllocatorWithDefaultOptions alloc;
  input_name_ = session_.GetInputNameAllocated(0, alloc).get();
  output_name_ = session_.GetOutputNameAllocated(0, alloc).get();
  // TypeInfo yaşamalı: GetTensorTypeAndShapeInfo() ona bağlı bir görünüm döner
  const auto type_info = session_.GetInputTypeInfo(0);
  const auto info = type_info.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("model girişi float32 değil (sevk edilen E3c ONNX FP32 bekleniyor)");
  }
  input_shape_ = info.GetShape();
  if (input_shape_.size() != 4) {
    throw std::runtime_error("model girişi [1, C, H, W] değil");
  }
  input_shape_[0] = 1;
}

std::vector<float> Detector::infer(const std::vector<float>& chw) {
  const std::size_t want =
      static_cast<std::size_t>(input_shape_[1] * input_shape_[2] * input_shape_[3]);
  if (chw.size() != want) {
    throw std::invalid_argument("giriş tensörü boyutu modelle uyuşmuyor");
  }
  const auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(chw.data()), chw.size(),
                                               input_shape_.data(), input_shape_.size());
  const char* in_names[] = {input_name_.c_str()};
  const char* out_names[] = {output_name_.c_str()};
  auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
  const auto n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
  if (n != static_cast<std::size_t>(kRows) * kCols) {
    throw std::runtime_error("model çıktısı [1, 300, 6] değil");
  }
  const float* p = outputs[0].GetTensorData<float>();
  return std::vector<float>(p, p + n);
}

}  // namespace baykus
