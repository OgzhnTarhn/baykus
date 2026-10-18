#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace baykus {

// ONNX Runtime oturumu (CPU). Giriş adı, şekli ve tipi modelden okunur; çıktı uçtan uca [1, 300, 6].
// threads = 0: ORT varsayılanı (Python OrtBackend gibi).
class Detector {
 public:
  explicit Detector(const std::string& onnx_path, int threads = 0);

  std::vector<float> infer(const std::vector<float>& chw);
  int imgsz() const { return static_cast<int>(input_shape_[2]); }
  int channels() const { return static_cast<int>(input_shape_[1]); }

 private:
  Ort::Env env_;
  Ort::Session session_;
  std::string input_name_;
  std::string output_name_;
  std::vector<int64_t> input_shape_;
};

}  // namespace baykus
