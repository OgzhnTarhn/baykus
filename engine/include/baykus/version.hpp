#pragma once

#include <string>

namespace baykus {

// Derlemeye giren kütüphane sürümleri (Python referansıyla eşleşmeli: OpenCV 5.0.0, ORT 1.30.0).
std::string opencv_version();
std::string onnxruntime_version();
std::string opencv_build_information();

}  // namespace baykus
