// Derleme bilgisi (JSON): scripts/opencv_build_info.py Python tarafıyla yan yana kaydeder.
#include <iostream>

#include <nlohmann/json.hpp>

#include "baykus/version.hpp"

int main() {
  nlohmann::json out = {
      {"opencv_version", baykus::opencv_version()},
      {"onnxruntime_version", baykus::onnxruntime_version()},
      {"opencv_build_information", baykus::opencv_build_information()},
  };
  std::cout << out.dump(2) << std::endl;
  return 0;
}
