#pragma once

// Model dizini (Faz 3 tasarımı §4.1): ONNX + eşikler + kalibrasyon tablosu + kapı modeli; Python
// `baykus.engine_ref.models.load_models`.
#include <nlohmann/json.hpp>
#include <string>

#include "baykus/calibration.hpp"
#include "baykus/health.hpp"

namespace baykus {

struct Models {
  std::string onnx_path;
  Thresholds thresholds;
  Calibration calibration;
  GateModel gate;
  nlohmann::json manifest;
};

// `dir` altındaki dört dosya manifestteki (`configs/engine/models.json`) sha256'larla denetlenir.
// Eksik ya da farklı dosya, ya da başka bir eşik dosyasından üretilmiş kapı modeli:
// std::runtime_error (dosya adıyla).
Models load_models(const std::string& dir, const std::string& manifest_path);

}  // namespace baykus
