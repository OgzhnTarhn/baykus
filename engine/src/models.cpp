#include "baykus/models.hpp"

#include <filesystem>
#include <stdexcept>

#include "baykus/files.hpp"

namespace baykus {

namespace fs = std::filesystem;

Models load_models(const std::string& dir, const std::string& manifest_path) {
  const fs::path d(dir);
  for (const char* name : {"e3c.onnx", "thresholds.json", "calibration.json", "gate_model.json"}) {
    if (!fs::is_regular_file(d / name)) {
      throw std::runtime_error(dir + ": eksik model dosyası " + name);
    }
  }
  auto manifest = nlohmann::json::parse(read_bytes(manifest_path));
  for (const auto& [name, rec] : manifest.at("files").items()) {
    const fs::path p = d / name;
    if (!fs::is_regular_file(p)) throw std::runtime_error(p.string() + " yok");
    if (sha256_hex(read_bytes(p.string())) != rec.at("sha256").get<std::string>()) {
      throw std::runtime_error(name + ": sha256 manifestle uyuşmuyor (" + manifest_path + ")");
    }
  }
  auto th = Thresholds::load((d / "thresholds.json").string());
  auto gate = GateModel::load((d / "gate_model.json").string());
  if (gate.thresholds_sha256 != th.sha256) {
    throw std::runtime_error(
        "gate_model.json başka bir thresholds.json'dan üretilmiş (sha256 farklı)");
  }
  return Models{(d / "e3c.onnx").string(), std::move(th),
                Calibration::load((d / "calibration.json").string()), std::move(gate),
                std::move(manifest)};
}

}  // namespace baykus
