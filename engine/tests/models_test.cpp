// Model dizini: manifest sha256 denetimi (Faz 3 tasarımı §4.1: eksik ya da farklıysa açılmaz).
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "baykus/models.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

fs::path manifest_path() {
  const char* c = std::getenv("BAYKUS_CONFIGS");
  return fs::path(c && *c ? c : "/configs") / "engine" / "models.json";
}

fs::path temp_dir(const std::string& name) {
  const auto d = fs::temp_directory_path() / ("baykus_" + name);
  fs::remove_all(d);
  fs::create_directories(d);
  return d;
}

fs::path write_json(const fs::path& p, const json& j) {
  std::ofstream(p) << j.dump();
  return p;
}

// İleti beklenen dosya adını içermeli: kullanıcı neyin eksik/bozuk olduğunu görsün.
template <typename F>
void expect_error_naming(F f, const std::string& name) {
  try {
    f();
    ADD_FAILURE() << "hata bekleniyordu: " << name;
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(name), std::string::npos) << e.what();
  }
}

}  // namespace

class ModelsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto m = bt::models_root();
    BAYKUS_REQUIRE_FILES({m / "e3c.onnx", m / "thresholds.json", m / "calibration.json",
                          m / "gate_model.json", manifest_path()});
  }
};

TEST_F(ModelsTest, LoadsWithManifest) {
  const auto m = baykus::load_models(bt::models_root().string(), manifest_path().string());
  EXPECT_EQ(m.onnx_path, (bt::models_root() / "e3c.onnx").string());
  EXPECT_EQ(m.gate.thresholds_sha256, m.thresholds.sha256);
  EXPECT_EQ(m.manifest.at("files").at("thresholds.json").at("sha256"), m.thresholds.sha256);
  EXPECT_FALSE(m.calibration.x().empty());
}

TEST_F(ModelsTest, ShaMismatchThrows) {
  json man = bt::read_json(manifest_path());
  man["files"]["thresholds.json"]["sha256"] = std::string(64, '0');
  const auto p = write_json(temp_dir("sha") / "models.json", man);
  expect_error_naming([&] { baykus::load_models(bt::models_root().string(), p.string()); },
                      "thresholds.json");
}

TEST(Models, MissingFileThrows) {
  const auto d = temp_dir("missing");
  const auto p = write_json(d / "models.json",
                            {{"files", {{"e3c.onnx", {{"sha256", std::string(64, '0')}}}}}});
  expect_error_naming([&] { baykus::load_models(d.string(), p.string()); }, "e3c.onnx");
}
