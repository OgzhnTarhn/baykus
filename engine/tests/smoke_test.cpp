#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "baykus/version.hpp"
#include "fixtures.hpp"

TEST(Versions, OpenCvMatchesPythonReference) { EXPECT_EQ(baykus::opencv_version(), "5.0.0"); }

TEST(Versions, OnnxRuntimeMatchesPythonReference) {
  EXPECT_EQ(baykus::onnxruntime_version(), "1.30.0");
}

TEST(Fixtures, Sha256KnownVector) {
  const std::string abc = "abc";
  EXPECT_EQ(baykus::test::sha256_hex(abc.data(), abc.size()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(Fixtures, IndexListsKinds) {
  const auto index = baykus::test::fixtures_root() / "index.json";
  BAYKUS_REQUIRE_FILES({index});
  EXPECT_FALSE(baykus::test::fixtures_of("frame").empty());
  EXPECT_EQ(baykus::test::fixtures_of("clip").size(), 3u);
}

TEST(Models, ManifestFilesPresent) {
  const auto m = baykus::test::models_root();
  BAYKUS_REQUIRE_FILES(
      {m / "e3c.onnx", m / "thresholds.json", m / "calibration.json", m / "gate_model.json"});
  SUCCEED();
}

TEST(Fixtures, RawReferenceOverride) {
  namespace bt = baykus::test;
  const auto p = std::filesystem::temp_directory_path() / "baykus_raw_ref.json";
  std::ofstream(p) << R"({"frames": {"a": [[1, 2, 3, 4, 5, 6]]}})";
  const bt::json fixture = {{"modules", {{"raw", {{0}}}}}};
  ::unsetenv("BAYKUS_RAW_REF");
  EXPECT_EQ(bt::raw_reference("a", fixture), fixture.at("modules").at("raw"));
  EXPECT_EQ(bt::raw_reference_source(), "fixture");
  ::setenv("BAYKUS_RAW_REF", p.c_str(), 1);
  EXPECT_EQ(bt::raw_reference("a", fixture), bt::json::parse("[[1, 2, 3, 4, 5, 6]]"));
  EXPECT_EQ(bt::raw_reference_source(), "same_machine");
  ::unsetenv("BAYKUS_RAW_REF");
}

// NFR-1 (kapalı ağ): ONNX Runtime 1.30 telemetrisi HTTPS ile gönderim dener; imajda kapalı olmalı
// (ORT ortamı kurulmadan önce okunur, bu yüzden Dockerfile ENV).
TEST(Offline, OrtTelemetryDisabledInImage) {
  const char* v = std::getenv("ORT_DISABLE_TELEMETRY");
  ASSERT_NE(v, nullptr);
  EXPECT_STREQ(v, "1");
}
