#include <gtest/gtest.h>

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
