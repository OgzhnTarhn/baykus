// Kabul 2: C++ letterbox tensörü Python `baykus.edge.frames.preprocess` ile bayt bayt aynı
// (fixtures/expected/<ad>.json "modules.letterbox"; özgün pikseller, kapı off).
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "baykus/image_io.hpp"
#include "baykus/preprocess.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;

TEST(Letterbox, MatchesPythonSha) {
  const auto root = bt::fixtures_root();
  BAYKUS_REQUIRE_FILES({root / "index.json"});
  const auto frames = bt::fixtures_of("frame");
  ASSERT_FALSE(frames.empty());
  int resized = 0;
  for (const auto& e : frames) {
    SCOPED_TRACE(e.at("name").get<std::string>());
    const auto exp = bt::read_json(root / e.at("path").get<std::string>());
    const auto& in = exp.at("inputs");
    auto p = baykus::decode_pair(bt::read_file(root / in.at("rgb").get<std::string>()),
                                 bt::read_file(root / in.at("ir").get<std::string>()));
    baykus::Letterbox lb{};
    const std::vector<float> x = baykus::letterbox(baykus::stack_rgbt(p.bgr, p.ir), 640, &lb);
    const auto& want = exp.at("modules").at("letterbox");
    EXPECT_EQ(x.size(), 4u * 640u * 640u);
    EXPECT_EQ(lb.gain, want.at("gain").get<double>());
    EXPECT_EQ(lb.pad_x, want.at("pad_x").get<int>());
    EXPECT_EQ(lb.pad_y, want.at("pad_y").get<int>());
    EXPECT_EQ(bt::sha256_hex(x.data(), x.size() * sizeof(float)), want.at("sha256").get<std::string>());
    resized += lb.gain != 1.0;
  }
  EXPECT_GE(resized, 2);  // resize_800x600 ve tiny_33x17: yeniden boyutlandırma yolu sınandı
}

TEST(Letterbox, PadAndGainForWideImage) {
  cv::Mat img(100, 200, CV_8UC4, cv::Scalar(1, 2, 3, 4));
  baykus::Letterbox lb{};
  const auto x = baykus::letterbox(img, 640, &lb);
  EXPECT_DOUBLE_EQ(lb.gain, 3.2);
  EXPECT_EQ(lb.pad_x, 0);
  EXPECT_EQ(lb.pad_y, 160);
  EXPECT_FLOAT_EQ(x[0], 114.0f / 255.0f);  // dolgu
}
