#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "baykus/image_io.hpp"

namespace {

std::string encode(const std::string& ext, const cv::Mat& img) {
  std::vector<uchar> buf;
  EXPECT_TRUE(cv::imencode(ext, img, buf));
  return std::string(buf.begin(), buf.end());
}

cv::Mat pattern(int rows, int cols, int type) {
  cv::Mat m(rows, cols, type);
  cv::randu(m, 0, 255);
  return m;
}

}  // namespace

TEST(ImageIo, Gray3ChannelIr) {
  cv::Mat g = pattern(64, 80, CV_8UC1);
  cv::Mat g3;
  cv::merge(std::vector<cv::Mat>{g, g, g}, g3);
  auto p = baykus::decode_pair(encode(".png", pattern(64, 80, CV_8UC3)), encode(".png", g3));
  ASSERT_EQ(p.ir.type(), CV_8UC1);
  EXPECT_EQ(cv::countNonZero(p.ir != g), 0);
  EXPECT_EQ(p.bgr.type(), CV_8UC3);
}

TEST(ImageIo, SixteenBitIrBecomesEightBit) {
  cv::Mat g16(32, 32, CV_16UC1, cv::Scalar(40000));
  auto p = baykus::decode_pair(encode(".png", pattern(32, 32, CV_8UC3)), encode(".png", g16));
  EXPECT_EQ(p.ir.type(), CV_8UC1);
}

TEST(ImageIo, SizeMismatchThrows) {
  EXPECT_THROW(baykus::decode_pair(encode(".png", pattern(64, 80, CV_8UC3)),
                                   encode(".png", pattern(60, 80, CV_8UC1))),
               std::invalid_argument);
}

TEST(ImageIo, CorruptPngThrows) {
  std::string ir = encode(".png", pattern(64, 80, CV_8UC1));
  ir.resize(ir.size() / 3);
  EXPECT_THROW(baykus::decode_pair(encode(".png", pattern(64, 80, CV_8UC3)), ir),
               std::invalid_argument);
  EXPECT_THROW(baykus::decode_pair("", encode(".png", pattern(8, 8, CV_8UC1))),
               std::invalid_argument);
}

TEST(ImageIo, StackRgbtOrder) {
  cv::Mat bgr(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));  // B, G, R
  cv::Mat ir(2, 2, CV_8UC1, cv::Scalar(40));
  cv::Mat x = baykus::stack_rgbt(bgr, ir);
  ASSERT_EQ(x.type(), CV_8UC4);
  const auto v = x.at<cv::Vec4b>(1, 1);
  EXPECT_EQ(v[0], 30);
  EXPECT_EQ(v[1], 20);
  EXPECT_EQ(v[2], 10);
  EXPECT_EQ(v[3], 40);
}
