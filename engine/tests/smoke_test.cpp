#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "baykus/version.hpp"
#include "fixtures.hpp"

TEST(Versions, OpenCvMatchesPythonReference) { EXPECT_EQ(baykus::opencv_version(), "5.0.0"); }

TEST(Versions, OnnxRuntimeMatchesPythonReference) {
  EXPECT_EQ(baykus::onnxruntime_version(), "1.30.0");
}

namespace {

// Sabit tohumlu, köşe ve izlenebilir doku üreten size x size gri sentetik görüntü.
cv::Mat TexturedImage(int size) {
  cv::Mat img(size, size, CV_8UC1, cv::Scalar(128));
  cv::RNG rng(0x5EED);
  const int count = std::max(24, size * size / 170);
  for (int i = 0; i < count; ++i) {
    const int x = rng.uniform(0, size - 16);
    const int y = rng.uniform(0, size - 16);
    const int w = rng.uniform(6, 16);
    const int h = rng.uniform(6, 16);
    const int value = rng.uniform(0, 256);
    cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(value), cv::FILLED);
  }
  return img;
}

// cv::getBuildInformation() çıktısındaki "To be built:" satırında listelenen modül adları.
// Alt dizge araması yapılmaz: "Video I/O", "CPU/HW features" gibi satırlar yanlış pozitif verir.
std::vector<std::string> BuiltModules(const std::string& build_info) {
  const std::string prefix = "To be built:";
  std::istringstream lines(build_info);
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t first = line.find_first_not_of(" \t");
    if (first == std::string::npos) {
      continue;
    }
    const std::string trimmed = line.substr(first);
    if (trimmed.compare(0, prefix.size(), prefix) != 0) {
      continue;
    }
    std::vector<std::string> modules;
    std::istringstream names(trimmed.substr(prefix.size()));
    std::string name;
    while (names >> name) {
      modules.push_back(name);
    }
    return modules;
  }
  return {};
}

}  // namespace

// OpenCV 5 modül adları: video (optik akış), features (köşe/ORB/BFMatcher),
// geometry (benzerlik kestirimi). calib3d / features2d başlıkları kullanılmaz.
TEST(Smoke, OpenCvModulesPresent) {
  const std::vector<std::string> modules = BuiltModules(cv::getBuildInformation());
  ASSERT_FALSE(modules.empty()) << "cv::getBuildInformation() içinde 'To be built:' satırı yok";
  for (const char* required :
       {"core", "imgproc", "imgcodecs", "flann", "geometry", "features", "video"}) {
    EXPECT_NE(std::find(modules.begin(), modules.end(), required), modules.end())
        << "derlenmiş modül listesinde yok: " << required;
  }

  const cv::Mat img = TexturedImage(64);
  cv::Mat shifted;
  cv::warpAffine(img, shifted, cv::Matx23d(1, 0, 2, 0, 1, 0), img.size(), cv::INTER_LINEAR,
                 cv::BORDER_REPLICATE);

  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(img, corners, 100, 0.01, 5.0);
  ASSERT_FALSE(corners.empty());

  std::vector<cv::Point2f> src;
  for (const cv::Point2f& p : corners) {
    if (p.x >= 10.0F && p.x <= 54.0F && p.y >= 10.0F && p.y <= 54.0F) {
      src.push_back(p);
    }
  }
  ASSERT_FALSE(src.empty());

  std::vector<cv::Point2f> tracked;
  std::vector<unsigned char> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(img, shifted, src, tracked, status, err, cv::Size(21, 21), 3);
  ASSERT_EQ(status.size(), src.size());
  int tracked_ok = 0;
  for (const unsigned char s : status) {
    tracked_ok += (s != 0) ? 1 : 0;
  }
  EXPECT_GT(tracked_ok, 0);

  std::vector<cv::Point2f> from;
  std::vector<cv::Point2f> to;
  for (std::size_t i = 0; i < src.size(); ++i) {
    if (status[i] != 0) {
      from.push_back(src[i]);
      to.push_back(tracked[i]);
    }
  }
  ASSERT_GE(from.size(), 3u);
  const cv::Mat affine = cv::estimateAffinePartial2D(from, to);
  ASSERT_FALSE(affine.empty());
  EXPECT_EQ(affine.rows, 2);
  EXPECT_EQ(affine.cols, 3);
  EXPECT_NEAR(affine.at<double>(0, 2), 2.0, 0.5);
  EXPECT_NEAR(affine.at<double>(1, 2), 0.0, 0.5);

  // ORB varsayılan kenar payı 31 px: 64x64 görüntüde tanımlayıcı çıkmayabilir, 256x256 kullan.
  const cv::Mat orb_img = TexturedImage(256);
  cv::Mat orb_shifted;
  cv::warpAffine(orb_img, orb_shifted, cv::Matx23d(1, 0, 2, 0, 1, 0), orb_img.size(),
                 cv::INTER_LINEAR, cv::BORDER_REPLICATE);

  cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
  std::vector<cv::KeyPoint> keypoints_a;
  std::vector<cv::KeyPoint> keypoints_b;
  cv::Mat descriptors_a;
  cv::Mat descriptors_b;
  orb->detectAndCompute(orb_img, cv::noArray(), keypoints_a, descriptors_a);
  orb->detectAndCompute(orb_shifted, cv::noArray(), keypoints_b, descriptors_b);
  ASSERT_FALSE(descriptors_a.empty());
  ASSERT_FALSE(descriptors_b.empty());

  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  std::vector<cv::DMatch> matches;
  matcher.match(descriptors_a, descriptors_b, matches);
  EXPECT_FALSE(matches.empty());
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
