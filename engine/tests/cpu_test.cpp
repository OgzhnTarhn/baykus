// CPU iş parçacığı yapılandırması: BAYKUS_CV_THREADS / BAYKUS_ORT_THREADS / BAYKUS_ORT_SPINNING.
#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <stdexcept>

#include "baykus/cpu.hpp"

namespace {

constexpr const char* kCvThreads = "BAYKUS_CV_THREADS";
constexpr const char* kOrtThreads = "BAYKUS_ORT_THREADS";
constexpr const char* kOrtSpinning = "BAYKUS_ORT_SPINNING";

void clear_env() {
  unsetenv(kCvThreads);
  unsetenv(kOrtThreads);
  unsetenv(kOrtSpinning);
}

}  // namespace

TEST(CpuCfg, UnsetMeansDefaults) {
  clear_env();
  const baykus::CpuCfg cfg = baykus::cpu_cfg_from_env();
  EXPECT_FALSE(cfg.cv_threads.has_value());
  EXPECT_FALSE(cfg.ort_threads.has_value());
  EXPECT_FALSE(cfg.ort_spinning.has_value());
  clear_env();
}

TEST(CpuCfg, ParsesValues) {
  clear_env();
  setenv(kCvThreads, "1", 1);
  setenv(kOrtThreads, "8", 1);
  setenv(kOrtSpinning, "0", 1);
  {
    const baykus::CpuCfg cfg = baykus::cpu_cfg_from_env();
    ASSERT_TRUE(cfg.cv_threads.has_value());
    EXPECT_EQ(*cfg.cv_threads, 1);
    ASSERT_TRUE(cfg.ort_threads.has_value());
    EXPECT_EQ(*cfg.ort_threads, 8);
    ASSERT_TRUE(cfg.ort_spinning.has_value());
    EXPECT_FALSE(*cfg.ort_spinning);
  }
  setenv(kOrtSpinning, "1", 1);
  {
    const baykus::CpuCfg cfg = baykus::cpu_cfg_from_env();
    ASSERT_TRUE(cfg.ort_spinning.has_value());
    EXPECT_TRUE(*cfg.ort_spinning);
  }
  clear_env();
}

TEST(CpuCfg, EmptyStringMeansDefault) {
  clear_env();
  setenv(kCvThreads, "", 1);
  const baykus::CpuCfg cfg = baykus::cpu_cfg_from_env();
  EXPECT_FALSE(cfg.cv_threads.has_value());
  clear_env();
}

TEST(CpuCfg, RejectsBadValues) {
  clear_env();

  setenv(kCvThreads, "abc", 1);
  EXPECT_THROW(baykus::cpu_cfg_from_env(), std::invalid_argument);
  setenv(kCvThreads, "-1", 1);
  EXPECT_THROW(baykus::cpu_cfg_from_env(), std::invalid_argument);
  setenv(kCvThreads, "0", 1);
  EXPECT_THROW(baykus::cpu_cfg_from_env(), std::invalid_argument);
  setenv(kCvThreads, "2x", 1);
  EXPECT_THROW(baykus::cpu_cfg_from_env(), std::invalid_argument);
  unsetenv(kCvThreads);

  setenv(kOrtSpinning, "yes", 1);
  EXPECT_THROW(baykus::cpu_cfg_from_env(), std::invalid_argument);
  setenv(kOrtSpinning, "2", 1);
  EXPECT_THROW(baykus::cpu_cfg_from_env(), std::invalid_argument);
  unsetenv(kOrtSpinning);

  clear_env();
}
