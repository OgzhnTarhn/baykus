#include "baykus/cpu.hpp"

#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace baykus {
namespace {

using json = nlohmann::json;

// Değişken yoksa ya da boşsa nullopt; aksi halde tüm dize tam sayı (>= 1) olmalı.
std::optional<int> read_threads(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return std::nullopt;
  const std::string text(raw);
  int value = 0;
  const char* end = text.data() + text.size();
  const auto res = std::from_chars(text.data(), end, value);
  if (res.ec != std::errc() || res.ptr != end || value < 1) {
    throw std::invalid_argument(std::string(name) + ": tam sayı (>= 1) olmalı, alınan '" + text +
                                "'");
  }
  return value;
}

// Yalnız "0"/"1"; yok/boş → nullopt.
std::optional<bool> read_spinning(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return std::nullopt;
  const std::string text(raw);
  if (text == "0") return false;
  if (text == "1") return true;
  throw std::invalid_argument(std::string(name) + ": '0' ya da '1' olmalı, alınan '" + text + "'");
}

}  // namespace

CpuCfg cpu_cfg_from_env() {
  CpuCfg cfg;
  cfg.cv_threads = read_threads("BAYKUS_CV_THREADS");
  cfg.ort_threads = read_threads("BAYKUS_ORT_THREADS");
  cfg.ort_spinning = read_spinning("BAYKUS_ORT_SPINNING");
  return cfg;
}

json cpu_cfg_json(const CpuCfg& cfg) {
  return json{{"cv_threads", cfg.cv_threads ? json(*cfg.cv_threads) : json(nullptr)},
              {"ort_threads", cfg.ort_threads ? json(*cfg.ort_threads) : json(nullptr)},
              {"ort_spinning", cfg.ort_spinning ? json(*cfg.ort_spinning) : json(nullptr)}};
}

}  // namespace baykus
