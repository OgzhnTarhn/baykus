#pragma once

// Eşlik fikstürlerine erişim (Faz 3 plan Görev 16). Kök: BAYKUS_FIXTURES (varsayılan /fixtures),
// modeller: BAYKUS_MODELS (varsayılan /models). BAYKUS_REQUIRE_FIXTURES=1 iken eksik dosya testi
// başarısız kılar (CI'da kabul testleri atlanamaz); değilse test atlanır (yerel hızlı döngü).

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "baykus/detections.hpp"

namespace baykus::test {

using json = nlohmann::json;

std::filesystem::path fixtures_root();
std::filesystem::path models_root();
bool fixtures_required();

// Eksik yolların açıklaması; hepsi varsa boş.
std::string missing(const std::vector<std::filesystem::path>& paths);

json load_index();
std::vector<json> fixtures_of(const std::string& kind);
std::string read_file(const std::filesystem::path& path);
json read_json(const std::filesystem::path& path);
std::string sha256_hex(const void* data, std::size_t size);

// Python `asdict(Det)` listesi (index, cls_id, score_raw, score, box) → Det.
std::vector<Det> dets_from_json(const json& j);

}  // namespace baykus::test

// Test gövdesinin başında: dosyalar yoksa zorunluysa FAIL, değilse SKIP.
#define BAYKUS_REQUIRE_FILES(...)                                          \
  do {                                                                     \
    const std::string baykus_missing_ = ::baykus::test::missing(__VA_ARGS__); \
    if (!baykus_missing_.empty()) {                                        \
      if (::baykus::test::fixtures_required()) {                           \
        FAIL() << "fikstür eksik: " << baykus_missing_;                    \
      } else {                                                             \
        GTEST_SKIP() << "fikstür yok: " << baykus_missing_;                \
      }                                                                    \
    }                                                                      \
  } while (0)
