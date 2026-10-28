#pragma once

#include <nlohmann/json.hpp>

#include <optional>

namespace baykus {

// Gecikme ayarı (plan F4 Görev 17): sonuçları değiştirmez, yalnız iş parçacığı sayısını etkiler;
// ayarsız (değişken yok ya da boş) = kütüphane varsayılanı.
struct CpuCfg {
  std::optional<int> cv_threads;     // BAYKUS_CV_THREADS: cv::setNumThreads
  std::optional<int> ort_threads;    // BAYKUS_ORT_THREADS: ORT intra-op iş parçacığı
  std::optional<bool> ort_spinning;  // BAYKUS_ORT_SPINNING: ORT intra-op dönmesi
};

// Ortamdan okur: yok/boş → nullopt; tam sayı (tüm dize, >= 1) değilse std::invalid_argument
// (mesajda değişken adı). BAYKUS_ORT_SPINNING yalnız "0"/"1".
CpuCfg cpu_cfg_from_env();

// {"cv_threads": int|null, "ort_threads": int|null, "ort_spinning": bool|null}
nlohmann::json cpu_cfg_json(const CpuCfg& cfg);

}  // namespace baykus
