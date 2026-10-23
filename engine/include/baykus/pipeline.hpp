#pragma once

// Motorun işlem sırası (Faz 3 tasarımı §4.1; Python `baykus.engine_ref.pipeline.Pipeline`):
// çöz → sağlık (her modda; oturum varsa zaman kuralı) → kapı (`auto` ve karar ≠ none ise kanal
// sıfırlanır) → 4 kanal → letterbox → ORT → çözümleme + taban süzgeci (ham ≥ floor) → kalibre skor
// → sıralama (kalibre azalan, ham azalan, satır indeksi artan).
// tiles: 1 = tek kare (bugünkü yol), 4 = 2 × 2 pencere (tile_windows + merge_tiles).
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "baykus/detections.hpp"
#include "baykus/detector.hpp"
#include "baykus/frozen.hpp"
#include "baykus/image_io.hpp"
#include "baykus/models.hpp"
#include "baykus/temporal.hpp"
#include "baykus/tiles.hpp"

namespace baykus {

struct FrameResult {
  std::vector<Det> dets_all;                // taban süzgecinden geçenler, kalibre ve sıralı
  nlohmann::json health;                    // `applied` dahil
  std::string tensor_sha256;                // letterbox tensörü; yalnız hash_tensor açıkken
  std::map<std::string, double> timing_ms;  // decode, health, pre, infer, post
  cv::Mat rgb_half;                         // yarı çözünürlük gri (kapıdan önce); frozen yoksa boş
  cv::Mat ir_half;
  nlohmann::json frozen;  // FrozenDetector çıktısı; frozen yoksa null
};

class Pipeline {
 public:
  // hash_tensor: her karede tensör sha256'sı (eşlik testleri); sunucuda kapalı (gecikme).
  explicit Pipeline(const Models& models, bool hash_tensor = false);

  // gate_mode: off | advise | auto (değilse std::invalid_argument). window: oturumun zaman kuralı
  // penceresi; nullptr = oturumsuz (`misaligned` = `misaligned_frame`).
  FrameResult run(const std::string& rgb, const std::string& ir, const std::string& gate_mode,
                  float floor, MisalignWindow* window, FrozenDetector* frozen = nullptr,
                  int tiles = 1);
  // Önceden çözülmüş çiftle (çözme süresi `decode_ms` olarak kayda geçer). Runtime çözmeyi oturum
  // açılmadan önce yapar: bozuk görüntü oturum tablosunu değiştirmez.
  FrameResult run(Pair p, double decode_ms, const std::string& gate_mode, float floor,
                  MisalignWindow* window, FrozenDetector* frozen = nullptr, int tiles = 1);

  int imgsz() const { return detector_.imgsz(); }
  int channels() const { return detector_.channels(); }

 private:
  Models models_;
  Detector detector_;
  bool hash_tensor_;
};

}  // namespace baykus
