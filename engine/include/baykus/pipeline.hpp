#pragma once

// Motorun işlem sırası (Faz 3 tasarımı §4.1; Python `baykus.engine_ref.pipeline.Pipeline`):
// çöz → sağlık (her modda; oturum varsa zaman kuralı) → kapı (`auto` ve karar ≠ none ise kanal
// sıfırlanır) → 4 kanal → letterbox → ORT → çözümleme + taban süzgeci (ham ≥ floor) → kalibre skor
// → sıralama (kalibre azalan, ham azalan, satır indeksi artan).
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "baykus/detections.hpp"
#include "baykus/detector.hpp"
#include "baykus/models.hpp"
#include "baykus/temporal.hpp"

namespace baykus {

struct FrameResult {
  std::vector<Det> dets_all;                // taban süzgecinden geçenler, kalibre ve sıralı
  nlohmann::json health;                    // `applied` dahil
  std::string tensor_sha256;                // letterbox tensörü; yalnız hash_tensor açıkken
  std::map<std::string, double> timing_ms;  // decode, health, pre, infer, post
};

class Pipeline {
 public:
  // hash_tensor: her karede tensör sha256'sı (eşlik testleri); sunucuda kapalı (gecikme).
  explicit Pipeline(const Models& models, bool hash_tensor = false);

  // gate_mode: off | advise | auto (değilse std::invalid_argument). window: oturumun zaman kuralı
  // penceresi; nullptr = oturumsuz (`misaligned` = `misaligned_frame`).
  FrameResult run(const std::string& rgb, const std::string& ir, const std::string& gate_mode,
                  float floor, MisalignWindow* window);

  int imgsz() const { return detector_.imgsz(); }
  int channels() const { return detector_.channels(); }

 private:
  Models models_;
  Detector detector_;
  bool hash_tensor_;
};

}  // namespace baykus
