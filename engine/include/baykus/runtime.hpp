#pragma once

// Ortak yürütme yolu (Faz 3 tasarımı §4.1, işlem sırası 1-5): sunucu, `baykus_bench` ve
// `baykus_detect` yalnız bunu kullanır. Oturum varsa zaman kuralı ve takip o oturumun durumuyla;
// takip katmanının bütün tespitleri (ham ≥ 0,1, `conf`'tan bağımsız) takipçiye, yanıt gösterim
// katmanı (kalibre ≥ conf) ve o tespitlerin kimlikleri. Çıkarım tek işçide (aynı anda tek ORT
// çağrısı; gecikme ölçümü kararlı).
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "baykus/detections.hpp"
#include "baykus/models.hpp"
#include "baykus/pipeline.hpp"
#include "baykus/sessions.hpp"
#include "baykus/tracker.hpp"

namespace baykus {

struct Request {
  std::string rgb;
  std::string ir;
  std::string gate = "advise";
  std::string session;  // boş: oturumsuz (tek kare; `misaligned` = `misaligned_frame`)
  double conf = 0.25;
  bool track = true;
  float floor = 0.1f;
};

struct Response {
  std::vector<Det> dets;       // gösterim katmanı, sıralı
  std::vector<int> track_ids;  // dets ile aynı sırada; -1 = kimlik yok (API'de null)
  nlohmann::json health;
  std::map<std::string, double> timing_ms;  // decode, health, pre, infer, post, track, total
  bool session_new = false;
};

class Runtime {
 public:
  Runtime(const Models& models, const TrackerConfig& tracker_cfg, Sessions& sessions);

  // Geçersiz istek (conf [0, 1] dışı, bilinmeyen gate, oturumsuz takip, çözülemeyen görüntü):
  // std::invalid_argument.
  Response process(const Request& req);

  int imgsz() const { return pipeline_.imgsz(); }
  int channels() const { return pipeline_.channels(); }

 private:
  Pipeline pipeline_;
  TrackerConfig tracker_cfg_;
  Sessions& sessions_;
  std::mutex infer_mutex_;
};

// Bir dizi klasörü (rgb/NNNNNN.jpg, ir/NNNNNN.png, index.json) → Python `baykus-engine detect`
// biçiminde kayıt (seq, split, floor, gate, frames[frame, dets[index, cls_id, score_raw, score,
// box]]). Takip kapalı, bütün katman (conf 0). Test dizisi allow_test olmadan
// std::invalid_argument.
nlohmann::json detect_sequence(Runtime& runtime, const std::string& seq_dir, double floor,
                               const std::string& gate, bool allow_test);

}  // namespace baykus
