#pragma once

// Ortak yürütme yolu (Faz 3 tasarımı §4.1, işlem sırası 1-5): sunucu, `baykus_bench` ve
// `baykus_detect` yalnız bunu kullanır. Oturum varsa zaman kuralı ve takip o oturumun durumuyla;
// takip katmanının bütün tespitleri (ham ≥ 0,1, `conf`'tan bağımsız) takipçiye, yanıt gösterim
// katmanı (kalibre ≥ conf) ve o tespitlerin kimlikleri. Çıkarım tek işçide (aynı anda tek ORT
// çağrısı; gecikme ölçümü kararlı).
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "baykus/detections.hpp"
#include "baykus/ego.hpp"
#include "baykus/models.hpp"
#include "baykus/motion.hpp"
#include "baykus/pipeline.hpp"
#include "baykus/sessions.hpp"
#include "baykus/tracker.hpp"
#include "baykus/zones.hpp"

namespace baykus {

struct Request {
  std::string rgb;
  std::string ir;
  std::string gate = "advise";
  std::string session;  // boş: oturumsuz (tek kare; `misaligned` = `misaligned_frame`)
  double conf = 0.25;
  bool track = true;
  float floor = 0.1f;
  std::optional<int> frame;  // ego'ya geçen kare numarası
  std::optional<int> focus;  // iz özeti kimliği; yanıtta "focus_track" (yoksa null)
  int tiles = 1;             // parçalı çıkarım: 1 (tam kare) ya da 4
};

struct Response {
  std::vector<Det> dets;         // gösterim katmanı, sıralı
  std::vector<int> track_ids;    // dets ile aynı sırada; -1 = kimlik yok (API'de null)
  std::vector<TrackOut> tracks;  // takip katmanı, kimliği olanlar (API kimliği)
  std::optional<EgoResult> ego;
  std::map<int, TrackMotion> motion;                    // anahtar API kimliği
  std::vector<ZoneEvent> events;                        // bu karenin bölge olayları
  nlohmann::json log_events = nlohmann::json::array();  // bu karenin günlük kayıtları
  std::optional<nlohmann::json> focus_track;            // `focus` iz özeti (yoksa boş)
  std::vector<Alarm> alarms;                            // açık bölge alarmları
  nlohmann::json zone_counts = nlohmann::json::object();
  nlohmann::json health;
  // decode, health, pre, infer, post, track, ego, motion, zones, total
  std::map<std::string, double> timing_ms;
  bool session_new = false;
};

// `/v1/detect` gövdesi (Python `SessionRef.step` çıktısının ölçülen kısmı).
nlohmann::json response_json(const Response& r);

// Ego maskesine yalnız ham skoru >= min_score olan takip katmanı kutuları; düşük skorlu kutular
// çoğu zaman sabit doku (Görev 15C, Kabul 10 b teşhisi, val). Giriş sırası korunur.
std::vector<Box> ego_mask_boxes(const std::vector<Det>& layer, double min_score);

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
