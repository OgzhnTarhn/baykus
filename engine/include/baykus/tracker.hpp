#pragma once

// ByteTrack: trackers 2.6.1 `ByteTrackTracker.update` akışının satır satır uyarlaması, zaman
// damgasız (sabit hızlı) kip (Faz 3 tasarımı §4.3, "sabit sürümde doğrulama"). Eşikler ham skora;
// skor eşikleri double, IoU eşiği float32'de (NumPy 2: float32 IoU ile Python float'ı float32'de
// karşılaştırılır). Kimlik sayacı nesne başına, 0'dan; `reset()` sıfırlar.
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "baykus/detections.hpp"
#include "baykus/kalman.hpp"

namespace baykus {

struct TrackerConfig {
  int lost_track_buffer = 30;
  double frame_rate = 30.0;
  double track_activation_threshold = 0.7;
  int minimum_consecutive_frames = 2;
  double minimum_iou_threshold = 0.1;
  double high_conf_det_threshold = 0.6;

  // `configs/engine/tracker.json`: altı anahtar zorunlu, fazlası hata (Python `load_cfg`).
  static TrackerConfig load(const std::string& path);
  nlohmann::json to_json() const;
  // 0 → 0; değilse max(1, ceil(frame_rate / 30 · lost_track_buffer)).
  int maximum_frames_without_update() const;
};

class ByteTrack {
 public:
  explicit ByteTrack(const TrackerConfig& cfg);

  // Bir kare: dönüş (Det::index, track_id) çiftleri referansın çıktı sırasıyla (1. aşama
  // eşleşmeleri, 2. aşama eşleşmeleri, eşleşmeyen düşükler, eşleşmeyen yüksekler); kimliği
  // verilmemiş iz ya da eşleşmeyen tespit -1. `dump` verilirse Python koşucusunun döküm biçimi
  // (tracks_before, tracks_after, dets, iou1, assign1, iou2, assign2).
  std::vector<std::pair<int, int>> update(const std::vector<Det>& dets,
                                          nlohmann::json* dump = nullptr);
  void reset();

 private:
  struct Track {
    KalmanBox kf;
    int tracker_id = -1;
    int time_since_update = 0;
    int consecutive = 1;  // ilk kutu ilk başarılı güncelleme sayılır
  };

  nlohmann::json track_rows() const;

  TrackerConfig cfg_;
  int max_frames_;
  float iou_threshold_;
  std::vector<Track> tracks_;
  int next_id_ = 0;
};

// Tespit dosyası (`baykus-engine detect` çıktısı ya da `sequences` fikstürü) → iz dosyası
// (Python `baykus-track run` biçimi: tracker, cfg, sequences[seq, split, frames, dumps]). Takip
// katmanı ham ≥ 0,1; her dizi yeni takipçi durumuyla. Test dizisi allow_test olmadan
// std::invalid_argument (yalnız son ölçüm).
nlohmann::json track_file(const nlohmann::json& dets, const TrackerConfig& cfg,
                          const std::set<int>& dump_frames, bool allow_test);

}  // namespace baykus
