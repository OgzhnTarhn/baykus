#pragma once

// Oturumlar (Faz 3 tasarımı §4.1): takipçi ve zaman kuralı penceresi oturum başına; en çok `max`
// oturum (dolunca en uzun süredir kullanılmayan çıkar), `idle`'dan uzun boşta kalan silinir.
// Oturum başına kilit; oturum tablosu kendi kilidiyle korunur.
#include <chrono>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "baykus/ego.hpp"
#include "baykus/event_log.hpp"
#include "baykus/frozen.hpp"
#include "baykus/motion.hpp"
#include "baykus/motion_config.hpp"
#include "baykus/reloc.hpp"
#include "baykus/temporal.hpp"
#include "baykus/track_summary.hpp"
#include "baykus/tracker.hpp"
#include "baykus/zones.hpp"

namespace baykus {

enum class Scope { All, Tracks };

// Hareket alt sistemi (ego + yeniden konumlama + hareket durumu + donmuş dedektörü). Kopyalanamaz,
// taşınamaz: `ego` içindeki `Relocalizer*` bu nesnedeki `reloc`'a işaret eder.
struct MotionParts {
  explicit MotionParts(const MotionCfg& cfg)
      : reloc(cfg),
        ego(cfg, &reloc),
        motion(cfg.motion),
        frozen(cfg.frozen),
        zones(cfg.zones),
        log(cfg.log),
        summary(cfg.summary) {}

  MotionParts(const MotionParts&) = delete;
  MotionParts& operator=(const MotionParts&) = delete;
  MotionParts(MotionParts&&) = delete;
  MotionParts& operator=(MotionParts&&) = delete;

  Relocalizer reloc;
  Ego ego;
  MotionState motion;
  FrozenDetector frozen;
  Zones zones;
  EventLog log;
  TrackSummaries summary;
};

struct Session {
  Session(const TrackerConfig& tracker_cfg, int window, int min_frames, double tol_px,
          const MotionCfg* motion_cfg = nullptr)
      : tracker(tracker_cfg), window(window, min_frames, tol_px) {
    if (motion_cfg != nullptr && motion_cfg->ego.enabled) {
      motion = std::make_unique<MotionParts>(*motion_cfg);
    }
  }

  ByteTrack tracker;
  MisalignWindow window;
  std::unique_ptr<MotionParts> motion;  // null = hareket kapalı
  int id_base = 0;
  int max_api_id = -1;
  int n = 0;
  int last_frame = 0;
  bool has_ego = false;
  int last_gen = 1;
  bool last_provisional = false;
  std::mutex m;  // aynı oturuma eşzamanlı iki istek sırayla işlenir
};

class Sessions {
 public:
  using TimePoint = std::chrono::steady_clock::time_point;
  using Clock = std::function<TimePoint()>;

  Sessions(
      TrackerConfig tracker_cfg, int window, int min_frames, double tol_px, std::size_t max = 8,
      std::chrono::seconds idle = std::chrono::seconds(600),
      Clock clock = [] { return std::chrono::steady_clock::now(); });

  // Hareketli kurucu: `motion_cfg.ego.enabled` ise yeni açılan her oturuma `MotionParts` kurulur.
  Sessions(
      TrackerConfig tracker_cfg, MotionCfg motion_cfg, int window, int min_frames, double tol_px,
      std::size_t max = 8, std::chrono::seconds idle = std::chrono::seconds(600),
      Clock clock = [] { return std::chrono::steady_clock::now(); });

  // Oturum yoksa (hiç açılmadı, çıkarıldı ya da süresi doldu) yenisi açılır, *created = true.
  // shared_ptr: başka bir istek oturumu tablodan çıkarsa da kullanan istek güvende kalır.
  std::shared_ptr<Session> get(const std::string& id, bool* created);
  // Oturum varsa döner (son kullanım güncellenir, süresi dolanlar önce silinir), yoksa nullptr;
  // oturum açmaz.
  std::shared_ptr<Session> find(const std::string& id);
  // Takipçi ve pencere sıfırlanır (kimlikler baştan); oturum yoksa yeni açılır.
  void reset(const std::string& id, Scope scope = Scope::All);
  std::size_t size();

  const std::optional<MotionCfg>& motion_cfg() const { return motion_cfg_; }

 private:
  struct Entry {
    std::shared_ptr<Session> session;
    TimePoint last;
  };
  void expire(TimePoint now);

  Sessions(TrackerConfig tracker_cfg, std::optional<MotionCfg> motion_cfg, int window,
           int min_frames, double tol_px, std::size_t max, std::chrono::seconds idle, Clock clock);

  TrackerConfig tracker_cfg_;
  std::optional<MotionCfg> motion_cfg_;
  int window_;
  int min_frames_;
  double tol_px_;
  std::size_t max_;
  std::chrono::seconds idle_;
  Clock clock_;
  std::mutex m_;
  std::map<std::string, Entry> entries_;
};

}  // namespace baykus
