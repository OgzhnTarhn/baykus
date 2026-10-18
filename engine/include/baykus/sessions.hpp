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
#include <string>

#include "baykus/temporal.hpp"
#include "baykus/tracker.hpp"

namespace baykus {

struct Session {
  Session(const TrackerConfig& tracker_cfg, int window, int min_frames, double tol_px)
      : tracker(tracker_cfg), window(window, min_frames, tol_px) {}

  ByteTrack tracker;
  MisalignWindow window;
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

  // Oturum yoksa (hiç açılmadı, çıkarıldı ya da süresi doldu) yenisi açılır, *created = true.
  // shared_ptr: başka bir istek oturumu tablodan çıkarsa da kullanan istek güvende kalır.
  std::shared_ptr<Session> get(const std::string& id, bool* created);
  // Takipçi ve pencere sıfırlanır (kimlikler baştan); oturum yoksa yeni açılır.
  void reset(const std::string& id);
  std::size_t size();

 private:
  struct Entry {
    std::shared_ptr<Session> session;
    TimePoint last;
  };
  void expire(TimePoint now);

  TrackerConfig tracker_cfg_;
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
