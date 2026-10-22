#include "baykus/sessions.hpp"

#include <algorithm>
#include <utility>

namespace baykus {

Sessions::Sessions(TrackerConfig tracker_cfg, int window, int min_frames, double tol_px,
                   std::size_t max, std::chrono::seconds idle, Clock clock)
    : Sessions(std::move(tracker_cfg), std::nullopt, window, min_frames, tol_px, max, idle,
               std::move(clock)) {}

Sessions::Sessions(TrackerConfig tracker_cfg, MotionCfg motion_cfg, int window, int min_frames,
                   double tol_px, std::size_t max, std::chrono::seconds idle, Clock clock)
    : Sessions(std::move(tracker_cfg), std::optional<MotionCfg>(std::move(motion_cfg)), window,
               min_frames, tol_px, max, idle, std::move(clock)) {}

Sessions::Sessions(TrackerConfig tracker_cfg, std::optional<MotionCfg> motion_cfg, int window,
                   int min_frames, double tol_px, std::size_t max, std::chrono::seconds idle,
                   Clock clock)
    : tracker_cfg_(std::move(tracker_cfg)),
      motion_cfg_(std::move(motion_cfg)),
      window_(window),
      min_frames_(min_frames),
      tol_px_(tol_px),
      max_(std::max<std::size_t>(max, 1)),
      idle_(idle),
      clock_(std::move(clock)) {}

void Sessions::expire(TimePoint now) {
  for (auto it = entries_.begin(); it != entries_.end();) {
    it = now - it->second.last > idle_ ? entries_.erase(it) : std::next(it);
  }
}

std::shared_ptr<Session> Sessions::get(const std::string& id, bool* created) {
  std::lock_guard<std::mutex> lock(m_);
  const TimePoint now = clock_();
  expire(now);
  const auto it = entries_.find(id);
  if (it != entries_.end()) {
    it->second.last = now;
    if (created != nullptr) *created = false;
    return it->second.session;
  }
  if (entries_.size() >= max_) {
    const auto oldest = std::min_element(
        entries_.begin(), entries_.end(),
        [](const auto& a, const auto& b) { return a.second.last < b.second.last; });
    entries_.erase(oldest);
  }
  auto s = std::make_shared<Session>(tracker_cfg_, window_, min_frames_, tol_px_,
                                     motion_cfg_ ? &*motion_cfg_ : nullptr);
  entries_[id] = Entry{s, now};
  if (created != nullptr) *created = true;
  return s;
}

std::shared_ptr<Session> Sessions::find(const std::string& id) {
  std::lock_guard<std::mutex> lock(m_);
  const TimePoint now = clock_();
  expire(now);
  const auto it = entries_.find(id);
  if (it == entries_.end()) return nullptr;
  it->second.last = now;
  return it->second.session;
}

void Sessions::reset(const std::string& id, Scope scope) {
  bool created = false;
  auto s = get(id, &created);
  if (created) return;  // yeni oturum zaten boş
  std::lock_guard<std::mutex> lock(s->m);
  s->tracker.reset();
  s->window.reset();
  if (s->motion) {
    s->motion->motion.reset();
    s->motion->frozen.reset();
    if (scope == Scope::Tracks) {
      s->id_base = s->max_api_id + 1;
      s->motion->ego.break_continuity();  // kütüphane korunur
      s->motion->zones.reset_tracks();
      ZoneEvent e;
      e.n = s->n;
      e.frame = s->last_frame;
      e.kind = "session_reset_tracks";
      e.gen = s->has_ego ? s->last_gen : 1;
      s->motion->log.append(e, "", "");
    } else {
      s->motion->ego.reset();
      s->motion->zones.reset();
      s->motion->log.reset();
      s->motion->summary.reset();
      s->has_ego = false;
      s->last_gen = 1;
      s->last_provisional = false;
      s->last_frame = 0;
      s->id_base = 0;
      s->max_api_id = -1;
      s->n = 0;
    }
  }
}

std::size_t Sessions::size() {
  std::lock_guard<std::mutex> lock(m_);
  expire(clock_());
  return entries_.size();
}

}  // namespace baykus
