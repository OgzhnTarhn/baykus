#include "baykus/sessions.hpp"

#include <algorithm>
#include <utility>

namespace baykus {

Sessions::Sessions(TrackerConfig tracker_cfg, int window, int min_frames, double tol_px,
                   std::size_t max, std::chrono::seconds idle, Clock clock)
    : tracker_cfg_(std::move(tracker_cfg)),
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
  auto s = std::make_shared<Session>(tracker_cfg_, window_, min_frames_, tol_px_);
  entries_[id] = Entry{s, now};
  if (created != nullptr) *created = true;
  return s;
}

void Sessions::reset(const std::string& id) {
  bool created = false;
  auto s = get(id, &created);
  if (created) return;  // yeni oturum zaten boş
  std::lock_guard<std::mutex> lock(s->m);
  s->tracker.reset();
  s->window.reset();
}

std::size_t Sessions::size() {
  std::lock_guard<std::mutex> lock(m_);
  expire(clock_());
  return entries_.size();
}

}  // namespace baykus
