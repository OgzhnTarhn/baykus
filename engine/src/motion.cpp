#include "baykus/motion.hpp"

#include <algorithm>
#include <cmath>

namespace baykus {

MotionState::MotionState(const StateCfg& cfg) : cfg_(cfg) { reset(); }

void MotionState::reset() {
  n_ = 0;
  gen_.reset();
  ego_.clear();
  tracks_.clear();
}

void MotionState::relabel_(int from_gen, int to_gen, const cv::Matx23d& t) {
  const double sig = scale(t);
  for (auto& entry : tracks_) {
    for (auto& h : entry.second.hist) {
      if (h.second.gen == from_gen) {
        h.second.gen = to_gen;
        h.second.c = apply(t, h.second.c);
        h.second.s = sig * h.second.s;
      }
    }
  }
  for (auto& e : ego_) {
    if (e.gen == from_gen) e.gen = to_gen;
  }
  if (gen_ && *gen_ == from_gen) gen_ = to_gen;
}

bool MotionState::window_ok_(int gen) const {
  const int k = cfg_.k;
  if (static_cast<int>(ego_.size()) != k) return false;
  for (const auto& e : ego_) {
    if (!e.ok || e.gen != gen) return false;
  }
  return true;
}

std::map<int, TrackMotion> MotionState::update(const EgoResult& ego,
                                               const std::vector<TrackObs>& tracks) {
  ++n_;
  const int n = n_;
  if (ego.ok && ego.merge) {
    relabel_(ego.merge->from_gen, ego.merge->to_gen, ego.merge->transform);
  }
  if (ego.ok && gen_ && ego.map_gen != *gen_) {
    tracks_.clear();  // yeni nesil: geçmiş ve kararlar silinir
    ego_.clear();
  }
  if (ego.ok) gen_ = ego.map_gen;
  ego_.push_back(EgoEntry{n, ego.ok, ego.map_gen});
  while (static_cast<int>(ego_.size()) > cfg_.k) ego_.pop_front();
  const bool window_ok = ego.ok && window_ok_(ego.map_gen);

  std::map<int, TrackMotion> out;
  for (const auto& obs : tracks) {
    Track& tr = tracks_[obs.track_id];
    tr.last_seen = n;
    std::string prev = tr.decision;
    if (!tr.last_v_n || n - 1 - *tr.last_v_n > cfg_.hold) prev = "unknown";

    std::optional<double> v;
    std::optional<cv::Point2d> c;
    if (ego.ok && ego.frame_to_map) {
      const cv::Matx23d& m = *ego.frame_to_map;
      const double x1 = obs.box.x1;
      const double y1 = obs.box.y1;
      const double x2 = obs.box.x2;
      const double y2 = obs.box.y2;
      c = apply(m, cv::Point2d((x1 + x2) / 2.0, (y1 + y2) / 2.0));
      const double s = std::sqrt(std::max(x2 - x1, 0.0) * std::max(y2 - y1, 0.0)) * scale(m);
      tr.hist[n] = HistEntry{ego.map_gen, *c, s};
      const auto old = tr.hist.find(n - cfg_.k);
      if (window_ok && old != tr.hist.end() && old->second.gen == ego.map_gen && s > 0.0) {
        const double dx = c->x - old->second.c.x;
        const double dy = c->y - old->second.c.y;
        v = std::sqrt(dx * dx + dy * dy) / cfg_.k / s;
      }
    }

    if (v) {
      if (*v <= cfg_.stationary_max) {
        tr.decision = "stationary";
      } else if (*v >= cfg_.moving_min) {
        tr.decision = "moving";
      } else {
        tr.decision = prev;
      }
      tr.last_v_n = n;
    } else if (!tr.last_v_n || n - *tr.last_v_n > cfg_.hold) {
      tr.decision = "unknown";
    }

    for (auto it = tr.hist.begin(); it != tr.hist.end();) {
      if (it->first < n - cfg_.k) {
        it = tr.hist.erase(it);
      } else {
        ++it;
      }
    }

    TrackMotion tm;
    tm.motion = tr.decision;
    tm.speed_rel = v;
    if (c) tm.map_xy = *c;
    out[obs.track_id] = tm;
  }
  prune_(n);
  return out;
}

void MotionState::prune_(int n) {
  const int horizon = cfg_.k + cfg_.hold;
  for (auto it = tracks_.begin(); it != tracks_.end();) {
    if (n - it->second.last_seen > horizon) {
      it = tracks_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace baykus
