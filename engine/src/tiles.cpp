#include "baykus/tiles.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace baykus {

namespace {

// İki kutunun IoU'su (double); kesişim yoksa ya da birleşim ≤ 0 ise 0.
double iou(const Det& a, const Det& b) {
  const double ix = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  const double iy = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (ix <= 0 || iy <= 0) {
    return 0.0;
  }
  const double inter = ix * iy;
  const double uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return uni <= 0 ? 0.0 : inter / uni;
}

}  // namespace

std::vector<cv::Rect> tile_windows(int w, int h) {
  const int pw = (w * 9 + 15) / 16;
  const int ph = (h * 9 + 15) / 16;
  return {cv::Rect(0, 0, pw, ph), cv::Rect(w - pw, 0, pw, ph), cv::Rect(0, h - ph, pw, ph),
          cv::Rect(w - pw, h - ph, pw, ph)};
}

std::vector<Det> merge_tiles(const std::vector<std::vector<Det>>& per_tile,
                             const std::vector<cv::Rect>& wins) {
  std::vector<std::pair<int, Det>> cands;
  for (std::size_t t = 0; t < per_tile.size(); ++t) {
    const int x0 = wins[t].x;
    const int y0 = wins[t].y;
    for (const Det& d : per_tile[t]) {
      Det s = d;
      s.x1 += x0;
      s.y1 += y0;
      s.x2 += x0;
      s.y2 += y0;
      cands.emplace_back(static_cast<int>(t), s);
    }
  }
  std::stable_sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) {
    if (a.second.score_raw != b.second.score_raw) {
      return a.second.score_raw > b.second.score_raw;
    }
    if (a.first != b.first) {
      return a.first < b.first;
    }
    return a.second.index < b.second.index;
  });
  std::vector<Det> kept;
  for (const auto& c : cands) {
    bool suppressed = false;
    for (const Det& k : kept) {
      if (iou(c.second, k) > kTileNmsIou) {
        suppressed = true;
        break;
      }
    }
    if (suppressed) {
      continue;
    }
    Det d = c.second;
    d.index = static_cast<int>(kept.size());
    kept.push_back(d);
  }
  order(kept);
  return kept;
}

}  // namespace baykus
