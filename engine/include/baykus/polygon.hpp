#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace baykus {

// (b−a)×(c−a) işaretli alan; kesin double, epsilon yok.
inline double cross(const cv::Point2d& a, const cv::Point2d& b, const cv::Point2d& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// p, [a, b] doğru parçası üstünde mi? Uçlar dahil, doğrusal ve kutu içinde.
inline bool on_segment(const cv::Point2d& p, const cv::Point2d& a, const cv::Point2d& b) {
  if (cross(a, b, p) != 0.0) return false;
  return p.x >= std::min(a.x, b.x) && p.x <= std::max(a.x, b.x) && p.y >= std::min(a.y, b.y) &&
         p.y <= std::max(a.y, b.y);
}

// [a, b] ile [c, d] kesişiyor mu? Uca dokunma ve doğrusal çakışma dahil.
inline bool segments_intersect(const cv::Point2d& a, const cv::Point2d& b, const cv::Point2d& c,
                               const cv::Point2d& d) {
  const double d1 = cross(c, d, a);
  const double d2 = cross(c, d, b);
  const double d3 = cross(a, b, c);
  const double d4 = cross(a, b, d);
  if (((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) &&
      ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))) {
    return true;
  }
  if (d1 == 0.0 && on_segment(a, c, d)) return true;
  if (d2 == 0.0 && on_segment(b, c, d)) return true;
  if (d3 == 0.0 && on_segment(c, a, b)) return true;
  if (d4 == 0.0 && on_segment(d, a, b)) return true;
  return false;
}

// Kenar üstü true; değilse ışın atma ile iç/dış.
inline bool point_in_polygon(const cv::Point2d& p, const std::vector<cv::Point2d>& poly) {
  const std::size_t n = poly.size();
  if (n < 3) return false;
  for (std::size_t i = 0; i < n; ++i) {
    if (on_segment(p, poly[i], poly[(i + 1) % n])) return true;
  }
  bool inside = false;
  for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
    const cv::Point2d& a = poly[i];
    const cv::Point2d& b = poly[j];
    if (((a.y > p.y) != (b.y > p.y)) && (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)) {
      inside = !inside;
    }
  }
  return inside;
}

// Shoelace, mutlak değer.
inline double polygon_area(const std::vector<cv::Point2d>& poly) {
  const std::size_t n = poly.size();
  if (n < 3) return 0.0;
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const cv::Point2d& a = poly[i];
    const cv::Point2d& b = poly[(i + 1) % n];
    s += a.x * b.y - b.x * a.y;
  }
  return std::abs(s) * 0.5;
}

// Basit (kendini kesmeyen) çokgen mi?
inline bool is_simple(const std::vector<cv::Point2d>& poly) {
  const std::size_t n = poly.size();
  if (n < 3) return false;
  // Sıfır uzunluklu kenar (son→ilk dahil) basitliği bozar.
  for (std::size_t i = 0; i < n; ++i) {
    const cv::Point2d& a = poly[i];
    const cv::Point2d& b = poly[(i + 1) % n];
    if (a.x == b.x && a.y == b.y) return false;
  }
  // Komşu olmayan kenar çiftleri kesişemez.
  for (std::size_t i = 0; i < n; ++i) {
    const cv::Point2d& a = poly[i];
    const cv::Point2d& b = poly[(i + 1) % n];
    for (std::size_t j = i + 1; j < n; ++j) {
      if (j == i + 1 || (i == 0 && j == n - 1)) continue;  // komşu kenarlar
      const cv::Point2d& c = poly[j];
      const cv::Point2d& d = poly[(j + 1) % n];
      if (segments_intersect(a, b, c, d)) return false;
    }
  }
  // Komşu kenarlar doğrusal ve geri katlanıyorsa basit değil; düz geçen serbest.
  for (std::size_t i = 0; i < n; ++i) {
    const cv::Point2d& prev = poly[(i + n - 1) % n];
    const cv::Point2d& cur = poly[i];
    const cv::Point2d& next = poly[(i + 1) % n];
    if (cross(prev, cur, next) == 0.0) {
      const double dot = (prev.x - cur.x) * (next.x - cur.x) + (prev.y - cur.y) * (next.y - cur.y);
      if (dot > 0.0) return false;
    }
  }
  return true;
}

}  // namespace baykus
