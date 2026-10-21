#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace baykus {

struct Box {
  double x1, y1, x2, y2;
};

// Python baykus.motion.images.half_gray: 3 kanallıysa BGR2GRAY, sonra INTER_AREA ile yarı boyut.
inline cv::Mat half_gray(const cv::Mat& img) {
  cv::Mat g;
  if (img.channels() == 3) {
    cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
  } else {
    g = img;
  }
  cv::Mat out;
  cv::resize(g, out, cv::Size(g.cols / 2, g.rows / 2), 0, 0, cv::INTER_AREA);
  return out;
}

namespace detail {

inline cv::Matx33d to_h(const cv::Matx23d& a) {
  return cv::Matx33d(a(0, 0), a(0, 1), a(0, 2), a(1, 0), a(1, 1), a(1, 2), 0.0, 0.0, 1.0);
}

inline cv::Matx23d from_h(const cv::Matx33d& h) {
  return cv::Matx23d(h(0, 0), h(0, 1), h(0, 2), h(1, 0), h(1, 1), h(1, 2));
}

}  // namespace detail

// Yarı çözünürlük → tam çözünürlük: S·A·S⁻¹, S = [[2,0,0.5],[0,2,0.5],[0,0,1]].
inline cv::Matx23d to_full(const cv::Matx23d& a_half) {
  static const cv::Matx33d S(2.0, 0.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0);
  static const cv::Matx33d S_inv = S.inv();
  return detail::from_h(S * detail::to_h(a_half) * S_inv);
}

inline cv::Matx23d invert(const cv::Matx23d& a) { return detail::from_h(detail::to_h(a).inv()); }

// Önce b sonra a (a∘b).
inline cv::Matx23d compose(const cv::Matx23d& a, const cv::Matx23d& b) {
  return detail::from_h(detail::to_h(a) * detail::to_h(b));
}

inline cv::Point2d apply(const cv::Matx23d& a, const cv::Point2d& p) {
  return cv::Point2d(a(0, 0) * p.x + a(0, 1) * p.y + a(0, 2),
                     a(1, 0) * p.x + a(1, 1) * p.y + a(1, 2));
}

inline double scale(const cv::Matx23d& a) {
  return std::sqrt(std::abs(a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)));
}

inline double corner_error(const cv::Matx23d& a, const cv::Matx23d& b, double w, double h) {
  const cv::Point2d corners[4] = {{0.0, 0.0}, {w, 0.0}, {w, h}, {0.0, h}};
  double m = 0.0;
  for (const auto& c : corners) {
    const cv::Point2d d = apply(a, c) - apply(b, c);
    m = std::max(m, std::hypot(d.x, d.y));
  }
  return m;
}

// `(cx, cy)` etrafında `deg` derece dönme ve `s` ölçek, sonra (tx, ty) öteleme.
inline cv::Matx23d similarity(double tx, double ty, double deg, double s, double cx = 0.0,
                              double cy = 0.0) {
  const double r = deg * CV_PI / 180.0;
  const double c = s * std::cos(r);
  const double k = s * std::sin(r);
  return cv::Matx23d(c, -k, cx - c * cx + k * cy + tx, k, c, cy - k * cx - c * cy + ty);
}

inline cv::Matx23d identity() { return cv::Matx23d(1.0, 0.0, 0.0, 0.0, 1.0, 0.0); }

}  // namespace baykus
