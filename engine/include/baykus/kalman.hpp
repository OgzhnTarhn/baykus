#pragma once

// ByteTrack izinin Kalman süzgeci: trackers 2.6.1 `KalmanFilter` + `XYXYStateEstimator` +
// `ByteTrackTracklet._configure_noise`, sabit hızlı kip (frame_step = 1; Faz 3 tasarımı §4.3,
// "sabit sürümde doğrulama"). Durum (x1, y1, x2, y2, vx1, vy1, vx2, vy2), double; P₀ = I,
// Q = 0,01·I, R = 0,1·I, H = [I₄ 0], F sabit hız (dt = 1). Kazanç P Hᵀ S⁻¹ (LU ile ters),
// kovaryans güncellemesi Joseph biçimi (I − KH) P (I − KH)ᵀ + K R Kᵀ.
#include <Eigen/Core>
#include <array>

namespace baykus {

class KalmanBox {
 public:
  using State = Eigen::Matrix<double, 8, 1>;
  using Cov = Eigen::Matrix<double, 8, 8>;

  explicit KalmanBox(const std::array<double, 4>& bbox);

  void predict();
  void update(const std::array<double, 4>& bbox);

  std::array<double, 4> bbox() const;
  const State& state() const { return x_; }
  const Cov& covariance() const { return p_; }

 private:
  State x_;
  Cov p_;
};

}  // namespace baykus
