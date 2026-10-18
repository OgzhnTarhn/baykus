#include "baykus/kalman.hpp"

#include <Eigen/LU>

namespace baykus {

namespace {

using H = Eigen::Matrix<double, 4, 8>;

KalmanBox::Cov transition() {  // sabit hız, dt = 1
  KalmanBox::Cov f = KalmanBox::Cov::Identity();
  for (int i = 0; i < 4; ++i) f(i, i + 4) = 1.0;
  return f;
}

H observation() {
  H h = H::Zero();
  h.leftCols<4>().setIdentity();
  return h;
}

const KalmanBox::Cov kF = transition();
const KalmanBox::Cov kQ = KalmanBox::Cov::Identity() * 0.01;
const Eigen::Matrix4d kR = Eigen::Matrix4d::Identity() * 0.1;
const H kH = observation();

}  // namespace

KalmanBox::KalmanBox(const std::array<double, 4>& bbox) : x_(State::Zero()), p_(Cov::Identity()) {
  for (int i = 0; i < 4; ++i) x_(i) = bbox[i];
}

void KalmanBox::predict() {
  x_ = kF * x_;
  p_ = kF * p_ * kF.transpose() + kQ;
}

void KalmanBox::update(const std::array<double, 4>& bbox) {
  const Eigen::Vector4d z(bbox[0], bbox[1], bbox[2], bbox[3]);
  const Eigen::Vector4d y = z - kH * x_;
  const Eigen::Matrix<double, 8, 4> pht = p_ * kH.transpose();
  const Eigen::Matrix4d s = kH * pht + kR;
  // np.linalg.inv: LAPACK LU (kısmi pivot); Eigen'in 4×4 kapalı formülü yerine aynı yol
  const Eigen::Matrix<double, 8, 4> k = pht * s.partialPivLu().inverse();
  x_ = x_ + k * y;
  const Cov ikh = Cov::Identity() - k * kH;
  p_ = ikh * p_ * ikh.transpose() + k * kR * k.transpose();
}

std::array<double, 4> KalmanBox::bbox() const { return {x_(0), x_(1), x_(2), x_(3)}; }

}  // namespace baykus
