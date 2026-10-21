#include "baykus/ego.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "baykus/geometry.hpp"
#include "baykus/reloc.hpp"

namespace baykus {

namespace {

// Python `_in_mask`: floor(x + 0,5) pikseli; sınır dışı false.
bool in_mask(const cv::Mat& mask, const cv::Point2f& p) {
  const int h = mask.rows;
  const int w = mask.cols;
  const int xi = static_cast<int>(std::floor(p.x + 0.5f));
  const int yi = static_cast<int>(std::floor(p.y + 0.5f));
  if (xi < 0 || xi >= w || yi < 0 || yi >= h) return false;
  return mask.at<uchar>(yi, xi) > 0;
}

}  // namespace

nlohmann::json Merge::to_json() const {
  return nlohmann::json{
      {"from", from_gen},
      {"to", to_gen},
      {"transform", nlohmann::json::array({transform(0, 0), transform(0, 1), transform(0, 2),
                                           transform(1, 0), transform(1, 1), transform(1, 2)})}};
}

nlohmann::json EgoResult::to_json() const {
  nlohmann::json fm = nullptr;
  if (frame_to_map) {
    const cv::Matx23d& m = *frame_to_map;
    fm = nlohmann::json::array({m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2)});
  }
  nlohmann::json inl = nullptr;
  if (inliers) inl = *inliers;
  nlohmann::json ratio = nullptr;
  if (inlier_ratio) ratio = *inlier_ratio;
  nlohmann::json mg = nullptr;
  if (merge) mg = merge->to_json();
  return nlohmann::json{{"ok", ok},
                        {"source", source},
                        {"map_gen", map_gen},
                        {"provisional", provisional},
                        {"frame_to_map", fm},
                        {"inliers", inl},
                        {"inlier_ratio", ratio},
                        {"keyframe", keyframe},
                        {"relinked", relinked},
                        {"relocalized", relocalized},
                        {"merge", mg}};
}

cv::Mat ego_mask(cv::Size half_size, const std::vector<Box>& boxes, double dilate_px,
                 int border_px) {
  const int h = half_size.height;
  const int w = half_size.width;
  cv::Mat mask(h, w, CV_8U, cv::Scalar(255));
  const int b = border_px / 2;  // Python `border_px // 2`
  if (b > 0) {
    const int tb = std::min(b, h);
    const int bb = std::min(b, h);
    const int lb = std::min(b, w);
    const int rb = std::min(b, w);
    mask.rowRange(0, tb).setTo(0);
    mask.rowRange(h - bb, h).setTo(0);
    mask.colRange(0, lb).setTo(0);
    mask.colRange(w - rb, w).setTo(0);
  }
  for (const auto& box : boxes) {
    const int hx1 = std::max(0, static_cast<int>(std::floor((box.x1 - dilate_px - 0.5) / 2.0)));
    const int hy1 = std::max(0, static_cast<int>(std::floor((box.y1 - dilate_px - 0.5) / 2.0)));
    const int hx2 = std::min(w - 1, static_cast<int>(std::ceil((box.x2 + dilate_px - 0.5) / 2.0)));
    const int hy2 = std::min(h - 1, static_cast<int>(std::ceil((box.y2 + dilate_px - 0.5) / 2.0)));
    if (hx1 <= hx2 && hy1 <= hy2) {
      mask(cv::Rect(hx1, hy1, hx2 - hx1 + 1, hy2 - hy1 + 1)).setTo(0);
    }
  }
  return mask;
}

Ego::Ego(const MotionCfg& cfg, Relocalizer* reloc) : cfg_(cfg), reloc_(reloc) { reset(); }

void Ego::reset() {
  n_ = 0;
  gen_ = 1;
  gens_.clear();
  last_gen_ = 1;
  prov_ok_ = 0;
  key_ = cv::Mat();
  key_pts_.clear();
  m_key_ = identity();
  gap_ = 0;
  age_ = 0;
  source_.reset();
  pending_break_ = false;
  if (reloc_ != nullptr) reloc_->reset();
}

void Ego::break_continuity() {
  if (!key_.empty()) pending_break_ = true;
}

const std::map<int, GenInfo>& Ego::generations() const { return gens_; }

EgoResult Ego::update(const cv::Mat* gray_half, const std::string& source,
                      const std::vector<Box>& boxes, std::optional<int> frame) {
  ++n_;
  EgoResult res;
  if (gray_half == nullptr || source == "none") {
    res = lost_("none");
  } else if (key_.empty()) {
    res = first_(*gray_half, source, boxes);
  } else if (pending_break_ || !source_ || *source_ != source || gap_ > cfg_.ego.relink_max_gap) {
    res = break_(*gray_half, source, boxes);
  } else {
    res = track_(*gray_half, source, boxes);
  }
  if (res.ok && reloc_ != nullptr) {
    res = relocalize_(res, *gray_half, boxes, frame);
  }
  return close_window_(res);
}

EgoResult Ego::result_(bool ok, const std::string& source, const std::optional<cv::Matx23d>& m,
                       std::optional<int> inliers, std::optional<double> inlier_ratio,
                       bool keyframe, bool relinked, bool relocalized,
                       std::optional<Merge> merge) const {
  EgoResult r;
  r.ok = ok;
  r.source = source;
  r.map_gen = gen_;
  const auto it = gens_.find(gen_);
  r.provisional = it != gens_.end() ? it->second.provisional : false;
  r.frame_to_map = m;
  r.inliers = inliers;
  r.inlier_ratio = inlier_ratio;
  r.keyframe = keyframe;
  r.relinked = relinked;
  r.relocalized = relocalized;
  r.merge = merge;
  return r;
}

EgoResult Ego::lost_(const std::string& source) {
  if (!key_.empty()) {
    ++gap_;
    ++age_;
  }
  return result_(false, source, std::nullopt);
}

std::vector<cv::Point2f> Ego::corners_(const cv::Mat& gray, const std::vector<Box>& boxes) const {
  const EgoCfg& e = cfg_.ego;
  const cv::Mat mask = ego_mask(gray.size(), boxes, e.mask_dilate_px, e.border_px);
  std::vector<cv::Point2f> pts;
  cv::goodFeaturesToTrack(gray, pts, e.max_corners, e.quality, e.min_distance, mask, e.block_size,
                          false);
  return pts;
}

void Ego::set_key_(const cv::Mat& gray, const std::vector<cv::Point2f>& pts, const cv::Matx23d& m) {
  key_ = gray.clone();
  key_pts_ = pts;
  m_key_ = m;
  age_ = 0;
}

std::optional<EgoResult> Ego::open_(const cv::Mat& gray, const std::string& source,
                                    const std::vector<Box>& boxes, bool provisional) {
  const std::vector<cv::Point2f> pts = corners_(gray, boxes);
  if (static_cast<int>(pts.size()) < cfg_.ego.ok_min_inliers) return std::nullopt;
  if (provisional) {
    for (auto& kv : gens_) kv.second.provisional = false;  // açık geçici nesil kapanır (§5.11)
    ++last_gen_;
    gen_ = last_gen_;
  }
  gens_[gen_] = GenInfo{provisional, n_, std::nullopt};
  set_key_(gray, pts, identity());
  source_ = source;
  gap_ = 0;
  pending_break_ = false;
  prov_ok_ = 0;
  return result_(true, source, identity(), std::nullopt, std::nullopt, true);
}

EgoResult Ego::first_(const cv::Mat& gray, const std::string& source,
                      const std::vector<Box>& boxes) {
  const auto res = open_(gray, source, boxes, false);
  if (res) return *res;
  return result_(false, source, std::nullopt);
}

EgoResult Ego::break_(const cv::Mat& gray, const std::string& source,
                      const std::vector<Box>& boxes) {
  const auto res = open_(gray, source, boxes, true);
  if (!res) {  // köşe az: kopma bekler
    pending_break_ = true;
    return lost_(source);
  }
  return *res;
}

Ego::RegisterResult Ego::register_(const cv::Mat& gray, const std::vector<Box>& boxes,
                                   int levels) const {
  const EgoCfg& e = cfg_.ego;
  RegisterResult out{std::nullopt, 0, 0.0};
  if (key_pts_.empty()) return out;
  const cv::Size win(e.lk_win, e.lk_win);
  std::vector<cv::Point2f> p1;
  std::vector<uchar> st1;
  std::vector<float> err1;
  cv::calcOpticalFlowPyrLK(key_, gray, key_pts_, p1, st1, err1, win, levels);
  if (p1.empty()) return out;
  std::vector<cv::Point2f> pb;
  std::vector<uchar> st2;
  std::vector<float> err2;
  cv::calcOpticalFlowPyrLK(gray, key_, p1, pb, st2, err2, win, levels);
  if (pb.empty()) return out;

  const cv::Mat mask = ego_mask(gray.size(), boxes, e.mask_dilate_px, e.border_px);
  std::vector<cv::Point2f> from;
  std::vector<cv::Point2f> to;
  from.reserve(key_pts_.size());
  to.reserve(key_pts_.size());
  for (std::size_t i = 0; i < key_pts_.size(); ++i) {
    if (st1[i] != 1 || st2[i] != 1) continue;
    const float dx = pb[i].x - key_pts_[i].x;
    const float dy = pb[i].y - key_pts_[i].y;
    const float fb = std::sqrt(dx * dx + dy * dy);
    if (!std::isfinite(fb)) continue;
    if (fb > e.fb_max_px) continue;
    if (!in_mask(mask, p1[i])) continue;
    from.push_back(key_pts_[i]);
    to.push_back(p1[i]);
  }
  const int n = static_cast<int>(from.size());
  if (n < 3) return out;  // estimateAffinePartial2D en az 2 nokta ister; 3'ün altı anlamsız

  cv::Mat inl;
  const cv::Mat a_half = cv::estimateAffinePartial2D(from, to, inl, cv::RANSAC, e.ransac_px,
                                                     e.ransac_iters, e.confidence, e.refine_iters);
  if (a_half.empty() || inl.empty()) return out;
  const int k = cv::countNonZero(inl);
  out.a_half =
      cv::Matx23d(a_half.at<double>(0, 0), a_half.at<double>(0, 1), a_half.at<double>(0, 2),
                  a_half.at<double>(1, 0), a_half.at<double>(1, 1), a_half.at<double>(1, 2));
  out.inliers = k;
  out.ratio = static_cast<double>(k) / n;
  return out;
}

EgoResult Ego::track_(const cv::Mat& gray, const std::string& source,
                      const std::vector<Box>& boxes) {
  const EgoCfg& e = cfg_.ego;
  const bool relink = gap_ > 0;
  const RegisterResult r = register_(gray, boxes, relink ? e.relink_levels : e.lk_levels);
  const bool ok =
      r.a_half.has_value() && r.inliers >= e.ok_min_inliers && r.ratio >= e.ok_min_ratio;
  if (!ok) {
    if (relink) pending_break_ = true;
    lost_(source);
    return result_(false, source, std::nullopt, r.inliers, r.ratio);
  }
  gap_ = 0;
  ++age_;
  const cv::Matx23d m = compose(m_key_, invert(to_full(*r.a_half)));
  const double shift = std::hypot((*r.a_half)(0, 2), (*r.a_half)(1, 2));
  const bool key = r.inliers < e.key_min_inliers || r.ratio < e.key_min_ratio ||
                   shift > e.key_max_shift_px || (!relink && age_ >= e.key_max_age);
  if (key) set_key_(gray, corners_(gray, boxes), m);
  return result_(true, source, m, r.inliers, r.ratio, key, relink);
}

EgoResult Ego::relocalize_(EgoResult res, const cv::Mat& gray, const std::vector<Box>& boxes,
                           std::optional<int> frame) {
  const RelocCfg& rc = cfg_.reloc;
  if (res.keyframe) {
    reloc_->add_keyframe(gray, boxes, gen_, *res.frame_to_map, frame, n_, res.source);
  }
  GenInfo& g = gens_[gen_];
  if (!g.provisional || n_ - g.opened_n >= rc.window_frames) return res;
  const bool due = prov_ok_ % rc.every == 0;
  ++prov_ok_;
  if (!due) return res;
  const std::optional<Merge> mg =
      reloc_->try_merge(gray, boxes, res.source, frame, *res.frame_to_map, gen_);
  if (!mg) return res;
  const cv::Matx23d t = mg->transform;
  reloc_->library().remap_gen(mg->from_gen, mg->to_gen, t);
  m_key_ = compose(t, m_key_);
  g.provisional = false;
  g.merged_into = mg->to_gen;
  gen_ = mg->to_gen;
  const auto it = gens_.find(mg->to_gen);
  res.map_gen = mg->to_gen;
  res.provisional = it != gens_.end() ? it->second.provisional : false;
  res.frame_to_map = compose(t, *res.frame_to_map);
  res.relocalized = true;
  res.merge = mg;
  return res;
}

EgoResult Ego::close_window_(EgoResult res) {
  const auto it = gens_.find(gen_);
  if (it == gens_.end() || !it->second.provisional) return res;
  if (n_ - it->second.opened_n < cfg_.reloc.window_frames) return res;
  it->second.provisional = false;
  res.provisional = false;
  return res;
}

}  // namespace baykus
