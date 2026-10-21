#include "baykus/reloc.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>

#include "baykus/ego.hpp"
#include "baykus/geometry.hpp"

namespace baykus {

namespace {

cv::Matx23d mat_to_matx23d(const cv::Mat& m) {
  return cv::Matx23d(m.at<double>(0, 0), m.at<double>(0, 1), m.at<double>(0, 2), m.at<double>(1, 0),
                     m.at<double>(1, 1), m.at<double>(1, 2));
}

}  // namespace

Library::Library(const MotionCfg& cfg) : cfg_(cfg) {}

void Library::add(Record rec) {
  records_.push_back(std::move(rec));
  if (records_.size() > static_cast<std::size_t>(cfg_.reloc.library_max)) {
    records_.erase(records_.begin(), records_.begin() + (records_.size() - cfg_.reloc.library_max));
  }
}

std::vector<Record> Library::candidates(const std::string& source, std::optional<int> frame,
                                        int exclude_gen) const {
  const auto& rc = cfg_.reloc;
  std::vector<Record> pool;
  for (const auto& r : records_) {
    if (r.source == source && r.gen != exclude_gen) {
      pool.push_back(r);
    }
  }
  if (!frame.has_value()) {
    std::stable_sort(pool.begin(), pool.end(),
                     [](const Record& a, const Record& b) { return a.n > b.n; });
    if (pool.size() > static_cast<std::size_t>(rc.candidates)) {
      pool.resize(rc.candidates);
    }
    return pool;
  }
  std::vector<Record> near;
  for (const auto& r : pool) {
    if (r.frame.has_value() && std::abs(*r.frame - *frame) <= rc.max_dframe) {
      near.push_back(r);
    }
  }
  std::stable_sort(near.begin(), near.end(), [&](const Record& a, const Record& b) {
    const int da = std::abs(*a.frame - *frame);
    const int db = std::abs(*b.frame - *frame);
    if (da != db) return da < db;
    return a.n > b.n;
  });
  if (near.size() > static_cast<std::size_t>(rc.candidates)) {
    near.resize(rc.candidates);
  }
  return near;
}

void Library::remap_gen(int from_gen, int to_gen, const cv::Matx23d& t) {
  for (auto& r : records_) {
    if (r.gen == from_gen) {
      r.gen = to_gen;
      r.m_key = compose(t, r.m_key);
    }
  }
}

void Library::reset() { records_.clear(); }

std::size_t Library::size() const { return records_.size(); }

const std::vector<Record>& Library::records() const { return records_; }

std::pair<std::vector<cv::Point2f>, cv::Mat> orb_features(const cv::Mat& gray_half,
                                                          const std::vector<Box>& boxes,
                                                          const MotionCfg& cfg) {
  const auto& e = cfg.ego;
  cv::Mat mask =
      ego_mask(cv::Size(gray_half.cols, gray_half.rows), boxes, e.mask_dilate_px, e.border_px);
  cv::Ptr<cv::ORB> orb = cv::ORB::create(cfg.reloc.orb_features);
  std::vector<cv::KeyPoint> kps;
  cv::Mat desc;
  orb->detectAndCompute(gray_half, mask, kps, desc);
  if (desc.empty() || kps.empty()) {
    return {std::vector<cv::Point2f>{}, cv::Mat(0, 32, CV_8U)};
  }
  // ORB anahtar nokta sırası platforma bağlı; kanonik sıra için (y, x, octave, response, angle).
  std::vector<std::size_t> order(kps.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    const cv::KeyPoint& ka = kps[a];
    const cv::KeyPoint& kb = kps[b];
    return std::tie(ka.pt.y, ka.pt.x, ka.octave, ka.response, ka.angle) <
           std::tie(kb.pt.y, kb.pt.x, kb.octave, kb.response, kb.angle);
  });
  std::vector<cv::Point2f> pts;
  pts.reserve(kps.size());
  cv::Mat desc_sorted(desc.rows, desc.cols, desc.type());
  for (std::size_t i = 0; i < order.size(); ++i) {
    const std::size_t j = order[i];
    pts.push_back(kps[j].pt);
    desc.row(static_cast<int>(j)).copyTo(desc_sorted.row(static_cast<int>(i)));
  }
  return {std::move(pts), desc_sorted};
}

MatchResult match(const std::vector<cv::Point2f>& cur_kps, const cv::Mat& cur_desc,
                  const Record& rec, const MotionCfg& cfg) {
  const auto& rc = cfg.reloc;
  const auto& e = cfg.ego;
  if (cur_desc.empty() || rec.desc.empty()) {
    return {std::nullopt, 0, 0.0};
  }
  std::vector<cv::DMatch> pairs;
  cv::BFMatcher(cv::NORM_HAMMING, true).match(rec.desc, cur_desc, pairs);
  if (pairs.size() < 3) {
    return {std::nullopt, 0, 0.0};
  }
  std::vector<cv::Point2f> src;
  std::vector<cv::Point2f> dst;
  src.reserve(pairs.size());
  dst.reserve(pairs.size());
  for (const auto& p : pairs) {
    src.push_back(rec.kps[p.queryIdx]);
    dst.push_back(cur_kps[p.trainIdx]);
  }
  cv::Mat inl;
  cv::Mat a = cv::estimateAffinePartial2D(src, dst, inl, cv::RANSAC, rc.ransac_px,
                                          static_cast<std::size_t>(e.ransac_iters), e.confidence,
                                          static_cast<std::size_t>(e.refine_iters));
  if (a.empty() || inl.empty()) {
    return {std::nullopt, 0, 0.0};
  }
  const int k = cv::countNonZero(inl);
  const double ratio = static_cast<double>(k) / static_cast<double>(pairs.size());
  const bool ok = k >= rc.min_inliers && ratio >= rc.min_ratio;
  if (!ok) {
    return {std::nullopt, k, ratio};
  }
  return {mat_to_matx23d(a), k, ratio};
}

Relocalizer::Relocalizer(const MotionCfg& cfg) : cfg_(cfg), library_(cfg) {}

void Relocalizer::reset() { library_.reset(); }

void Relocalizer::add_keyframe(const cv::Mat& gray_half, const std::vector<Box>& boxes, int gen,
                               const cv::Matx23d& m_key, std::optional<int> frame, int n,
                               const std::string& source) {
  auto [kps, desc] = orb_features(gray_half, boxes, cfg_);
  Record rec;
  rec.gen = gen;
  rec.m_key = m_key;
  rec.frame = frame;
  rec.n = n;
  rec.source = source;
  rec.kps = std::move(kps);
  rec.desc = std::move(desc);
  library_.add(std::move(rec));
}

std::optional<Merge> Relocalizer::try_merge(const cv::Mat& gray_half, const std::vector<Box>& boxes,
                                            const std::string& source, std::optional<int> frame,
                                            const cv::Matx23d& m_prov, int prov_gen) {
  try {
    std::vector<Record> cands = library_.candidates(source, frame, prov_gen);
    if (cands.empty()) {
      return std::nullopt;
    }
    auto [kps, desc] = orb_features(gray_half, boxes, cfg_);
    bool have_best = false;
    std::tuple<int, int, int> best_key;
    Record best_rec;
    cv::Matx23d best_a;
    for (const auto& rec : cands) {
      MatchResult mr = match(kps, desc, rec, cfg_);
      if (!mr.a.has_value()) {
        continue;
      }
      int dframe = 0;
      if (rec.frame.has_value() && frame.has_value()) {
        dframe = std::abs(*rec.frame - *frame);
      }
      std::tuple<int, int, int> key(-mr.inliers, dframe, -rec.n);
      if (!have_best || key < best_key) {
        have_best = true;
        best_key = key;
        best_rec = rec;
        best_a = *mr.a;
      }
    }
    if (!have_best) {
      return std::nullopt;
    }
    const cv::Matx23d m_target = compose(best_rec.m_key, invert(to_full(best_a)));
    Merge merge;
    merge.from_gen = prov_gen;
    merge.to_gen = best_rec.gen;
    merge.transform = compose(m_target, invert(m_prov));
    return merge;
  } catch (const cv::Exception&) {
    return std::nullopt;
  }
}

Library& Relocalizer::library() { return library_; }

}  // namespace baykus
