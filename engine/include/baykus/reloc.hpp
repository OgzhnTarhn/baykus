#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "baykus/ego.hpp"
#include "baykus/geometry.hpp"
#include "baykus/motion_config.hpp"

namespace baykus {

// Python `baykus.motion.reloc.Record`: anahtar kare kaydı (yarı çözünürlük ORB).
struct Record {
  int gen;
  cv::Matx23d m_key;
  std::optional<int> frame;
  int n;
  std::string source;
  std::vector<cv::Point2f> kps;  // N×2, yarı çözünürlük
  cv::Mat desc;                  // N×32 uint8
};

// Python `Library`: oturum başına harita belleği.
class Library {
 public:
  explicit Library(const MotionCfg& cfg);

  void add(Record rec);

  // `frame` varsa `|Δframe| ≤ max_dframe` olan en yakın kayıtlar (eşitlikte yeni `n`); yoksa en
  // yeni kayıtlar. Yakın kayıt yoksa aday yok.
  std::vector<Record> candidates(const std::string& source, std::optional<int> frame,
                                 int exclude_gen) const;

  void remap_gen(int from_gen, int to_gen, const cv::Matx23d& t);
  void reset();
  std::size_t size() const;
  const std::vector<Record>& records() const;

 private:
  MotionCfg cfg_;
  std::vector<Record> records_;
};

// ORB noktaları + tanımlayıcıları (ego maskesiyle).
std::pair<std::vector<cv::Point2f>, cv::Mat> orb_features(const cv::Mat& gray_half,
                                                          const std::vector<Box>& boxes,
                                                          const MotionCfg& cfg);

// Kayıt → geçerli kare `A_half`; `ok` değilse `a` boş.
struct MatchResult {
  std::optional<cv::Matx23d> a;
  int inliers;
  double ratio;
};

MatchResult match(const std::vector<cv::Point2f>& cur_kps, const cv::Mat& cur_desc,
                  const Record& rec, const MotionCfg& cfg);

// Python `Relocalizer`: kütüphane + birleşme denemesi.
class Relocalizer {
 public:
  explicit Relocalizer(const MotionCfg& cfg);

  void reset();

  void add_keyframe(const cv::Mat& gray_half, const std::vector<Box>& boxes, int gen,
                    const cv::Matx23d& m_key, std::optional<int> frame, int n,
                    const std::string& source);

  // `T = M' · M_prov⁻¹` (geçici nesil → hedef nesil); aday yoksa/başarısızsa null.
  std::optional<Merge> try_merge(const cv::Mat& gray_half, const std::vector<Box>& boxes,
                                 const std::string& source, std::optional<int> frame,
                                 const cv::Matx23d& m_prov, int prov_gen);

  Library& library();

 private:
  MotionCfg cfg_;
  Library library_;
};

}  // namespace baykus
