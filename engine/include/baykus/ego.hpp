#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "baykus/geometry.hpp"
#include "baykus/motion_config.hpp"

namespace baykus {

// Python `baykus.motion.ego.Merge`: geçici nesil koordinatı → hedef nesil (2×3, tam çözünürlük).
struct Merge {
  int from_gen;
  int to_gen;
  cv::Matx23d transform;

  // `{"from", "to", "transform"}`; transform 6 sayı (satır sırası).
  nlohmann::json to_json() const;
};

// Python `baykus.motion.ego.EgoResult`; `frame_to_map` `[a, b, tx, c, d, ty]` ya da null.
struct EgoResult {
  bool ok;
  std::string source;
  int map_gen;
  bool provisional;
  std::optional<cv::Matx23d> frame_to_map;
  std::optional<int> inliers;  // kayıt yapılmadıysa (neslin ilk karesi, kayıp karesi) null
  std::optional<double> inlier_ratio;
  bool keyframe;
  bool relinked;
  bool relocalized;
  std::optional<Merge> merge;

  nlohmann::json to_json() const;
};

// Python `Ego.gens` değeri.
struct GenInfo {
  bool provisional;
  int opened_n;
  std::optional<int> merged_into;
};

// Yarı çözünürlük maske (255 = kullanılabilir): kutular `dilate_px` (tam çöz.) genişletilmiş,
// kenar bandı `border_px // 2`. Tam `x` ↔ yarı `(x − 0,5) / 2`; kutu dışa yuvarlanır.
cv::Mat ego_mask(cv::Size half_size, const std::vector<Box>& boxes, double dilate_px,
                 int border_px);

class Relocalizer;  // baykus/reloc.hpp

// Python `baykus.motion.ego.Ego`: anahtar kareli seyrek optik akış, nesil yönetimi.
class Ego {
 public:
  explicit Ego(const MotionCfg& cfg, Relocalizer* reloc = nullptr);

  // Kayıp kare (`gray_half` null ya da `source == "none"`) da buradan geçer.
  EgoResult update(const cv::Mat* gray_half, const std::string& source,
                   const std::vector<Box>& boxes, std::optional<int> frame);

  // Sonraki geçerli kare geçici nesil açar.
  void break_continuity();

  // Nesil 1'e döner, her şey silinir.
  void reset();

  const std::map<int, GenInfo>& generations() const;

 private:
  // `_register` dönüşü: (A_half | null, iç nokta, oran).
  struct RegisterResult {
    std::optional<cv::Matx23d> a_half;
    int inliers;
    double ratio;
  };

  EgoResult result_(bool ok, const std::string& source, const std::optional<cv::Matx23d>& m,
                    std::optional<int> inliers = std::nullopt,
                    std::optional<double> inlier_ratio = std::nullopt, bool keyframe = false,
                    bool relinked = false, bool relocalized = false,
                    std::optional<Merge> merge = std::nullopt) const;

  EgoResult lost_(const std::string& source);
  std::vector<cv::Point2f> corners_(const cv::Mat& gray, const std::vector<Box>& boxes) const;
  void set_key_(const cv::Mat& gray, const std::vector<cv::Point2f>& pts, const cv::Matx23d& m);
  std::optional<EgoResult> open_(const cv::Mat& gray, const std::string& source,
                                 const std::vector<Box>& boxes, bool provisional);
  EgoResult first_(const cv::Mat& gray, const std::string& source, const std::vector<Box>& boxes);
  EgoResult break_(const cv::Mat& gray, const std::string& source, const std::vector<Box>& boxes);
  RegisterResult register_(const cv::Mat& gray, const std::vector<Box>& boxes, int levels) const;
  EgoResult track_(const cv::Mat& gray, const std::string& source, const std::vector<Box>& boxes);
  EgoResult relocalize_(EgoResult res, const cv::Mat& gray, const std::vector<Box>& boxes,
                        std::optional<int> frame);
  EgoResult close_window_(EgoResult res);

  MotionCfg cfg_;
  Relocalizer* reloc_;

  int n_ = 0;
  int gen_ = 1;
  std::map<int, GenInfo> gens_;
  int last_gen_ = 1;  // nesil numaraları yeniden kullanılmaz (birleşen nesil dahil)
  int prov_ok_ = 0;   // geçici nesildeki ok kare sayısı (deneme sıklığı)
  cv::Mat key_;       // anahtar kare (yarı çöz. gri); boşsa anahtar yok
  std::vector<cv::Point2f> key_pts_;
  cv::Matx23d m_key_ = cv::Matx23d(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  int gap_ = 0;
  int age_ = 0;
  std::optional<std::string> source_;
  bool pending_break_ = false;
};

}  // namespace baykus
