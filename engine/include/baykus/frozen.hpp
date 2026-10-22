#pragma once

#include <deque>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "baykus/motion_config.hpp"

namespace baykus {

// `python/src/baykus/health/frozen.py` portu (Faz 4 tasarımı §5.6; motor `health`).
// Kanal başına, oturumlu istekte: yarı çözünürlük gri görüntünün önceki kareyle ortalama mutlak
// farkı `diff_mad` (0-255); `frozen` = `diff_mad ≤ mad_max` (sınır dahil). Kayıp kanal (`lost`)
// donmuş sayılmaz. İlk kare, görüntüsü olmayan kare ya da boyutu değişen kare: `diff_mad = null`.
// `alarm` = son `window` karenin (geçerli dahil) en az `min_frames`'i donmuş.

class FrozenDetector {
 public:
  explicit FrozenDetector(const FrozenCfg& cfg);

  // Boş cv::Mat = görüntü yok (Python None).
  nlohmann::json update(const cv::Mat& rgb_half, const cv::Mat& ir_half, bool rgb_lost,
                        bool ir_lost);

  void reset();

 private:
  struct Channel {
    cv::Mat prev;
    std::deque<bool> buf;
  };

  nlohmann::json channel(Channel& ch, const cv::Mat& img, bool lost);

  FrozenCfg cfg_;
  Channel rgb_;
  Channel ir_;
};

}  // namespace baykus
