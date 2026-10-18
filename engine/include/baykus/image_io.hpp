#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <stdexcept>
#include <string>

namespace baykus {

// RGB BGR uint8 (H, W, 3), IR gri uint8 (H, W); Python `baykus.engine_ref.compose.decode_pair`.
struct Pair {
  cv::Mat bgr;
  cv::Mat ir;
};

// Açılmış görüntü kenar sınırı (piksel). Boyut çözmeden önce başlıktan okunur: birkaç baytlık bir
// dosya GB'larca belleğe açılıp motoru düşürmesin (tasarım §4.1: tek hatalı istekle çökmez).
constexpr int kMaxImageSide = 4096;

// Sınırı aşan görüntü (sunucuda 413; parça boyutu sınırıyla aynı kod).
struct ImageTooLarge : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// PNG (IHDR) ya da JPEG (SOFn) başlığından genişlik × yükseklik; başka biçim ya da bozuk başlık:
// boş.
std::optional<cv::Size> header_size(const std::string& bytes);

// Kodlanmış baytlar → çift. Yalnız PNG / JPEG (tasarım §4.1). Boş, başka biçim, çözülemeyen görüntü
// ya da farklı boyut: std::invalid_argument; kenarı kMaxImageSide'ı aşan: ImageTooLarge (çözmeden).
Pair decode_pair(const std::string& rgb_bytes, const std::string& ir_bytes);

// R, G, B, T sırasıyla CV_8UC4 (Python `baykus.data.multichannel.stack_rgbt`).
cv::Mat stack_rgbt(const cv::Mat& bgr, const cv::Mat& ir);

}  // namespace baykus
