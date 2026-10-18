#include "baykus/image_io.hpp"

#include <cstdint>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <vector>

namespace baykus {

namespace {

int be16(const std::string& b, std::size_t i) {
  return (static_cast<unsigned char>(b[i]) << 8) | static_cast<unsigned char>(b[i + 1]);
}

std::uint32_t be32(const std::string& b, std::size_t i) {
  return (static_cast<std::uint32_t>(be16(b, i)) << 16) |
         static_cast<std::uint32_t>(be16(b, i + 2));
}

std::optional<cv::Size> png_size(const std::string& b) {
  static const std::string kSig = {char(0x89), 'P',        'N',        'G',
                                   char(0x0D), char(0x0A), char(0x1A), char(0x0A)};
  if (b.size() < 24 || b.compare(0, 8, kSig) != 0 || b.compare(12, 4, "IHDR") != 0) {
    return std::nullopt;
  }
  const std::uint32_t w = be32(b, 16), h = be32(b, 20);
  if (w == 0 || h == 0 || w > 0x7fffffffu || h > 0x7fffffffu) return std::nullopt;
  return cv::Size(static_cast<int>(w), static_cast<int>(h));
}

// JPEG: işaretçiler boyunca SOFn'e (C0-CF; C4 DHT, C8 JPG, CC DAC hariç) kadar ilerle.
std::optional<cv::Size> jpeg_size(const std::string& b) {
  if (b.size() < 4 || static_cast<unsigned char>(b[0]) != 0xFF ||
      static_cast<unsigned char>(b[1]) != 0xD8) {
    return std::nullopt;
  }
  std::size_t i = 2;
  while (i + 4 <= b.size()) {
    if (static_cast<unsigned char>(b[i]) != 0xFF) return std::nullopt;
    const int m = static_cast<unsigned char>(b[i + 1]);
    if (m == 0xFF) {  // dolgu baytı
      ++i;
      continue;
    }
    if (m == 0x01 || (m >= 0xD0 && m <= 0xD7)) {  // uzunluğu olmayan işaretçiler
      i += 2;
      continue;
    }
    const int len = be16(b, i + 2);
    if (len < 2) return std::nullopt;
    if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) {
      if (i + 9 > b.size()) return std::nullopt;
      const int h = be16(b, i + 5), w = be16(b, i + 7);
      if (w == 0 || h == 0) return std::nullopt;
      return cv::Size(w, h);
    }
    if (m == 0xD9 || m == 0xDA) return std::nullopt;  // görüntü sonu / tarama: SOF yoktu
    i += 2 + static_cast<std::size_t>(len);
  }
  return std::nullopt;
}

// Boş, başka biçim: 400; kenar sınırı aşılmış: 413 (çözmeden).
cv::Size checked_size(const std::string& bytes, const char* name) {
  if (bytes.empty()) {
    throw std::invalid_argument(std::string(name) + " görüntüsü boş");
  }
  const auto size = header_size(bytes);
  if (!size) {
    throw std::invalid_argument(std::string(name) + " görüntüsü PNG ya da JPEG değil");
  }
  if (size->width > kMaxImageSide || size->height > kMaxImageSide) {
    throw ImageTooLarge(std::string(name) + " görüntüsü " + std::to_string(size->width) + "×" +
                        std::to_string(size->height) + "; kenar sınırı " +
                        std::to_string(kMaxImageSide) + " px");
  }
  return *size;
}

cv::Mat decode(const std::string& bytes, int flag, const char* name) {
  if (bytes.empty()) {
    throw std::invalid_argument(std::string(name) + " görüntüsü boş");
  }
  cv::Mat img;
  try {
    const cv::Mat buf(1, static_cast<int>(bytes.size()), CV_8UC1, const_cast<char*>(bytes.data()));
    img = cv::imdecode(buf, flag);
  } catch (const cv::Exception&) {
    img.release();  // kütüphane istisnası sınırda yakalanır (tasarım §4.1)
  }
  if (img.empty()) {
    throw std::invalid_argument(std::string(name) + " görüntüsü çözülemedi");
  }
  return img;
}

}  // namespace

std::optional<cv::Size> header_size(const std::string& bytes) {
  if (auto s = png_size(bytes)) return s;
  return jpeg_size(bytes);
}

Pair decode_pair(const std::string& rgb_bytes, const std::string& ir_bytes) {
  // boyutlar çözmeden önce: sınır ve uyuşmazlık büyük görüntü belleğe açılmadan yakalanır
  if (checked_size(rgb_bytes, "RGB") != checked_size(ir_bytes, "IR")) {
    throw std::invalid_argument("RGB ve IR boyutu farklı");
  }
  Pair p{decode(rgb_bytes, cv::IMREAD_COLOR, "RGB"), decode(ir_bytes, cv::IMREAD_GRAYSCALE, "IR")};
  if (p.bgr.size() != p.ir.size()) {
    throw std::invalid_argument("RGB ve IR boyutu farklı");
  }
  return p;
}

cv::Mat stack_rgbt(const cv::Mat& bgr, const cv::Mat& ir) {
  if (bgr.size() != ir.size()) {
    throw std::invalid_argument("RGB ve IR boyutu farklı");
  }
  std::vector<cv::Mat> ch;
  cv::split(bgr, ch);
  cv::Mat out;
  cv::merge(std::vector<cv::Mat>{ch[2], ch[1], ch[0], ir}, out);
  return out;
}

}  // namespace baykus
