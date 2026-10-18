#include "baykus/image_io.hpp"

#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <vector>

namespace baykus {

namespace {

cv::Mat decode(const std::string& bytes, int flag, const char* name) {
  if (bytes.empty()) {
    throw std::invalid_argument(std::string(name) + " görüntüsü boş");
  }
  cv::Mat img;
  try {
    const cv::Mat buf(1, static_cast<int>(bytes.size()), CV_8UC1,
                      const_cast<char*>(bytes.data()));
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

Pair decode_pair(const std::string& rgb_bytes, const std::string& ir_bytes) {
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
