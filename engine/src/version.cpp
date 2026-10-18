#include "baykus/version.hpp"

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>

namespace baykus {

std::string opencv_version() { return cv::getVersionString(); }

std::string onnxruntime_version() { return Ort::GetVersionString(); }

std::string opencv_build_information() { return cv::getBuildInformation(); }

}  // namespace baykus
