#include "fixtures.hpp"

#include <openssl/evp.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace baykus::test {

namespace {

std::filesystem::path env_path(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return std::filesystem::path(v != nullptr && *v != '\0' ? v : fallback);
}

}  // namespace

std::filesystem::path fixtures_root() { return env_path("BAYKUS_FIXTURES", "/fixtures"); }

std::filesystem::path models_root() { return env_path("BAYKUS_MODELS", "/models"); }

bool fixtures_required() {
  const char* v = std::getenv("BAYKUS_REQUIRE_FIXTURES");
  return v != nullptr && std::string(v) == "1";
}

std::string missing(const std::vector<std::filesystem::path>& paths) {
  std::string out;
  for (const auto& p : paths) {
    if (!std::filesystem::exists(p)) {
      out += (out.empty() ? "" : ", ") + p.string();
    }
  }
  return out;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("okunamadı: " + path.string());
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

json read_json(const std::filesystem::path& path) { return json::parse(read_file(path)); }

json load_index() { return read_json(fixtures_root() / "index.json"); }

std::vector<json> fixtures_of(const std::string& kind) {
  std::vector<json> out;
  for (const auto& e : load_index()) {
    if (e.at("kind").get<std::string>() == kind) {
      out.push_back(e);
    }
  }
  return out;
}

std::string sha256_hex(const void* data, std::size_t size) {
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int len = 0;
  if (EVP_Digest(data, size, digest, &len, EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("sha256 hesaplanamadı");
  }
  std::ostringstream ss;
  for (unsigned int i = 0; i < len; ++i) {
    ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[i]);
  }
  return ss.str();
}

std::vector<Det> dets_from_json(const json& j) {
  std::vector<Det> out;
  for (const auto& d : j) {
    const auto& b = d.at("box");
    out.push_back(Det{d.at("index").get<int>(), d.at("cls_id").get<int>(),
                      static_cast<float>(d.at("score_raw").get<double>()), d.at("score").get<double>(),
                      b.at(0).get<double>(), b.at(1).get<double>(), b.at(2).get<double>(),
                      b.at(3).get<double>()});
  }
  return out;
}

}  // namespace baykus::test
