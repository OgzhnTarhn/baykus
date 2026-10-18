#include "baykus/files.hpp"

#include <openssl/evp.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace baykus {

std::string read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("okunamadı: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string sha256_hex(const void* data, std::size_t size) {
  unsigned char d[EVP_MAX_MD_SIZE];
  unsigned int n = 0;
  if (EVP_Digest(data, size, d, &n, EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("sha256 hesaplanamadı");
  }
  std::ostringstream ss;
  for (unsigned int i = 0; i < n; ++i)
    ss << std::hex << std::setw(2) << std::setfill('0') << int(d[i]);
  return ss.str();
}

std::string sha256_hex(const std::string& data) { return sha256_hex(data.data(), data.size()); }

}  // namespace baykus
