#include "baykus/files.hpp"

#include <openssl/evp.h>
#include <zlib.h>

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

std::string read_gzip(const std::string& path) {
  gzFile f = gzopen(path.c_str(), "rb");
  if (f == nullptr) throw std::runtime_error("açılamadı: " + path);
  std::string out;
  char buf[1 << 16];
  int n = 0;
  while ((n = gzread(f, buf, sizeof(buf))) > 0) out.append(buf, static_cast<std::size_t>(n));
  int err = Z_OK;
  const std::string msg = n < 0 ? gzerror(f, &err) : "";
  gzclose(f);
  if (n < 0) throw std::runtime_error("gzip okunamadı: " + path + " (" + msg + ")");
  return out;
}

void write_gzip(const std::string& path, const std::string& data) {
  gzFile f = gzopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("yazılamadı: " + path);
  const bool ok = data.empty() || gzwrite(f, data.data(), static_cast<unsigned>(data.size())) ==
                                      static_cast<int>(data.size());
  if (gzclose(f) != Z_OK || !ok) throw std::runtime_error("gzip yazılamadı: " + path);
}

}  // namespace baykus
