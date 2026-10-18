#pragma once

#include <cstddef>
#include <string>

namespace baykus {

// Dosyanın bütün baytları; açılamazsa std::runtime_error (yol iletide).
std::string read_bytes(const std::string& path);

// Küçük harfli onaltılık SHA-256.
std::string sha256_hex(const void* data, std::size_t size);
std::string sha256_hex(const std::string& data);

}  // namespace baykus
