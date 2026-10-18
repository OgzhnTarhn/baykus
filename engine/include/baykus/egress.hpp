// Kapalı ağ kanıtı 2, motor tarafı (Faz 3 tasarımı §4.5): dış adrese TCP bağlantısı ve DNS çözümü
// denenir. Geçer yalnız beklenen ağ hatalarında (ENETUNREACH, EHOSTUNREACH, ECONNREFUSED,
// ETIMEDOUT, EAI_AGAIN, ENOTFOUND); bağlantı kurulursa ya da başka hata çıkarsa düşer. ui
// imajındaki scripts/egress_check.mjs ile aynı kural ve aynı JSON biçimi.
#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace baykus::egress {

struct Probe {
  std::string target;
  std::string result;  // hata adı ya da "connected" / "resolved <adres>"
  bool connected = false;
  bool expected = false;
};

inline constexpr const char* kTcpHost = "1.1.1.1";
inline constexpr int kTcpPort = 443;
inline constexpr const char* kDnsHost = "example.com";
inline constexpr int kTimeoutMs = 3000;

bool expected(const std::string& result);
std::string errno_name(int e);
std::string gai_name(int code);  // EAI_NONAME → "ENOTFOUND" (Node ile aynı ad)

Probe probe_tcp(const std::string& ip, int port, int timeout_ms);
Probe probe_dns(const std::string& host);

nlohmann::json to_json(const Probe& p);
nlohmann::json verdict(const Probe& tcp, const Probe& dns);  // {tcp, dns, blocked, message}

}  // namespace baykus::egress
