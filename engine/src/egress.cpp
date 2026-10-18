#include "baykus/egress.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <set>

namespace baykus::egress {

bool expected(const std::string& result) {
  static const std::set<std::string> kExpected = {"ENETUNREACH", "EHOSTUNREACH", "ECONNREFUSED",
                                                  "ETIMEDOUT",   "EAI_AGAIN",    "ENOTFOUND"};
  return kExpected.count(result) > 0;
}

std::string errno_name(int e) {
  switch (e) {
    case ENETUNREACH:
      return "ENETUNREACH";
    case EHOSTUNREACH:
      return "EHOSTUNREACH";
    case ECONNREFUSED:
      return "ECONNREFUSED";
    case ETIMEDOUT:
      return "ETIMEDOUT";
    case EACCES:
      return "EACCES";
    case EPERM:
      return "EPERM";
    case EAFNOSUPPORT:
      return "EAFNOSUPPORT";
    default:
      return "errno " + std::to_string(e);
  }
}

std::string gai_name(int code) {
  switch (code) {
    case EAI_AGAIN:
      return "EAI_AGAIN";
    case EAI_NONAME:
      return "ENOTFOUND";
#ifdef EAI_NODATA
    case EAI_NODATA:
      return "ENOTFOUND";
#endif
    case EAI_FAIL:
      return "EAI_FAIL";
    case EAI_MEMORY:
      return "EAI_MEMORY";
    case EAI_SYSTEM:
      return "EAI_SYSTEM " + errno_name(errno);
    default:
      return "EAI " + std::to_string(code);
  }
}

namespace {

Probe make(std::string target, std::string result, bool connected) {
  const bool ok = !connected && expected(result);
  return {std::move(target), std::move(result), connected, ok};
}

}  // namespace

Probe probe_tcp(const std::string& ip, int port, int timeout_ms) {
  const std::string target = ip + ":" + std::to_string(port);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) return make(target, "EINVAL", false);
  const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if (fd < 0) return make(target, errno_name(errno), false);
  Probe out;
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0) {
    out = make(target, "connected", true);
  } else if (errno != EINPROGRESS) {
    out = make(target, errno_name(errno), false);
  } else {
    pollfd p{fd, POLLOUT, 0};
    const int n = ::poll(&p, 1, timeout_ms);
    if (n == 0) {
      out = make(target, "ETIMEDOUT", false);
    } else if (n < 0) {
      out = make(target, errno_name(errno), false);
    } else {
      int err = 0;
      socklen_t len = sizeof err;
      ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
      out = err == 0 ? make(target, "connected", true) : make(target, errno_name(err), false);
    }
  }
  ::close(fd);
  return out;
}

Probe probe_dns(const std::string& host) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const int rc = ::getaddrinfo(host.c_str(), nullptr, &hints, &res);
  if (rc != 0) return make(host, gai_name(rc), false);
  char buf[INET6_ADDRSTRLEN] = "?";
  if (res->ai_family == AF_INET) {
    ::inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr, buf, sizeof buf);
  } else if (res->ai_family == AF_INET6) {
    ::inet_ntop(AF_INET6, &reinterpret_cast<sockaddr_in6*>(res->ai_addr)->sin6_addr, buf,
                sizeof buf);
  }
  ::freeaddrinfo(res);
  return make(host, std::string("resolved ") + buf, true);
}

nlohmann::json to_json(const Probe& p) {
  return {{"target", p.target},
          {"result", p.result},
          {"connected", p.connected},
          {"expected", p.expected}};
}

nlohmann::json verdict(const Probe& tcp, const Probe& dns) {
  const bool connected = tcp.connected || dns.connected;
  const bool blocked = !connected && tcp.expected && dns.expected;
  const char* message = blocked     ? "dış ağ kapalı"
                        : connected ? "dış adrese ulaşıldı"
                                    : "ağ engeli kanıtlanamadı";
  return {{"tcp", to_json(tcp)}, {"dns", to_json(dns)}, {"blocked", blocked}, {"message", message}};
}

}  // namespace baykus::egress
