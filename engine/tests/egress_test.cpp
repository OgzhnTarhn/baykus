// Kapalı ağ kanıtı 2, motor tarafı (Faz 3 tasarımı §4.5; plan Görev 33): `baykus_server
// --egress-check` ile ui imajındaki scripts/egress_check.mjs aynı kuralı uygular. Ağ sonucu yerel
// soketlerle sınanır (dış adres yok).
#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>

#include "baykus/egress.hpp"

namespace {

// 127.0.0.1'de dinleyen soket; port 0 → çekirdek seçer.
struct Listener {
  int fd = -1;
  int port = 0;
  Listener() {
    fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a);
    ::listen(fd, 1);
    socklen_t n = sizeof a;
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &n);
    port = ntohs(a.sin_port);
  }
  ~Listener() { ::close(fd); }
};

}  // namespace

TEST(Egress, ExpectedErrorsAreTheDesignList) {
  using baykus::egress::expected;
  for (const char* r :
       {"ENETUNREACH", "EHOSTUNREACH", "ECONNREFUSED", "ETIMEDOUT", "EAI_AGAIN", "ENOTFOUND"}) {
    EXPECT_TRUE(expected(r)) << r;
  }
  for (const char* r : {"EACCES", "EPERM", "EAI_MEMORY", "connected", ""}) {
    EXPECT_FALSE(expected(r)) << r;
  }
}

TEST(Egress, ErrorNames) {
  EXPECT_EQ(baykus::egress::errno_name(ENETUNREACH), "ENETUNREACH");
  EXPECT_EQ(baykus::egress::errno_name(ECONNREFUSED), "ECONNREFUSED");
  EXPECT_EQ(baykus::egress::gai_name(EAI_AGAIN), "EAI_AGAIN");
  EXPECT_EQ(baykus::egress::gai_name(EAI_NONAME), "ENOTFOUND");  // Node'daki adıyla
}

TEST(Egress, ConnectionThatSucceedsFails) {
  Listener l;
  const auto tcp = baykus::egress::probe_tcp("127.0.0.1", l.port, 3000);
  EXPECT_TRUE(tcp.connected);
  EXPECT_FALSE(tcp.expected);
  const baykus::egress::Probe dns{"example.com", "EAI_AGAIN", false, true};
  EXPECT_FALSE(baykus::egress::verdict(tcp, dns).at("blocked").get<bool>());
}

TEST(Egress, RefusedCountsAsBlocked) {
  int port = 0;
  {
    Listener l;
    port = l.port;
  }  // kapandı: bağlantı reddedilir
  const auto tcp = baykus::egress::probe_tcp("127.0.0.1", port, 3000);
  EXPECT_EQ(tcp.result, "ECONNREFUSED");
  EXPECT_TRUE(tcp.expected);
  const baykus::egress::Probe dns{"example.com", "ENOTFOUND", false, true};
  const auto v = baykus::egress::verdict(tcp, dns);
  EXPECT_TRUE(v.at("blocked").get<bool>());
  EXPECT_EQ(v.at("tcp").at("result"), "ECONNREFUSED");
}

TEST(Egress, UnexpectedErrorCannotProveTheBlock) {
  const baykus::egress::Probe tcp{"1.1.1.1:443", "EACCES", false, false};
  const baykus::egress::Probe dns{"example.com", "EAI_AGAIN", false, true};
  const auto v = baykus::egress::verdict(tcp, dns);
  EXPECT_FALSE(v.at("blocked").get<bool>());
  EXPECT_EQ(v.at("message"), "ağ engeli kanıtlanamadı");
}
