// Oturum tablosu: en çok N oturum (en uzun süre kullanılmayan çıkar), boşta kalan silinir, yeni
// açılan oturum işaretlenir (Review Focus 5: silinmiş oturuma istek → sessizce yeni oturum).
#include <gtest/gtest.h>

#include <chrono>

#include "baykus/sessions.hpp"

namespace {

using namespace std::chrono_literals;

baykus::Det det(int index, double x) { return baykus::Det{index, 0, 0.9f, 0.9, x, 0, x + 40, 40}; }

struct FakeClock {
  baykus::Sessions::TimePoint now{};
  baykus::Sessions::Clock fn() {
    return [this] { return now; };
  }
};

baykus::Sessions make(std::size_t max, FakeClock* clock = nullptr) {
  if (clock != nullptr) return baykus::Sessions({}, 5, 3, 3.0, max, 600s, clock->fn());
  return baykus::Sessions({}, 5, 3, 3.0, max);
}

}  // namespace

TEST(Sessions, EvictsOldestAndFlagsNew) {
  FakeClock clock;
  auto s = make(2, &clock);
  bool created = false;
  s.get("a", &created);
  EXPECT_TRUE(created);
  clock.now += 1s;
  s.get("b", &created);
  EXPECT_TRUE(created);
  clock.now += 1s;
  s.get("a", &created);  // a yenilendi: en eski artık b
  EXPECT_FALSE(created);
  clock.now += 1s;
  s.get("c", &created);
  EXPECT_TRUE(created);
  EXPECT_EQ(s.size(), 2u);
  s.get("a", &created);
  EXPECT_FALSE(created);
  s.get("b", &created);  // çıkarılmıştı → yeni
  EXPECT_TRUE(created);
}

TEST(Sessions, IdleExpiry) {
  FakeClock clock;
  auto s = make(8, &clock);
  bool created = false;
  s.get("a", &created);
  clock.now += 600s;  // sınır: tam 10 dk boşta kalan yaşar
  s.get("a", &created);
  EXPECT_FALSE(created);
  clock.now += 601s;
  s.get("a", &created);
  EXPECT_TRUE(created);
}

TEST(Sessions, ResetRestartsTrackIds) {
  auto s = make(8);
  bool created = false;
  auto a = s.get("a", &created);
  a->tracker.update({det(0, 10)});
  EXPECT_EQ(a->tracker.update({det(0, 10)}).at(0).second, 0);
  s.reset("a");
  a = s.get("a", &created);
  EXPECT_FALSE(created);
  a->tracker.update({det(1, 300)});
  EXPECT_EQ(a->tracker.update({det(1, 300)}).at(0).second, 0);  // sıfırlanmasa 1 olurdu
}

TEST(Sessions, HeldSessionSurvivesEviction) {
  auto s = make(1);
  bool created = false;
  auto a = s.get("a", &created);
  s.get("b", &created);  // a tablodan çıktı
  a->tracker.update({det(0, 10)});
  EXPECT_EQ(a->tracker.update({det(0, 10)}).at(0).second, 0);
  s.get("a", &created);
  EXPECT_TRUE(created);
}
