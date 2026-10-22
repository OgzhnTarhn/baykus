#include <gtest/gtest.h>

#include <string>

#include <nlohmann/json.hpp>

#include "baykus/event_log.hpp"
#include "baykus/motion_config.hpp"
#include "baykus/zones.hpp"

namespace {

using baykus::EventLog;
using baykus::LogCfg;
using baykus::ZoneEvent;

// Tek karelik olay; iz / yön / alarm alanları boş bırakılır.
ZoneEvent event(int n, const std::string& kind, const std::string& zone_id) {
  ZoneEvent e;
  e.n = n;
  e.frame = n + 100;
  e.kind = kind;
  e.zone_id = zone_id;
  e.gen = 1;
  return e;
}

}  // namespace

TEST(EventLog, SeqIncreasesAndSinceIsIncremental) {
  EventLog log(LogCfg{5});
  const auto r1 = log.append(event(1, "enter", "z1"), "Kuzey", "area");
  EXPECT_EQ(r1["seq"].get<long>(), 1);
  EXPECT_EQ(r1["n"].get<int>(), 1);
  EXPECT_EQ(r1["frame"].get<int>(), 101);
  EXPECT_EQ(r1["kind"].get<std::string>(), "enter");
  EXPECT_EQ(r1["zone_id"].get<std::string>(), "z1");
  EXPECT_EQ(r1["zone_name"].get<std::string>(), "Kuzey");
  EXPECT_EQ(r1["zone_kind"].get<std::string>(), "area");
  EXPECT_TRUE(r1["track_id"].is_null());
  EXPECT_TRUE(r1["dir"].is_null());
  EXPECT_TRUE(r1["alarm_id"].is_null());

  log.append(event(2, "exit", "z1"), "Kuzey", "area");
  const auto r3 = log.append(event(3, "enter", ""), "", "");
  EXPECT_EQ(r3["seq"].get<long>(), 3);
  // Boş zone_id / zone_name / zone_kind → null.
  EXPECT_TRUE(r3["zone_id"].is_null());
  EXPECT_TRUE(r3["zone_name"].is_null());
  EXPECT_TRUE(r3["zone_kind"].is_null());

  const auto s0 = log.since(0);
  ASSERT_EQ(s0["events"].size(), 3u);
  EXPECT_EQ(s0["events"][0]["seq"].get<long>(), 1);
  EXPECT_EQ(s0["events"][1]["seq"].get<long>(), 2);
  EXPECT_EQ(s0["events"][2]["seq"].get<long>(), 3);
  EXPECT_EQ(s0["last_seq"].get<long>(), 3);
  EXPECT_EQ(s0["dropped"].get<long>(), 0);

  const auto s2 = log.since(2);
  ASSERT_EQ(s2["events"].size(), 1u);
  EXPECT_EQ(s2["events"][0]["seq"].get<long>(), 3);
  EXPECT_EQ(s2["last_seq"].get<long>(), 3);

  const auto s3 = log.since(3);
  EXPECT_TRUE(s3["events"].empty());
  EXPECT_EQ(s3["last_seq"].get<long>(), 3);
  EXPECT_EQ(s3["dropped"].get<long>(), 0);
}

TEST(EventLog, DropsOldestAtCap) {
  EventLog log(LogCfg{5});
  for (int i = 1; i <= 8; ++i) log.append(event(i, "enter", "z1"), "Kuzey", "area");
  EXPECT_EQ(log.last_seq(), 8);
  EXPECT_EQ(log.dropped(), 3);
  const auto s0 = log.since(0);
  ASSERT_EQ(s0["events"].size(), 5u);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(s0["events"][i]["seq"].get<long>(), i + 4);
  }
  EXPECT_EQ(s0["last_seq"].get<long>(), 8);
  EXPECT_EQ(s0["dropped"].get<long>(), 3);
}

TEST(EventLog, CsvQuotesCommaQuoteAndTurkish) {
  EventLog log(LogCfg{5});
  log.append(event(1, "enter", "z1"), "Kavşak \"A\", kuzey", "area");
  const std::string header =
      "seq,n,frame,kind,zone_id,zone_name,zone_kind,track_id,dir,gen,alarm_id";
  const std::string row = "1,1,101,enter,z1,\"Kavşak \"\"A\"\", kuzey\",area,,,1,";
  EXPECT_EQ(log.csv_since(0), header + "\r\n" + row + "\r\n");

  // reset: kayıtlar, seq ve dropped sıfırlanır; yeni kayıt seq 1'den başlar.
  log.reset();
  EXPECT_EQ(log.last_seq(), 0);
  EXPECT_EQ(log.dropped(), 0);
  EXPECT_EQ(log.csv_since(0), header + "\r\n");
  const auto r = log.append(event(1, "enter", "z1"), "Kuzey", "area");
  EXPECT_EQ(r["seq"].get<long>(), 1);
}
