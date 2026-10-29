#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "baykus/ego.hpp"
#include "baykus/geometry.hpp"
#include "baykus/motion.hpp"
#include "baykus/motion_config.hpp"
#include "baykus/zones.hpp"

namespace {

using baykus::Box;
using baykus::EgoResult;
using baykus::MotionCfg;
using baykus::TrackMotion;
using baykus::TrackOut;
using baykus::ZoneDef;
using baykus::ZoneEvent;
using baykus::Zones;
using baykus::ZonesCfg;

const ZonesCfg& cfg() {
  const char* env = std::getenv("BAYKUS_CONFIGS");
  static const MotionCfg c =
      MotionCfg::load(std::string(env && *env ? env : "/configs") + "/engine/motion.json");
  return c.zones;
}

// Birim harita dönüşümü; `ok` false ise kayıp kare.
EgoResult ego(int gen = 1, bool provisional = false, bool ok = true) {
  EgoResult r;
  r.ok = ok;
  r.source = ok ? "ir" : "none";
  r.map_gen = gen;
  r.provisional = provisional;
  r.frame_to_map = ok ? std::optional<cv::Matx23d>(baykus::similarity(0, 0, 0, 1.0)) : std::nullopt;
  r.inliers = std::nullopt;
  r.inlier_ratio = std::nullopt;
  r.keyframe = false;
  r.relinked = false;
  r.relocalized = false;
  r.merge = std::nullopt;
  return r;
}

// Tek karelik iz gözlemi; `xy` yoksa iz görünür ama harita noktası yok.
struct Obs {
  int id;
  std::optional<cv::Point2d> xy;
  std::string motion = "moving";
};

// Kareyi kurup `zones.update(n, n + 100, ...)` döner.
std::vector<ZoneEvent> step(Zones& z, int n, const std::vector<Obs>& obs,
                            const EgoResult& e = ego()) {
  std::vector<TrackOut> tracks;
  std::map<int, TrackMotion> motion;
  for (const auto& o : obs) {
    const cv::Point2d p = o.xy.value_or(cv::Point2d(0.0, 0.0));
    tracks.push_back(TrackOut{o.id, 0, Box{p.x - 2, p.y - 2, p.x + 2, p.y + 2}, 0.9f, 0.9});
    TrackMotion m;
    m.map_xy = o.xy;
    m.motion = o.motion;
    motion[o.id] = m;
  }
  return z.update(n, n + 100, e, tracks, motion);
}

int kinds(const std::vector<ZoneEvent>& events, const std::string& kind) {
  int c = 0;
  for (const auto& e : events) {
    if (e.kind == kind) ++c;
  }
  return c;
}

ZoneDef square(const std::string& id, const std::string& kind, double x0, double y0, double side) {
  ZoneDef d;
  d.id = id;
  d.name = id;
  d.kind = kind;
  d.gen = 1;
  d.points = {cv::Point2d(x0, y0), cv::Point2d(x0 + side, y0), cv::Point2d(x0 + side, y0 + side),
              cv::Point2d(x0, y0 + side)};
  return d;
}

ZoneDef line(const std::string& id, cv::Point2d a, cv::Point2d b, const std::string& count) {
  ZoneDef d;
  d.id = id;
  d.name = id;
  d.kind = "line";
  d.gen = 1;
  d.points = {a, b};
  d.count = count;
  return d;
}

// Bölge gövdesi: tek `area` bölgesi, verilen gen ile.
nlohmann::json area_body(int gen) {
  nlohmann::json zone = {{"id", "z1"},
                         {"name", "z1"},
                         {"kind", "area"},
                         {"gen", gen},
                         {"points", {{0, 0}, {10, 0}, {10, 10}, {0, 10}}}};
  return nlohmann::json{{"zones", nlohmann::json::array({zone})}};
}

}  // namespace

TEST(Zones, EnterNeedsThreeObs) {
  Zones z(cfg());
  z.set({square("z1", "area", 0, 0, 100)}, 1);
  const std::vector<Obs> inside{{1, cv::Point2d(50, 50)}};
  EXPECT_EQ(kinds(step(z, 1, inside), "enter"), 0);
  EXPECT_EQ(kinds(step(z, 2, inside), "enter"), 0);
  const auto ev = step(z, 3, inside);
  ASSERT_EQ(kinds(ev, "enter"), 1);
  for (const auto& e : ev) {
    if (e.kind != "enter") continue;
    EXPECT_EQ(e.zone_id, "z1");
    EXPECT_EQ(e.track_id, 1);
    EXPECT_EQ(e.n, 1);  // koşunun ilk gözlemi
    EXPECT_EQ(e.frame, 101);
  }
}

TEST(Zones, EdgePointInside) {
  Zones z(cfg());
  z.set({square("z1", "area", 0, 0, 100)}, 1);
  // Sol kenar üstündeki nokta içeride sayılır.
  const std::vector<Obs> edge{{1, cv::Point2d(0, 50)}};
  EXPECT_EQ(kinds(step(z, 1, edge), "enter"), 0);
  EXPECT_EQ(kinds(step(z, 2, edge), "enter"), 0);
  EXPECT_EQ(kinds(step(z, 3, edge), "enter"), 1);
}

TEST(Zones, ConcavePolygon) {
  Zones z(cfg());
  ZoneDef d;
  d.id = "z1";
  d.name = "L";
  d.kind = "area";
  d.gen = 1;
  d.points = {cv::Point2d(0, 0),   cv::Point2d(100, 0),  cv::Point2d(100, 40),
              cv::Point2d(40, 40), cv::Point2d(40, 100), cv::Point2d(0, 100)};
  z.set({d}, 1);
  // Girintideki nokta dışarıda: enter yok.
  const std::vector<Obs> notch{{1, cv::Point2d(70, 70)}};
  for (int n = 1; n <= 3; ++n) EXPECT_EQ(kinds(step(z, n, notch), "enter"), 0);
  // Dikey koldaki nokta içeride: 3 gözlemde enter.
  const std::vector<Obs> arm{{1, cv::Point2d(20, 70)}};
  EXPECT_EQ(kinds(step(z, 4, arm), "enter"), 0);
  EXPECT_EQ(kinds(step(z, 5, arm), "enter"), 0);
  EXPECT_EQ(kinds(step(z, 6, arm), "enter"), 1);
}

TEST(Zones, AckMarksAlarm) {
  Zones z(cfg());
  z.set({square("z1", "restricted", 0, 0, 100)}, 1);
  const std::vector<Obs> inside{{1, cv::Point2d(50, 50)}};
  std::vector<ZoneEvent> ev;
  for (int n = 1; n <= 3; ++n) ev = step(z, n, inside);
  ASSERT_EQ(kinds(ev, "alarm_start"), 1);
  const auto alarms = z.alarms();
  ASSERT_EQ(alarms.size(), 1u);
  EXPECT_EQ(alarms[0].kind, "intrusion");
  EXPECT_FALSE(alarms[0].acked);

  std::vector<ZoneEvent> out;
  EXPECT_TRUE(z.ack(alarms[0].id, 10, 110, &out));
  ASSERT_EQ(kinds(out, "alarm_ack"), 1);
  EXPECT_EQ(out[0].alarm_id, alarms[0].id);
  EXPECT_TRUE(z.alarms()[0].acked);

  std::vector<ZoneEvent> out2;
  EXPECT_FALSE(z.ack("yok", 11, 111, &out2));
  EXPECT_TRUE(out2.empty());
}

TEST(Zones, ValidationRejectsWithoutChange) {
  Zones z(cfg());
  z.set({square("z1", "area", 0, 0, 100), square("z2", "restricted", 200, 0, 100)}, 5);
  const std::vector<Obs> inside2{{1, cv::Point2d(250, 50)}};
  for (int n = 1; n <= 3; ++n) step(z, n, inside2);
  ASSERT_EQ(z.alarms().size(), 1u);
  const auto defs_before = z.defs_json();
  const auto counts_before = z.counts();
  const auto alarms_before = z.alarms().size();

  // Her geçersiz set hiçbir durumu değiştirmemeli.
  auto reject = [&](const std::vector<ZoneDef>& defs) {
    EXPECT_THROW(z.set(defs, 5), std::invalid_argument);
    EXPECT_EQ(z.defs_json(), defs_before);
    EXPECT_EQ(z.counts(), counts_before);
    EXPECT_EQ(z.alarms().size(), alarms_before);
  };

  // 17 bölge (max_zones 16).
  std::vector<ZoneDef> many;
  for (int i = 0; i < 17; ++i) {
    many.push_back(square("z" + std::to_string(i), "area", 200.0 * i, 0, 100));
  }
  reject(many);

  // 33 noktalı çokgen (max_points 32).
  ZoneDef circle;
  circle.id = "zc";
  circle.name = "zc";
  circle.kind = "area";
  circle.gen = 1;
  for (int i = 0; i < 33; ++i) {
    const double a = 6.283185307179586 * i / 33.0;
    circle.points.push_back(cv::Point2d(100.0 * std::cos(a), 100.0 * std::sin(a)));
  }
  reject({circle});

  // Papyon (kendini kesen çokgen).
  ZoneDef bow;
  bow.id = "zb";
  bow.name = "zb";
  bow.kind = "area";
  bow.gen = 1;
  bow.points = {cv::Point2d(0, 0), cv::Point2d(100, 100), cv::Point2d(100, 0), cv::Point2d(0, 100)};
  reject({bow});

  // Yinelenen id.
  reject({square("z1", "area", 0, 0, 100), square("z1", "area", 400, 0, 100)});

  // gen = max_gen + 1.
  ZoneDef bad_gen = square("zg", "area", 0, 0, 100);
  bad_gen.gen = 6;
  reject({bad_gen});

  // 41 karakterlik Türkçe harfli ad (sınır 40 karakter).
  std::string long_name;
  for (int i = 0; i < 41; ++i) long_name += "\xC4\x9F";  // "ğ"
  ZoneDef bad_name = square("zn", "area", 0, 0, 100);
  bad_name.name = long_name;
  reject({bad_name});

  // dwell_frames 0.
  ZoneDef bad_dwell = square("zd", "dwell", 0, 0, 100);
  bad_dwell.dwell_frames = 0;
  reject({bad_dwell});

  // 40 karakterlik Türkçe harfli ad kabul edilir.
  std::string ok_name;
  for (int i = 0; i < 40; ++i) ok_name += "\xC4\x9F";
  ZoneDef ok = square("zn", "area", 0, 0, 100);
  ok.name = ok_name;
  EXPECT_NO_THROW(z.set({ok}, 5));
}

TEST(Zones, SuspendedOnOtherGen) {
  Zones z(cfg());
  z.set({square("z1", "restricted", 0, 0, 100)}, 3);
  const std::vector<Obs> inside{{1, cv::Point2d(50, 50)}};
  for (int n = 1; n <= 3; ++n) step(z, n, inside);
  ASSERT_EQ(z.alarms().size(), 1u);
  const std::string alarm_id = z.alarms()[0].id;

  // Farklı map_gen: askı; bu update'te açık alarm temizlenir.
  const auto ev = step(z, 4, inside, ego(2));
  ASSERT_EQ(kinds(ev, "alarm_clear"), 1);
  EXPECT_EQ(ev.size(), 1u);
  for (const auto& e : ev) {
    if (e.kind != "alarm_clear") continue;
    EXPECT_EQ(e.zone_id, "z1");
    EXPECT_EQ(e.alarm_id, alarm_id);
    EXPECT_FALSE(e.track_id.has_value());
    EXPECT_EQ(e.n, 4);
    EXPECT_EQ(e.frame, 104);
  }
  EXPECT_EQ(z.counts()["z1"]["status"].get<std::string>(), "suspended");
  EXPECT_EQ(z.counts()["z1"]["inside"].get<int>(), 0);
  EXPECT_TRUE(z.alarms().empty());

  // Ayrı senaryo: aynı gen ama provisional = true da askıda.
  Zones z2(cfg());
  z2.set({square("z1", "restricted", 0, 0, 100)}, 3);
  for (int n = 1; n <= 3; ++n) step(z2, n, inside);
  ASSERT_EQ(z2.alarms().size(), 1u);
  const auto ev2 = step(z2, 4, inside, ego(1, true));
  EXPECT_EQ(kinds(ev2, "alarm_clear"), 1);
  EXPECT_EQ(z2.counts()["z1"]["status"].get<std::string>(), "suspended");
  EXPECT_EQ(z2.counts()["z1"]["inside"].get<int>(), 0);
  EXPECT_TRUE(z2.alarms().empty());
}

TEST(Zones, ReactivatedOnMerge) {
  Zones z(cfg());
  z.set({square("z1", "area", 0, 0, 100)}, 3);
  const std::vector<Obs> inside{{1, cv::Point2d(50, 50)}};
  // gen 2 + provisional: askıda, gözlem yapılmaz.
  for (int n = 1; n <= 3; ++n) {
    const auto ev = step(z, n, inside, ego(2, true));
    EXPECT_TRUE(ev.empty());
    EXPECT_EQ(z.counts()["z1"]["status"].get<std::string>(), "suspended");
    EXPECT_EQ(z.counts()["z1"]["inside"].get<int>(), 0);
  }
  // Birleşme (merge): gen 1, provisional false → yeniden aktif, boş durumdan devam.
  EXPECT_EQ(kinds(step(z, 4, inside, ego(1, false)), "enter"), 0);
  EXPECT_EQ(kinds(step(z, 5, inside, ego(1, false)), "enter"), 0);
  const auto ev = step(z, 6, inside, ego(1, false));
  ASSERT_EQ(kinds(ev, "enter"), 1);
  EXPECT_EQ(z.counts()["z1"]["status"].get<std::string>(), "active");
  for (const auto& e : ev) {
    if (e.kind != "enter") continue;
    EXPECT_EQ(e.zone_id, "z1");
    EXPECT_EQ(e.track_id, 1);
    EXPECT_EQ(e.n, 4);  // koşunun ilk gözlemi
    EXPECT_EQ(e.frame, 104);
  }
}

TEST(Zones, ResetTracksKeepsZones) {
  Zones z(cfg());
  z.set({square("z1", "restricted", 0, 0, 100)}, 3);
  const std::vector<Obs> inside{{1, cv::Point2d(50, 50)}};
  for (int n = 1; n <= 3; ++n) step(z, n, inside);
  ASSERT_EQ(z.alarms().size(), 1u);
  const std::string alarm_id = z.alarms()[0].id;
  const auto defs_before = z.defs_json();

  z.reset_tracks();
  EXPECT_EQ(z.defs_json(), defs_before);
  ASSERT_EQ(z.alarms().size(), 1u);  // tanımlar ve açık alarmlar kalır

  // Sonraki update alarm_clear üretir (iz durumu silinmiş).
  const auto ev = step(z, 4, inside);
  EXPECT_EQ(kinds(ev, "alarm_clear"), 1);
  EXPECT_EQ(kinds(ev, "enter"), 0);
  for (const auto& e : ev) {
    if (e.kind != "alarm_clear") continue;
    EXPECT_EQ(e.alarm_id, alarm_id);
  }
  EXPECT_TRUE(z.alarms().empty());

  // İz 3 gözlemde yeniden enter.
  EXPECT_EQ(kinds(step(z, 5, inside), "enter"), 0);
  const auto ev2 = step(z, 6, inside);
  ASSERT_EQ(kinds(ev2, "enter"), 1);
  for (const auto& e : ev2) {
    if (e.kind != "enter") continue;
    EXPECT_EQ(e.zone_id, "z1");
    EXPECT_EQ(e.track_id, 1);
    EXPECT_EQ(e.n, 4);  // reset sonrası koşunun ilk gözlemi
    EXPECT_EQ(e.frame, 104);
  }
}

TEST(Zones, ResetAllClears) {
  Zones z(cfg());
  z.set({square("z1", "restricted", 0, 0, 100), square("z2", "area", 200, 0, 100)}, 3);
  const std::vector<Obs> inside{{1, cv::Point2d(50, 50)}};
  for (int n = 1; n <= 3; ++n) step(z, n, inside);
  ASSERT_FALSE(z.defs_json().empty());
  ASSERT_FALSE(z.alarms().empty());
  ASSERT_FALSE(z.counts().empty());

  z.reset();
  EXPECT_TRUE(z.defs_json().empty());
  EXPECT_TRUE(z.alarms().empty());
  EXPECT_TRUE(z.counts().empty());
}

TEST(Zones, UpdateKeepsUnchangedZoneState) {
  Zones z(cfg());
  z.set({square("z1", "restricted", 0, 0, 100), square("z2", "restricted", 200, 0, 100)}, 1);
  const std::vector<Obs> both{{1, cv::Point2d(50, 50)}, {2, cv::Point2d(250, 50)}};
  std::vector<ZoneEvent> ev;
  for (int n = 1; n <= 3; ++n) ev = step(z, n, both);
  ASSERT_EQ(kinds(ev, "alarm_start"), 2);
  ASSERT_EQ(z.alarms().size(), 2u);
  ASSERT_EQ(z.alarms()[0].zone_id, "z1");
  ASSERT_EQ(z.alarms()[1].zone_id, "z2");
  const std::string z1_alarm = z.alarms()[0].id;
  const std::string z2_alarm = z.alarms()[1].id;

  // z1 değişmemiş (yalnız ad), z2'nin noktaları değişmiş: durumu ve alarmı taşınmaz.
  ZoneDef z1 = square("z1", "restricted", 0, 0, 100);
  z1.name = "yeni";
  std::vector<ZoneEvent> cleared;
  z.set({z1, square("z2", "restricted", 200, 0, 120)}, 1, &cleared);
  ASSERT_EQ(cleared.size(), 1u);
  EXPECT_EQ(kinds(cleared, "alarm_clear"), 1);
  EXPECT_EQ(cleared[0].zone_id, "z2");
  EXPECT_EQ(cleared[0].alarm_id, z2_alarm);
  EXPECT_EQ(cleared[0].n, 3);
  EXPECT_EQ(cleared[0].frame, 103);

  ASSERT_EQ(z.alarms().size(), 1u);
  EXPECT_EQ(z.alarms()[0].id, z1_alarm);
  EXPECT_EQ(z.alarms()[0].zone_id, "z1");
  EXPECT_EQ(z.counts()["z1"]["inside"].get<int>(), 1);
  EXPECT_EQ(z.counts()["z2"]["inside"].get<int>(), 0);
  EXPECT_EQ(z.defs_json()[0]["name"].get<std::string>(), "yeni");

  // z1'de yeni enter yok; z2 boş durumdan 3 gözlemde yeniden enter + yeni alarm.
  EXPECT_EQ(kinds(step(z, 4, both), "enter"), 0);
  EXPECT_EQ(kinds(step(z, 5, both), "enter"), 0);
  ev = step(z, 6, both);
  ASSERT_EQ(kinds(ev, "enter"), 1);
  for (const auto& e : ev) {
    if (e.kind != "enter") continue;
    EXPECT_EQ(e.zone_id, "z2");
    EXPECT_EQ(e.track_id, 2);
    EXPECT_EQ(e.n, 4);  // yeni koşunun ilk gözlemi
    EXPECT_EQ(e.frame, 104);
  }
  ASSERT_EQ(kinds(ev, "alarm_start"), 1);
  ASSERT_EQ(z.alarms().size(), 2u);
  EXPECT_EQ(z.alarms()[1].zone_id, "z2");
  EXPECT_EQ(z.alarms()[1].kind, "intrusion");
  EXPECT_NE(z.alarms()[1].id, z2_alarm);
}

TEST(Zones, GapUpToFiveKeepsRun) {
  Zones z(cfg());
  z.set({square("z1", "area", 0, 0, 100)}, 1);
  const std::vector<Obs> inside{{1, cv::Point2d(50, 50)}};
  const std::vector<Obs> lost{{1, std::nullopt}};
  EXPECT_EQ(kinds(step(z, 1, inside), "enter"), 0);
  EXPECT_EQ(kinds(step(z, 2, inside), "enter"), 0);
  // gap_max = 5: beş `yok` koşuyu bozmaz.
  for (int n = 3; n <= 7; ++n) EXPECT_EQ(kinds(step(z, n, lost), "enter"), 0);
  const auto ev = step(z, 8, inside);
  ASSERT_EQ(kinds(ev, "enter"), 1);
  for (const auto& e : ev) {
    if (e.kind != "enter") continue;
    EXPECT_EQ(e.zone_id, "z1");
    EXPECT_EQ(e.track_id, 1);
    EXPECT_EQ(e.n, 1);  // koşunun ilk gözlemi
    EXPECT_EQ(e.frame, 101);
  }
}

TEST(Zones, GapSixResetsRun) {
  Zones z(cfg());
  z.set({square("z1", "area", 0, 0, 100)}, 1);
  const std::vector<Obs> inside{{1, cv::Point2d(50, 50)}};
  const std::vector<Obs> lost{{1, std::nullopt}};
  EXPECT_EQ(kinds(step(z, 1, inside), "enter"), 0);
  EXPECT_EQ(kinds(step(z, 2, inside), "enter"), 0);
  // Altı `yok`: koşu sıfırlanır.
  for (int n = 3; n <= 8; ++n) EXPECT_EQ(kinds(step(z, n, lost), "enter"), 0);
  EXPECT_EQ(kinds(step(z, 9, inside), "enter"), 0);
  EXPECT_EQ(kinds(step(z, 10, inside), "enter"), 0);
  const auto ev = step(z, 11, inside);
  ASSERT_EQ(kinds(ev, "enter"), 1);
  for (const auto& e : ev) {
    if (e.kind != "enter") continue;
    EXPECT_EQ(e.track_id, 1);
    EXPECT_EQ(e.n, 9);  // yeni koşunun ilk gözlemi
    EXPECT_EQ(e.frame, 109);
  }
}

TEST(Zones, ExitLostAfterThirty) {
  Zones z(cfg());
  z.set({square("z1", "area", 0, 0, 100)}, 1);
  const std::vector<Obs> inside{{1, cv::Point2d(50, 50)}};
  const std::vector<Obs> gone;
  for (int n = 1; n <= 3; ++n) step(z, n, inside);
  // 29. yok karesine kadar olay yok.
  for (int n = 4; n <= 32; ++n) EXPECT_EQ(kinds(step(z, n, gone), "exit_lost"), 0);
  const auto ev = step(z, 33, gone);
  ASSERT_EQ(kinds(ev, "exit_lost"), 1);
  for (const auto& e : ev) {
    if (e.kind != "exit_lost") continue;
    EXPECT_EQ(e.zone_id, "z1");
    EXPECT_EQ(e.track_id, 1);
    EXPECT_EQ(e.n, 4);  // ilk yok karesi
    EXPECT_EQ(e.frame, 104);
  }
}

TEST(Zones, StoppedOncePerStay) {
  Zones z(cfg());
  z.set({square("z1", "area", 0, 0, 100)}, 1);
  const std::vector<Obs> in_moving{{1, cv::Point2d(50, 50), "moving"}};
  const std::vector<Obs> in_still{{1, cv::Point2d(50, 50), "stationary"}};
  const std::vector<Obs> out_moving{{1, cv::Point2d(200, 200), "moving"}};
  int n = 1;
  for (int i = 0; i < 3; ++i) step(z, n++, in_moving);  // enter
  int stopped = 0;
  for (int i = 0; i < 30; ++i) stopped += kinds(step(z, n++, in_still), "stopped");
  EXPECT_EQ(stopped, 1);
  // Hareket sonrası yeniden 30 durağan: kalış başına bir kez.
  for (int i = 0; i < 5; ++i) step(z, n++, in_moving);
  stopped = 0;
  for (int i = 0; i < 30; ++i) stopped += kinds(step(z, n++, in_still), "stopped");
  EXPECT_EQ(stopped, 0);
  // Çıkış + yeniden giriş: yeni kalış, yeni stopped.
  for (int i = 0; i < 3; ++i) step(z, n++, out_moving);
  for (int i = 0; i < 3; ++i) step(z, n++, in_moving);
  stopped = 0;
  for (int i = 0; i < 30; ++i) stopped += kinds(step(z, n++, in_still), "stopped");
  EXPECT_EQ(stopped, 1);
}

TEST(Zones, StoppedResetByUnknown) {
  Zones z(cfg());
  z.set({square("z1", "area", 0, 0, 100)}, 1);
  const std::vector<Obs> in_moving{{1, cv::Point2d(50, 50), "moving"}};
  const std::vector<Obs> in_still{{1, cv::Point2d(50, 50), "stationary"}};
  const std::vector<Obs> in_unknown{{1, cv::Point2d(50, 50), "unknown"}};
  int n = 1;
  for (int i = 0; i < 3; ++i) step(z, n++, in_moving);  // enter
  for (int i = 0; i < 29; ++i) EXPECT_EQ(kinds(step(z, n++, in_still), "stopped"), 0);
  // unknown durağan koşusunu sıfırlar.
  EXPECT_EQ(kinds(step(z, n++, in_unknown), "stopped"), 0);
  for (int i = 0; i < 29; ++i) EXPECT_EQ(kinds(step(z, n++, in_still), "stopped"), 0);
  EXPECT_EQ(kinds(step(z, n++, in_still), "stopped"), 1);
}

TEST(Zones, IntrusionAlarmMultiTrack) {
  Zones z(cfg());
  z.set({square("z1", "restricted", 0, 0, 100)}, 1);
  const std::vector<Obs> both{{1, cv::Point2d(30, 30)}, {2, cv::Point2d(70, 70)}};
  std::vector<ZoneEvent> ev;
  for (int n = 1; n <= 3; ++n) ev = step(z, n, both);
  ASSERT_EQ(kinds(ev, "alarm_start"), 1);
  ASSERT_EQ(z.alarms().size(), 1u);
  EXPECT_EQ(z.alarms()[0].kind, "intrusion");
  EXPECT_EQ(z.alarms()[0].track_ids, (std::vector<int>{1, 2}));

  // İz 2 çıkar: alarm açık kalır, track_ids tek.
  const std::vector<Obs> only1{{1, cv::Point2d(30, 30)}, {2, cv::Point2d(200, 200)}};
  for (int n = 4; n <= 6; ++n) ev = step(z, n, only1);
  EXPECT_EQ(kinds(ev, "alarm_clear"), 0);
  ASSERT_EQ(z.alarms().size(), 1u);
  EXPECT_EQ(z.alarms()[0].track_ids, (std::vector<int>{1}));

  // Son iz de çıkar: alarm temizlenir.
  const std::vector<Obs> only_out{{1, cv::Point2d(200, 200)}};
  for (int n = 7; n <= 9; ++n) ev = step(z, n, only_out);
  EXPECT_EQ(kinds(ev, "alarm_clear"), 1);
  EXPECT_TRUE(z.alarms().empty());
  EXPECT_EQ(z.counts()["z1"]["entered_total"].get<int>(), 2);
}

TEST(Zones, DwellAlarmAtExactFrames) {
  Zones z(cfg());
  ZoneDef d = square("z1", "dwell", 0, 0, 100);
  d.dwell_frames = 10;
  z.set({d}, 1);
  const std::vector<Obs> out_moving{{1, cv::Point2d(200, 200), "moving"}};
  const std::vector<Obs> in_moving{{1, cv::Point2d(50, 50), "moving"}};
  const std::vector<Obs> in_still{{1, cv::Point2d(50, 50), "stationary"}};
  int n = 1;
  for (int i = 0; i < 3; ++i) step(z, n++, out_moving);
  for (int i = 0; i < 3; ++i) step(z, n++, in_moving);  // enter
  // 9 durağan gözlemde alarm yok.
  for (int i = 0; i < 9; ++i) EXPECT_EQ(kinds(step(z, n++, in_still), "alarm_start"), 0);
  EXPECT_TRUE(z.alarms().empty());
  const auto ev = step(z, n++, in_still);
  ASSERT_EQ(kinds(ev, "alarm_start"), 1);
  ASSERT_EQ(z.alarms().size(), 1u);
  EXPECT_EQ(z.alarms()[0].kind, "dwell_exceeded");
  EXPECT_EQ(z.alarms()[0].id, "z1:dwell_exceeded:1:16");
  EXPECT_EQ(z.alarms()[0].track_ids, (std::vector<int>{1}));
  EXPECT_EQ(z.counts()["z1"]["exceeded_total"].get<int>(), 1);
}

TEST(Zones, DwellClearedByMoving) {
  Zones z(cfg());
  ZoneDef d = square("z1", "dwell", 0, 0, 100);
  d.dwell_frames = 10;
  z.set({d}, 1);
  const std::vector<Obs> out_moving{{1, cv::Point2d(200, 200), "moving"}};
  const std::vector<Obs> in_moving{{1, cv::Point2d(50, 50), "moving"}};
  const std::vector<Obs> in_still{{1, cv::Point2d(50, 50), "stationary"}};
  int n = 1;
  for (int i = 0; i < 3; ++i) step(z, n++, out_moving);
  for (int i = 0; i < 3; ++i) step(z, n++, in_moving);  // enter
  for (int i = 0; i < 10; ++i) step(z, n++, in_still);  // alarm açılır
  ASSERT_EQ(z.alarms().size(), 1u);
  // dwell_clear_moving = 3: iki hareketli gözlemde alarm açık.
  EXPECT_EQ(kinds(step(z, n++, in_moving), "alarm_clear"), 0);
  EXPECT_EQ(kinds(step(z, n++, in_moving), "alarm_clear"), 0);
  ASSERT_EQ(z.alarms().size(), 1u);
  EXPECT_EQ(kinds(step(z, n++, in_moving), "alarm_clear"), 1);
  EXPECT_TRUE(z.alarms().empty());
}

// Hat: A = (0, 0), B = (100, 0); cross(B − A, p − A) = 100 * p.y, +1 = sol = a tarafı.
TEST(Zones, LineBothDirections) {
  Zones z(cfg());
  z.set({line("z1", cv::Point2d(0, 0), cv::Point2d(100, 0), "both")}, 1);
  const std::vector<Obs> left{{1, cv::Point2d(50, 10)}};
  const std::vector<Obs> right{{1, cv::Point2d(50, -10)}};
  int n = 1;
  // Kararlı taraf (sol) 3 gözlemde belirlenir, olay yok.
  for (int i = 0; i < 3; ++i) EXPECT_EQ(kinds(step(z, n++, left), "line_cross"), 0);
  // Sağ × 3: koşunun ilk gözlemi olayda.
  EXPECT_EQ(kinds(step(z, n++, right), "line_cross"), 0);
  EXPECT_EQ(kinds(step(z, n++, right), "line_cross"), 0);
  auto ev = step(z, n++, right);
  ASSERT_EQ(kinds(ev, "line_cross"), 1);
  for (const auto& e : ev) {
    if (e.kind != "line_cross") continue;
    EXPECT_EQ(e.zone_id, "z1");
    EXPECT_EQ(e.track_id, 1);
    EXPECT_EQ(e.dir, "a_to_b");
    EXPECT_EQ(e.n, 4);
    EXPECT_EQ(e.frame, 104);
  }
  // Sol × 3: geri dönüş.
  EXPECT_EQ(kinds(step(z, n++, left), "line_cross"), 0);
  EXPECT_EQ(kinds(step(z, n++, left), "line_cross"), 0);
  ev = step(z, n++, left);
  ASSERT_EQ(kinds(ev, "line_cross"), 1);
  for (const auto& e : ev) {
    if (e.kind != "line_cross") continue;
    EXPECT_EQ(e.dir, "b_to_a");
    EXPECT_EQ(e.n, 7);
    EXPECT_EQ(e.frame, 107);
  }
  EXPECT_EQ(z.counts()["z1"]["a_to_b"].get<int>(), 1);
  EXPECT_EQ(z.counts()["z1"]["b_to_a"].get<int>(), 1);
}

TEST(Zones, LineAroundEndNotCounted) {
  Zones z(cfg());
  z.set({line("z1", cv::Point2d(0, 0), cv::Point2d(100, 0), "both")}, 1);
  // Uçların ötesinde dolaşan geçiş: parça B'nin sağından geçer, olay yok.
  const std::vector<Obs> left{{1, cv::Point2d(95, 10)}};
  const std::vector<Obs> right{{1, cv::Point2d(110, -10)}};
  int n = 1;
  for (int i = 0; i < 3; ++i) EXPECT_EQ(kinds(step(z, n++, left), "line_cross"), 0);
  for (int i = 0; i < 3; ++i) EXPECT_EQ(kinds(step(z, n++, right), "line_cross"), 0);
  EXPECT_EQ(z.counts()["z1"]["a_to_b"].get<int>(), 0);
  EXPECT_EQ(z.counts()["z1"]["b_to_a"].get<int>(), 0);
}

TEST(Zones, LineThroughEndpointCounted) {
  Zones z(cfg());
  z.set({line("z1", cv::Point2d(0, 0), cv::Point2d(100, 0), "both")}, 1);
  // Parça tam B'den geçer (uçlar dahil): sayılır.
  const std::vector<Obs> left{{1, cv::Point2d(90, 10)}};
  const std::vector<Obs> right{{1, cv::Point2d(110, -10)}};
  int n = 1;
  for (int i = 0; i < 3; ++i) EXPECT_EQ(kinds(step(z, n++, left), "line_cross"), 0);
  EXPECT_EQ(kinds(step(z, n++, right), "line_cross"), 0);
  EXPECT_EQ(kinds(step(z, n++, right), "line_cross"), 0);
  const auto ev = step(z, n++, right);
  ASSERT_EQ(kinds(ev, "line_cross"), 1);
  for (const auto& e : ev) {
    if (e.kind != "line_cross") continue;
    EXPECT_EQ(e.dir, "a_to_b");
    EXPECT_EQ(e.n, 4);
    EXPECT_EQ(e.frame, 104);
  }
  EXPECT_EQ(z.counts()["z1"]["a_to_b"].get<int>(), 1);
}

TEST(Zones, LineFlickerNotCounted) {
  Zones z(cfg());
  z.set({line("z1", cv::Point2d(0, 0), cv::Point2d(100, 0), "both")}, 1);
  const std::vector<Obs> left{{1, cv::Point2d(50, 10)}};
  const std::vector<Obs> right{{1, cv::Point2d(50, -10)}};
  int n = 1;
  for (int i = 0; i < 3; ++i) EXPECT_EQ(kinds(step(z, n++, left), "line_cross"), 0);
  // Tek seğirme (sağ × 1) kararlı tarafı bozmaz.
  EXPECT_EQ(kinds(step(z, n++, right), "line_cross"), 0);
  for (int i = 0; i < 3; ++i) EXPECT_EQ(kinds(step(z, n++, left), "line_cross"), 0);
  EXPECT_EQ(z.counts()["z1"]["a_to_b"].get<int>(), 0);
  EXPECT_EQ(z.counts()["z1"]["b_to_a"].get<int>(), 0);
}

TEST(Zones, LineRecrossSeparateEvent) {
  Zones z(cfg());
  z.set({line("z1", cv::Point2d(0, 0), cv::Point2d(100, 0), "both")}, 1);
  const std::vector<Obs> left{{1, cv::Point2d(50, 10)}};
  const std::vector<Obs> right{{1, cv::Point2d(50, -10)}};
  int n = 1;
  int events = 0;
  for (int i = 0; i < 3; ++i) EXPECT_EQ(kinds(step(z, n++, left), "line_cross"), 0);
  for (int i = 0; i < 3; ++i) events += kinds(step(z, n++, right), "line_cross");
  EXPECT_EQ(events, 1);  // a_to_b
  events = 0;
  for (int i = 0; i < 3; ++i) events += kinds(step(z, n++, left), "line_cross");
  EXPECT_EQ(events, 1);  // b_to_a
  events = 0;
  for (int i = 0; i < 3; ++i) events += kinds(step(z, n++, right), "line_cross");
  EXPECT_EQ(events, 1);  // a_to_b
  EXPECT_EQ(z.counts()["z1"]["a_to_b"].get<int>(), 2);
  EXPECT_EQ(z.counts()["z1"]["b_to_a"].get<int>(), 1);
}

TEST(Zones, LineCountFilter) {
  Zones z(cfg());
  z.set({line("z1", cv::Point2d(0, 0), cv::Point2d(100, 0), "a_to_b")}, 1);
  const std::vector<Obs> left{{1, cv::Point2d(50, 10)}};
  const std::vector<Obs> right{{1, cv::Point2d(50, -10)}};
  int n = 1;
  for (int i = 0; i < 3; ++i) EXPECT_EQ(kinds(step(z, n++, left), "line_cross"), 0);
  for (int i = 0; i < 2; ++i) EXPECT_EQ(kinds(step(z, n++, right), "line_cross"), 0);
  EXPECT_EQ(kinds(step(z, n++, right), "line_cross"), 1);  // a_to_b sayılır
  EXPECT_EQ(z.counts()["z1"]["a_to_b"].get<int>(), 1);
  // b_to_a count'ta yok: olay yok, sayaç null.
  for (int i = 0; i < 3; ++i) EXPECT_EQ(kinds(step(z, n++, left), "line_cross"), 0);
  EXPECT_EQ(z.counts()["z1"]["a_to_b"].get<int>(), 1);
  EXPECT_TRUE(z.counts()["z1"]["b_to_a"].is_null());
}

TEST(Zones, MergedGenRejected) {
  std::map<int, baykus::GenInfo> gens;
  gens[1] = baykus::GenInfo{false, 1, std::nullopt};
  gens[2] = baykus::GenInfo{false, 40, 1};  // gen 1'e birleşti
  gens[3] = baykus::GenInfo{true, 90, std::nullopt};

  // Birleşmiş nesil (gen 2) reddedilir; ileti "birleşti" içerir.
  try {
    baykus::check_zone_gens(baykus::parse_zone_defs(area_body(2)), gens);
    FAIL() << "birleşmiş nesil kabul edildi";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("birleşti"), std::string::npos) << e.what();
  }

  // Açık (gen 1) ve geçici (gen 3) nesiller kabul edilir.
  EXPECT_NO_THROW(baykus::check_zone_gens(baykus::parse_zone_defs(area_body(1)), gens));
  EXPECT_NO_THROW(baykus::check_zone_gens(baykus::parse_zone_defs(area_body(3)), gens));

  // Bilinmeyen nesil (gen 4) reddedilir.
  EXPECT_THROW(baykus::check_zone_gens(baykus::parse_zone_defs(area_body(4)), gens),
               std::invalid_argument);

  // Boş gens: yalnız gen 1 kabul edilir.
  const std::map<int, baykus::GenInfo> empty;
  EXPECT_NO_THROW(baykus::check_zone_gens(baykus::parse_zone_defs(area_body(1)), empty));
  EXPECT_THROW(baykus::check_zone_gens(baykus::parse_zone_defs(area_body(2)), empty),
               std::invalid_argument);
}

TEST(Zones, HugeIntsRejected) {
  // int aralığını aşan gen: ayrıştırma std::invalid_argument ile reddeder.
  nlohmann::json huge_gen = area_body(1);
  huge_gen["zones"][0]["gen"] = 4294967297LL;
  EXPECT_THROW(baykus::parse_zone_defs(huge_gen), std::invalid_argument);

  // int aralığını aşan dwell_frames: aynı şekilde reddedilir.
  nlohmann::json huge_dwell = area_body(1);
  huge_dwell["zones"][0]["kind"] = "dwell";
  huge_dwell["zones"][0]["dwell_frames"] = 4294967297LL;
  EXPECT_THROW(baykus::parse_zone_defs(huge_dwell), std::invalid_argument);
}

TEST(ZonesEvidence, Write) {
  const char* dir = std::getenv("BAYKUS_EVIDENCE_DIR");
  if (dir == nullptr || *dir == '\0') GTEST_SKIP() << "BAYKUS_EVIDENCE_DIR tanımsız";
  const std::string out_dir = dir;
  const std::string tmp =
      (std::filesystem::temp_directory_path() /
       ("zones_gtest_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json"))
          .string();
  const std::string cmd = std::filesystem::read_symlink("/proc/self/exe").string() +
                          " --gtest_filter=Zones.*:Polygon.*:EventLog.*:Summary.*:ClipZones.*:"
                          "MotionRuntimeTest.Zones*" +
                          " --gtest_output=json:" + tmp;
  const int rc = std::system(cmd.c_str());
  (void)rc;

  std::ifstream in(tmp);
  ASSERT_TRUE(in.good()) << "gtest JSON okunamadı: " << tmp;
  nlohmann::json j;
  in >> j;
  in.close();
  std::filesystem::remove(tmp);

  int total = 0;
  int passed = 0;
  int failed = 0;
  int skipped = 0;
  nlohmann::json tests = nlohmann::json::array();
  for (const auto& suite : j.at("testsuites")) {
    for (const auto& t : suite.at("testsuite")) {
      ++total;
      // gtest JSON: hata `failures` dizisinde; `result` COMPLETED | SKIPPED | SUPPRESSED.
      const bool has_failures = t.contains("failures") && !t.at("failures").empty();
      const std::string result = t.value("result", std::string());
      nlohmann::json e;
      e["name"] = suite.at("name").get<std::string>() + "." + t.at("name").get<std::string>();
      if (has_failures || result != "COMPLETED") {
        if (!has_failures && result == "SKIPPED") {
          ++skipped;
          e["result"] = "skipped";
        } else {
          ++failed;
          e["result"] = "failed";
        }
      } else {
        ++passed;
        e["result"] = "passed";
      }
      tests.push_back(e);
    }
  }

  nlohmann::json out;
  out["kabul"] = {"7", "11"};
  out["total"] = total;
  out["passed"] = passed;
  out["failed"] = failed;
  out["skipped"] = skipped;
  out["tests"] = tests;

  std::filesystem::create_directories(out_dir);
  const std::string path = out_dir + "/zones_tests.json";
  std::ofstream ofs(path);
  ASSERT_TRUE(ofs.good()) << "yazılamadı: " << path;
  ofs << out.dump(2) << "\n";
  ofs.close();

  EXPECT_GT(total, 0);
  EXPECT_EQ(failed, 0);
}
