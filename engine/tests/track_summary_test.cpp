// Kabul 11 araç özeti (docs/design/events_rules.md "TrackSummaries").
#include <gtest/gtest.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "baykus/ego.hpp"
#include "baykus/geometry.hpp"
#include "baykus/motion.hpp"
#include "baykus/track_summary.hpp"
#include "baykus/zones.hpp"

namespace {

using baykus::Box;
using baykus::EgoResult;
using baykus::SummaryCfg;
using baykus::TrackMotion;
using baykus::TrackOut;
using baykus::TrackSummaries;
using baykus::ZoneEvent;

EgoResult ego(int gen = 1, double s = 1.0) {
  EgoResult r;
  r.ok = true;
  r.source = "ir";
  r.map_gen = gen;
  r.provisional = false;
  r.frame_to_map = baykus::similarity(0, 0, 0, s);
  r.keyframe = false;
  r.relinked = false;
  r.relocalized = false;
  return r;
}

// Tek iz gözlemi: kare `frame`, sınıf, hareket, isteğe bağlı harita noktası (kutu 10 × 10).
struct Obs {
  int id;
  int cls = 0;
  std::string motion = "moving";
  std::optional<cv::Point2d> xy = std::nullopt;
};

void step(TrackSummaries& ts, int frame, const std::vector<Obs>& obs, const EgoResult& e = ego(),
          const std::vector<ZoneEvent>& events = {}) {
  std::vector<TrackOut> tracks;
  std::map<int, TrackMotion> motion;
  for (const auto& o : obs) {
    tracks.push_back(TrackOut{o.id, o.cls, Box{0, 0, 10, 10}, 0.9f, 0.9});
    motion[o.id] = TrackMotion{o.motion, std::nullopt, o.xy};
  }
  ts.update(frame, frame, tracks, motion, events, e);
}

ZoneEvent zone_event(const std::string& kind, int track_id, int frame) {
  ZoneEvent e;
  e.n = frame;
  e.frame = frame;
  e.kind = kind;
  e.zone_id = "z1";
  e.track_id = track_id;
  e.gen = 1;
  return e;
}

}  // namespace

TEST(Summary, MajorityClassTieSmallerId) {
  TrackSummaries ts(SummaryCfg{200, 500});
  int f = 1;
  for (const int cls : {2, 1, 2, 1}) step(ts, f++, {{1, cls}});
  EXPECT_EQ(ts.get(1)->at("cls_id"), 1);  // 2'ye 2 eşit → küçük
  EXPECT_EQ(ts.get(1)->at("class"), "truck");
  step(ts, f++, {{1, 2}});
  EXPECT_EQ(ts.get(1)->at("cls_id"), 2);
}

TEST(Summary, SegmentsAfterHysteresis) {
  TrackSummaries ts(SummaryCfg{200, 500});
  for (int f = 1; f <= 9; ++f) {
    const std::string m = f <= 3 ? "unknown" : (f <= 7 ? "moving" : "stationary");
    step(ts, f, {{1, 0, m}});
  }
  const auto j = *ts.get(1);
  const nlohmann::json want = {{{"state", "unknown"}, {"from_frame", 1}, {"to_frame", 3}},
                               {{"state", "moving"}, {"from_frame", 4}, {"to_frame", 7}},
                               {{"state", "stationary"}, {"from_frame", 8}, {"to_frame", 9}}};
  EXPECT_EQ(j.at("segments"), want);
  EXPECT_EQ(j.at("motion"), "stationary");
  EXPECT_EQ(j.at("motion_since_frame"), 8);
  EXPECT_EQ(j.at("stationary_frames"), 2);
  EXPECT_EQ(j.at("first_frame"), 1);
  EXPECT_EQ(j.at("last_frame"), 9);
}

TEST(Summary, SegmentCapMergesOldest) {
  TrackSummaries ts(SummaryCfg{3, 500});
  const std::vector<std::string> seq = {"moving",     "moving",     "stationary",
                                        "stationary", "stationary", "stationary",
                                        "stationary", "moving",     "stationary"};
  for (int f = 1; f <= 9; ++f) step(ts, f, {{1, 0, seq[f - 1]}});
  const auto j = *ts.get(1);
  const auto& segs = j.at("segments");
  ASSERT_EQ(segs.size(), 3u);
  // 1-2 moving + 3-7 stationary birleşti: süre uzun olanın durumu.
  EXPECT_EQ(segs[0], (nlohmann::json{{"state", "stationary"}, {"from_frame", 1}, {"to_frame", 7}}));
  EXPECT_EQ(segs[1].at("state"), "moving");
  EXPECT_EQ(segs[2].at("state"), "stationary");
}

TEST(Summary, PathRelInBoxUnits) {
  TrackSummaries ts(SummaryCfg{200, 500});
  // Kutu 10 × 10, ölçek 2 → harita boyu 20; her karede 20 px → kare başına 1 kutu boyu.
  for (int f = 1; f <= 4; ++f) {
    step(ts, f, {{1, 0, "moving", cv::Point2d(20.0 * f, 0.0)}}, ego(1, 2.0));
  }
  EXPECT_DOUBLE_EQ(ts.get(1)->at("path_rel").get<double>(), 3.0);
  // Başka nesil: adım eklenmez.
  step(ts, 5, {{1, 0, "moving", cv::Point2d(500.0, 0.0)}}, ego(2, 2.0));
  EXPECT_DOUBLE_EQ(ts.get(1)->at("path_rel").get<double>(), 3.0);
}

TEST(Summary, SizeSamplesBounded) {
  TrackSummaries ts(SummaryCfg{200, 500});
  // Kutu 10 × 10, ölçek 1 → her boy örneği 10; 10000 karede örnek sayısı sınırlı kalmalı.
  for (int f = 1; f <= 10000; ++f) {
    step(ts, f, {{1, 0, "moving", cv::Point2d(static_cast<double>(f), 0.0)}});
  }
  EXPECT_LE(ts.size_samples(1), 4096u);
  EXPECT_GT(ts.size_samples(1), 0u);
  EXPECT_NEAR(ts.get(1)->at("path_rel").get<double>(), 9999.0 / 10.0, 1e-9);
  EXPECT_EQ(ts.size_samples(99), 0u);
}

TEST(Summary, ZoneVisits) {
  TrackSummaries ts(SummaryCfg{200, 500});
  step(ts, 100, {{1}});
  step(ts, 106, {{1}}, ego(), {zone_event("enter", 1, 105)});
  step(ts, 111, {{1}}, ego(), {zone_event("exit", 1, 110)});
  step(ts, 121, {{1}}, ego(), {zone_event("enter", 1, 120), zone_event("enter", 99, 120)});
  const nlohmann::json want = {{{"zone_id", "z1"}, {"enter_frame", 105}, {"exit_frame", 110}},
                               {{"zone_id", "z1"}, {"enter_frame", 120}, {"exit_frame", nullptr}}};
  EXPECT_EQ(ts.get(1)->at("zones"), want);
  EXPECT_FALSE(ts.get(99).has_value());  // kaydı olmayan iz yok sayılır
}

TEST(Summary, EndedKeepCap) {
  TrackSummaries ts(SummaryCfg{200, 2});
  step(ts, 1, {{1}});
  step(ts, 2, {{2}});
  step(ts, 3, {{3}});
  for (int f = 4; f <= 6; ++f) step(ts, f, {{4}});
  EXPECT_FALSE(ts.get(1).has_value());  // bitmiş en eski
  ASSERT_TRUE(ts.get(2).has_value());
  EXPECT_FALSE(ts.get(2)->at("alive").get<bool>());
  ASSERT_TRUE(ts.get(3).has_value());
  EXPECT_FALSE(ts.get(3)->at("alive").get<bool>());
  EXPECT_TRUE(ts.get(4)->at("alive").get<bool>());
}
