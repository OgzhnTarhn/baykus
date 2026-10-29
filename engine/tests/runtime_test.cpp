// Ortak yürütme yolu: oturumlu takip, oturum bağımsızlığı, conf'un takibi etkilememesi, süreler,
// hatalı istekler ve `baykus_detect` biçimi. Girdi: video klibi (15 ardışık M3OT karesi).
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "baykus/models.hpp"
#include "baykus/motion_config.hpp"
#include "baykus/pipeline.hpp"
#include "baykus/runtime.hpp"
#include "baykus/sessions.hpp"
#include "baykus/tracker.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

fs::path configs() {
  const char* c = std::getenv("BAYKUS_CONFIGS");
  return fs::path(c && *c ? c : "/configs") / "engine";
}

fs::path clip_dir() { return bt::fixtures_root() / "m3ot" / "clip" / "sequences" / "clip"; }

struct Frame {
  std::string rgb;
  std::string ir;
};

std::vector<Frame> clip_frames() {
  std::vector<Frame> out;
  for (int f = 1; f <= 15; ++f) {
    char name[16];
    std::snprintf(name, sizeof(name), "%06d", f);
    out.push_back({bt::read_file(clip_dir() / "rgb" / (std::string(name) + ".jpg")),
                   bt::read_file(clip_dir() / "ir" / (std::string(name) + ".png"))});
  }
  return out;
}

baykus::Request request(const Frame& f, const std::string& session, double conf = 0.25) {
  baykus::Request r;
  r.rgb = f.rgb;
  r.ir = f.ir;
  r.session = session;
  r.conf = conf;
  return r;
}

std::map<int, int> ids_by_index(const baykus::Response& r) {
  std::map<int, int> m;
  for (std::size_t i = 0; i < r.dets.size(); ++i) m[r.dets[i].index] = r.track_ids[i];
  return m;
}

}  // namespace

class RuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto m = bt::models_root();
    BAYKUS_REQUIRE_FILES({clip_dir() / "index.json", m / "e3c.onnx", m / "thresholds.json",
                          m / "calibration.json", m / "gate_model.json", configs() / "models.json",
                          configs() / "tracker.json"});
    models_ = std::make_unique<baykus::Models>(
        baykus::load_models(m.string(), (configs() / "models.json").string()));
    cfg_ = baykus::TrackerConfig::load((configs() / "tracker.json").string());
    sessions_ = std::make_unique<baykus::Sessions>(cfg_, models_->gate.window,
                                                   models_->gate.min_frames, models_->gate.tol_px);
    runtime_ = std::make_unique<baykus::Runtime>(*models_, cfg_, *sessions_);
    frames_ = clip_frames();
  }

  std::unique_ptr<baykus::Models> models_;
  baykus::TrackerConfig cfg_;
  std::unique_ptr<baykus::Sessions> sessions_;
  std::unique_ptr<baykus::Runtime> runtime_;
  std::vector<Frame> frames_;
};

TEST_F(RuntimeTest, TrackIdsPersistWithinSession) {
  // Referans: aynı modüller elle (pipeline + takipçi + pencere) aynı sırayla
  baykus::Pipeline pipe(*models_);
  baykus::ByteTrack tracker(cfg_);
  baykus::MisalignWindow window(models_->gate.window, models_->gate.min_frames,
                                models_->gate.tol_px);
  std::map<int, int> frames_per_id;
  for (std::size_t k = 0; k < frames_.size(); ++k) {
    SCOPED_TRACE("kare " + std::to_string(k + 1));
    const auto got = runtime_->process(request(frames_[k], "s"));
    EXPECT_EQ(got.session_new, k == 0);
    const auto ref = pipe.run(frames_[k].rgb, frames_[k].ir, "advise", 0.1f, &window);
    std::map<int, int> want;
    for (const auto& [index, id] : tracker.update(ref.dets_all)) want[index] = id;
    const auto display = baykus::filter_display(ref.dets_all, 0.25);
    ASSERT_EQ(got.dets.size(), display.size());
    for (std::size_t i = 0; i < display.size(); ++i) {
      EXPECT_EQ(got.dets[i].index, display[i].index);
      EXPECT_EQ(got.track_ids[i], want.count(display[i].index) ? want[display[i].index] : -1);
      if (got.track_ids[i] >= 0) ++frames_per_id[got.track_ids[i]];
    }
    EXPECT_EQ(got.health, ref.health);
  }
  int longest = 0;
  for (const auto& [id, n] : frames_per_id) longest = std::max(longest, n);
  EXPECT_GE(longest, 10) << "aynı araç kimliğini korumalı";
}

TEST_F(RuntimeTest, SessionsAreIndependent) {
  std::vector<std::map<int, int>> alone;
  for (const auto& f : frames_) alone.push_back(ids_by_index(runtime_->process(request(f, "x"))));
  for (std::size_t k = 0; k < frames_.size(); ++k) {  // a ve b iç içe
    const auto a = runtime_->process(request(frames_[k], "a"));
    const auto b = runtime_->process(request(frames_[k], "b"));
    EXPECT_EQ(a.session_new, k == 0);
    EXPECT_EQ(b.session_new, k == 0);
    EXPECT_EQ(ids_by_index(a), alone[k]) << "kare " << k + 1;
    EXPECT_EQ(ids_by_index(b), alone[k]) << "kare " << k + 1;
  }
}

TEST_F(RuntimeTest, ConfDoesNotChangeTracks) {
  int common = 0;
  for (const auto& f : frames_) {
    const auto low = ids_by_index(runtime_->process(request(f, "low", 0.25)));
    const auto high = ids_by_index(runtime_->process(request(f, "high", 0.6)));
    EXPECT_LE(high.size(), low.size());
    for (const auto& [index, id] : high) {
      ASSERT_TRUE(low.count(index));
      EXPECT_EQ(low.at(index), id);
      ++common;
    }
  }
  EXPECT_GT(common, 0);
}

TEST_F(RuntimeTest, TotalIncludesTrack) {
  const auto r = runtime_->process(request(frames_[0], "t"));
  double sum = 0.0;
  for (const char* k : {"decode", "health", "pre", "infer", "post", "track"}) {
    ASSERT_TRUE(r.timing_ms.count(k)) << k;
    sum += r.timing_ms.at(k);
  }
  EXPECT_DOUBLE_EQ(r.timing_ms.at("total"), sum);
  EXPECT_GE(r.timing_ms.at("total"), r.timing_ms.at("track"));
}

TEST_F(RuntimeTest, NoSessionMeansNoTemporalRuleAndNoTracking) {
  auto req = request(frames_[0], "");
  req.track = false;
  const auto r = runtime_->process(req);
  EXPECT_FALSE(r.session_new);
  for (int id : r.track_ids) EXPECT_EQ(id, -1);
  EXPECT_EQ(r.health.at("pair").at("misaligned"), r.health.at("pair").at("misaligned_frame"));
  EXPECT_EQ(r.timing_ms.at("track"), 0.0);
  EXPECT_EQ(sessions_->size(), 0u);
}

TEST_F(RuntimeTest, InvalidRequestsThrow) {
  EXPECT_THROW(runtime_->process(request(frames_[0], "")), std::invalid_argument);  // takip var
  EXPECT_THROW(runtime_->process(request(frames_[0], "s", 1.5)), std::invalid_argument);
  EXPECT_THROW(runtime_->process(request(frames_[0], "s", -0.1)), std::invalid_argument);
  auto bad_gate = request(frames_[0], "s");
  bad_gate.gate = "bogus";
  EXPECT_THROW(runtime_->process(bad_gate), std::invalid_argument);
  auto bad_image = request(frames_[0], "s");
  bad_image.rgb = "not an image";
  EXPECT_THROW(runtime_->process(bad_image), std::invalid_argument);
}

TEST_F(RuntimeTest, DetectMatchesPythonFormat) {
  const json rec = baykus::detect_sequence(*runtime_, clip_dir().string(), 0.001, "advise", false);
  EXPECT_EQ(rec.at("seq"), "clip");
  EXPECT_EQ(rec.at("split"), "val");
  EXPECT_EQ(rec.at("floor").get<double>(), 0.001);  // verilen değer (Python gibi)
  EXPECT_EQ(rec.at("gate"), "advise");
  ASSERT_EQ(rec.at("frames").size(), 15u);
  baykus::Pipeline pipe(*models_);
  const auto ref = pipe.run(frames_[0].rgb, frames_[0].ir, "advise", 0.001f, nullptr);
  const auto& f1 = rec.at("frames").at(0);
  EXPECT_EQ(f1.at("frame"), 1);
  ASSERT_EQ(f1.at("dets").size(), ref.dets_all.size());
  for (std::size_t i = 0; i < ref.dets_all.size(); ++i) {
    const auto& d = f1.at("dets")[i];
    ASSERT_EQ(d.size(), 5u);  // index, cls_id, score_raw, score, box (Python asdict(Det))
    EXPECT_EQ(d.at("index"), ref.dets_all[i].index);
    EXPECT_EQ(d.at("cls_id"), ref.dets_all[i].cls_id);
    EXPECT_EQ(d.at("score_raw").get<double>(), static_cast<double>(ref.dets_all[i].score_raw));
    EXPECT_EQ(d.at("score").get<double>(), ref.dets_all[i].score);
    ASSERT_EQ(d.at("box").size(), 4u);
    EXPECT_EQ(d.at("box")[0].get<double>(), ref.dets_all[i].x1);
  }
  const auto round = bt::dets_from_json(f1.at("dets"));  // test desteğinin Python okuyucusu
  EXPECT_EQ(round.size(), ref.dets_all.size());
  EXPECT_FALSE(rec.contains("tiles"));  // tiles=1 çıktısı Python biçimini değiştirmez
}

TEST_F(RuntimeTest, DetectTiles4) {
  const json rec =
      baykus::detect_sequence(*runtime_, clip_dir().string(), 0.001, "advise", false, 4);
  EXPECT_EQ(rec.at("tiles"), 4);
  ASSERT_EQ(rec.at("frames").size(), 15u);
  baykus::Pipeline pipe(*models_);
  const auto ref = pipe.run(frames_[0].rgb, frames_[0].ir, "advise", 0.001f, nullptr, nullptr, 4);
  const auto& f1 = rec.at("frames").at(0);
  ASSERT_EQ(f1.at("dets").size(), ref.dets_all.size());
  for (std::size_t i = 0; i < ref.dets_all.size(); ++i) {
    const auto& d = f1.at("dets")[i];
    EXPECT_EQ(d.at("index"), ref.dets_all[i].index);
    EXPECT_EQ(d.at("cls_id"), ref.dets_all[i].cls_id);
    EXPECT_EQ(d.at("score_raw").get<double>(), static_cast<double>(ref.dets_all[i].score_raw));
    EXPECT_EQ(d.at("box")[0].get<double>(), ref.dets_all[i].x1);
  }
}

TEST_F(RuntimeTest, DetectRejectsBadTiles) {
  EXPECT_THROW(baykus::detect_sequence(*runtime_, clip_dir().string(), 0.001, "advise", false, 2),
               std::invalid_argument);
}

TEST_F(RuntimeTest, DetectRefusesTestSplit) {
  const auto d = fs::temp_directory_path() / "baykus_test_seq";
  fs::create_directories(d);
  std::ofstream(d / "index.json") << R"({"name": "2-01", "split": "test", "frames": [1]})";
  EXPECT_THROW(baykus::detect_sequence(*runtime_, d.string(), 0.001, "advise", false),
               std::invalid_argument);
}

// Faz 4: oturum başına ego, hareket ve kapsamlı sıfırlama (Python `SessionRef` sırası).
namespace {

fs::path f4_clip() { return bt::fixtures_root() / "f4" / "clip"; }

Frame f4_frame(int f) {
  char name[16];
  std::snprintf(name, sizeof(name), "%06d", f);
  return {bt::read_file(f4_clip() / "rgb" / (std::string(name) + ".jpg")),
          bt::read_file(f4_clip() / "ir" / (std::string(name) + ".png"))};
}

std::vector<std::pair<int, std::vector<double>>> track_rows(const baykus::Response& r) {
  std::vector<std::pair<int, std::vector<double>>> out;
  for (const auto& t : r.tracks)
    out.push_back({t.track_id, {t.box.x1, t.box.y1, t.box.x2, t.box.y2}});
  return out;
}

}  // namespace

class MotionRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto m = bt::models_root();
    BAYKUS_REQUIRE_FILES({f4_clip() / "index.json", m / "e3c.onnx", configs() / "models.json",
                          configs() / "tracker.json", configs() / "motion.json"});
    models_ = std::make_unique<baykus::Models>(
        baykus::load_models(m.string(), (configs() / "models.json").string()));
    cfg_ = baykus::TrackerConfig::load((configs() / "tracker.json").string());
    const auto motion = baykus::MotionCfg::load((configs() / "motion.json").string());
    sessions_ = std::make_unique<baykus::Sessions>(cfg_, motion, models_->gate.window,
                                                   models_->gate.min_frames, models_->gate.tol_px);
    runtime_ = std::make_unique<baykus::Runtime>(*models_, cfg_, *sessions_);
  }

  baykus::Response step(int f, const std::string& session, double conf = 0.25) {
    auto r = request(f4_frame(f), session, conf);
    r.frame = f;
    return runtime_->process(r);
  }

  std::unique_ptr<baykus::Models> models_;
  baykus::TrackerConfig cfg_;
  std::unique_ptr<baykus::Sessions> sessions_;
  std::unique_ptr<baykus::Runtime> runtime_;
};

TEST_F(MotionRuntimeTest, TracksIndependentOfConf) {  // Review Focus 3
  for (int f = 1; f <= 8; ++f) {
    SCOPED_TRACE("kare " + std::to_string(f));
    const auto a = step(f, "low", 0.25);
    const auto b = step(f, "high", 1.0);
    EXPECT_EQ(track_rows(a), track_rows(b));
    ASSERT_TRUE(a.ego && b.ego);
    EXPECT_EQ(a.ego->to_json(), b.ego->to_json());
    EXPECT_TRUE(b.dets.empty());
  }
}

TEST_F(MotionRuntimeTest, BadRequestLeavesEgoState) {  // Review Focus 2
  step(1, "a");
  step(2, "a");
  auto bad = request(f4_frame(3), "a");
  bad.rgb = "bozuk";
  EXPECT_THROW(runtime_->process(bad), std::invalid_argument);
  const auto a3 = step(3, "a");
  step(1, "b");
  step(2, "b");
  const auto b3 = step(3, "b");
  ASSERT_TRUE(a3.ego && b3.ego);
  EXPECT_EQ(a3.ego->to_json(), b3.ego->to_json());
  const auto before = sessions_->size();
  auto first_bad = request(f4_frame(1), "c");
  first_bad.ir = "bozuk";
  EXPECT_THROW(runtime_->process(first_bad), std::invalid_argument);
  EXPECT_EQ(sessions_->size(), before);  // bozuk ilk istek oturum açmaz (S1)
}

TEST_F(MotionRuntimeTest, NoSessionNoEgo) {
  auto r = request(f4_frame(1), "");
  r.track = false;
  const auto out = runtime_->process(r);
  EXPECT_FALSE(out.ego.has_value());
  EXPECT_TRUE(out.tracks.empty());
  const auto j = baykus::response_json(out);
  EXPECT_TRUE(j.at("ego").is_null());
  EXPECT_TRUE(j.at("health").at("ir").at("frozen").is_null());
  EXPECT_TRUE(j.at("health").at("rgb").at("diff_mad").is_null());
}

TEST_F(MotionRuntimeTest, TrackOffSkipsMotion) {
  // Tasarım §5.7: ego, hareket, bölgeler ve olay günlüğü yalnız track=true + oturumla çalışır.
  for (int f = 1; f <= 4; ++f) {
    SCOPED_TRACE("kare " + std::to_string(f));
    auto r = request(f4_frame(f), "off");
    r.frame = f;
    r.track = false;
    const auto out = runtime_->process(r);
    EXPECT_FALSE(out.ego.has_value());
    EXPECT_TRUE(out.tracks.empty());
    EXPECT_TRUE(out.log_events.is_array());
    EXPECT_TRUE(out.log_events.empty());
    EXPECT_TRUE(out.alarms.empty());
    EXPECT_TRUE(baykus::response_json(out).at("ego").is_null());
  }
  const auto s = sessions_->find("off");
  ASSERT_TRUE(s);
  ASSERT_TRUE(s->motion);
  EXPECT_EQ(s->motion->log.last_seq(), 0);
  const auto on = step(5, "off");
  ASSERT_TRUE(on.ego.has_value());
}

TEST_F(MotionRuntimeTest, ScopeTracksKeepsMapRaisesIdBase) {
  int max_before = -1;
  for (int f = 1; f <= 12; ++f) {
    for (const auto& t : step(f, "s").tracks) max_before = std::max(max_before, t.track_id);
  }
  ASSERT_GE(max_before, 0);
  sessions_->reset("s", baykus::Scope::Tracks);
  bool back_to_gen1 = false;
  for (int f = 13; f <= 24; ++f) {
    const auto r = step(f, "s");
    for (const auto& t : r.tracks) EXPECT_GT(t.track_id, max_before);
    if (r.ego && r.ego->ok && r.ego->map_gen == 1 && !r.ego->provisional) back_to_gen1 = true;
  }
  EXPECT_TRUE(back_to_gen1);  // kütüphane korundu: yeniden konumlama nesil 1'e birleştirir
}

TEST_F(MotionRuntimeTest, ScopeAllClears) {
  for (int f = 1; f <= 12; ++f) step(f, "s");
  sessions_->reset("s", baykus::Scope::All);
  const auto first = step(13, "s");
  ASSERT_TRUE(first.ego);
  EXPECT_TRUE(first.ego->keyframe);
  EXPECT_EQ(first.ego->map_gen, 1);
  int min_id = 1 << 30;
  for (int f = 14; f <= 20; ++f) {
    for (const auto& t : step(f, "s").tracks) min_id = std::min(min_id, t.track_id);
  }
  EXPECT_EQ(min_id, 0);  // kimlikler baştan
}

TEST_F(MotionRuntimeTest, ZonesConfDoesNotChangeEvents) {  // Review Focus 3
  std::vector<baykus::ZoneDef> defs;
  baykus::ZoneDef r;
  r.id = "r1";
  r.name = "r1";
  r.kind = "restricted";
  r.gen = 1;
  r.points = {cv::Point2d(-10000.0, -10000.0), cv::Point2d(10000.0, -10000.0),
              cv::Point2d(10000.0, 10000.0), cv::Point2d(-10000.0, 10000.0)};
  defs.push_back(r);
  baykus::ZoneDef h;
  h.id = "h1";
  h.name = "h1";
  h.kind = "line";
  h.gen = 1;
  h.points = {cv::Point2d(0.0, 256.0), cv::Point2d(640.0, 256.0)};
  h.count = "both";
  defs.push_back(h);
  for (const std::string id : {"low", "high"}) {
    bool created = false;
    const auto s = sessions_->get(id, &created);
    ASSERT_TRUE(s->motion);
    s->motion->zones.set(defs, 1);
  }
  bool saw_enter = false;
  baykus::Response last;
  for (int f = 1; f <= 40; ++f) {
    SCOPED_TRACE("kare " + std::to_string(f));
    const auto a = step(f, "low", 0.25);
    const auto b = step(f, "high", 1.0);
    json ea = json::array();
    json eb = json::array();
    for (const auto& e : a.events) ea.push_back(e.to_json());
    for (const auto& e : b.events) eb.push_back(e.to_json());
    EXPECT_EQ(ea, eb);
    json aa = json::array();
    json ab = json::array();
    for (const auto& x : a.alarms) aa.push_back(x.to_json());
    for (const auto& x : b.alarms) ab.push_back(x.to_json());
    EXPECT_EQ(aa, ab);
    EXPECT_EQ(a.zone_counts, b.zone_counts);
    for (const auto& e : a.events) {
      if (e.kind == "enter") saw_enter = true;
    }
    last = b;
  }
  EXPECT_TRUE(saw_enter);
  const auto j = baykus::response_json(last);
  EXPECT_TRUE(j.contains("events"));
  EXPECT_TRUE(j.contains("alarms"));
  EXPECT_TRUE(j.contains("zone_counts"));
}
