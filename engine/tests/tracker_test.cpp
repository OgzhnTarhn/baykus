// Kabul 4: C++ ByteTrack ↔ Python referansı (`trackers==2.6.1`) M3OT geliştirme dizilerinde
// (`dev_dets.json.gz` girdisi, `dev_tracks_py.json.gz` beklenen). Tanım tasarım §6.1: tespitler
// girdi indeksiyle eşlenir, -1 yalnız -1 ile; dizi başına birebir kimlik eşlemesi; ilk farkta
// durulur.
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

#include "baykus/files.hpp"
#include "baykus/tracker.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

fs::path tracker_json() {
  const char* c = std::getenv("BAYKUS_CONFIGS");
  return fs::path(c && *c ? c : "/configs") / "engine" / "tracker.json";
}

fs::path temp_file(const std::string& name, const std::string& text) {
  const auto p = fs::temp_directory_path() / ("baykus_" + name);
  std::ofstream(p) << text;
  return p;
}

baykus::Det det(int index, float score, double x1, double y1, double x2, double y2) {
  return baykus::Det{index, 0, score, score, x1, y1, x2, y2};
}

json read_gz_json(const fs::path& p) { return json::parse(baykus::read_gzip(p.string())); }

// Python `_frame_ok`: anahtarlar aynı, -1 ↔ -1, kimlik eşlemesi iki yönde tutarlı.
bool frame_ok(const json& py, const std::vector<std::pair<int, int>>& cpp, std::map<int, int>& fwd,
              std::map<int, int>& back) {
  std::map<int, int> a, b;
  for (const auto& t : py) a[t.at("index").get<int>()] = t.at("track_id").get<int>();
  for (const auto& [i, id] : cpp) b[i] = id;
  if (a.size() != b.size()) return false;
  for (const auto& [i, ta] : a) {
    const auto it = b.find(i);
    if (it == b.end()) return false;
    const int tb = it->second;
    if ((ta == -1) != (tb == -1)) return false;
    if (ta == -1) continue;
    if (fwd.emplace(ta, tb).first->second != tb || back.emplace(tb, ta).first->second != ta) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST(Tracker, ConfigMatchesJson) {
  BAYKUS_REQUIRE_FILES({tracker_json()});
  const auto cfg = baykus::TrackerConfig::load(tracker_json().string());
  EXPECT_EQ(cfg.to_json(), bt::read_json(tracker_json()));
  EXPECT_EQ(cfg.lost_track_buffer, 30);
  EXPECT_EQ(cfg.frame_rate, 30.0);
  // F4 Görev 15A, val HOTA kuralı (m3ot_dev_sweep_e3c.json); Faz 3'te 0,7 / 0,6.
  EXPECT_EQ(cfg.track_activation_threshold, 0.3);
  EXPECT_EQ(cfg.minimum_consecutive_frames, 2);
  EXPECT_EQ(cfg.minimum_iou_threshold, 0.1);
  EXPECT_EQ(cfg.high_conf_det_threshold, 0.3);
}

TEST(Tracker, ConfigRejectsMissingOrExtraKeys) {
  json j = baykus::TrackerConfig{}.to_json();
  j.erase("frame_rate");
  EXPECT_THROW(baykus::TrackerConfig::load(temp_file("cfg_missing.json", j.dump()).string()),
               std::runtime_error);
  j = baykus::TrackerConfig{}.to_json();
  j["extra"] = 1;
  EXPECT_THROW(baykus::TrackerConfig::load(temp_file("cfg_extra.json", j.dump()).string()),
               std::runtime_error);
}

TEST(Tracker, MaximumFramesWithoutUpdate) {  // trackers 2.6.1 belge örnekleri
  auto f = [](int buffer, double rate) {
    baykus::TrackerConfig c;
    c.lost_track_buffer = buffer;
    c.frame_rate = rate;
    return c.maximum_frames_without_update();
  };
  EXPECT_EQ(f(30, 30.0), 30);
  EXPECT_EQ(f(30, 60.0), 60);
  EXPECT_EQ(f(3, 15.0), 2);
  EXPECT_EQ(f(0, 30.0), 0);
}

TEST(Tracker, IdAfterSecondConsecutiveUpdateAndActivationThreshold) {
  baykus::ByteTrack tr(baykus::TrackerConfig{});
  // 0,65: yüksek skorlu (≥ 0,6) ama iz açmaz (< 0,7)
  EXPECT_EQ(tr.update({det(4, 0.65f, 10, 10, 50, 50)}),
            (std::vector<std::pair<int, int>>{{4, -1}}));
  EXPECT_EQ(tr.update({det(4, 0.65f, 10, 10, 50, 50)}),
            (std::vector<std::pair<int, int>>{{4, -1}}));
  // 0,75: iz açılır, ikinci karede kimlik 0
  EXPECT_EQ(tr.update({det(7, 0.75f, 10, 10, 50, 50)}),
            (std::vector<std::pair<int, int>>{{7, -1}}));
  EXPECT_EQ(tr.update({det(2, 0.75f, 11, 10, 51, 50)}), (std::vector<std::pair<int, int>>{{2, 0}}));
  tr.reset();
  EXPECT_EQ(tr.update({det(1, 0.75f, 10, 10, 50, 50)}),
            (std::vector<std::pair<int, int>>{{1, -1}}));
  EXPECT_EQ(tr.update({det(1, 0.75f, 10, 10, 50, 50)}), (std::vector<std::pair<int, int>>{{1, 0}}));
}

TEST(Tracker, DumpHasPythonFields) {
  baykus::ByteTrack tr(baykus::TrackerConfig{});
  json d0, d1;
  tr.update({det(0, 0.9f, 10, 10, 50, 50)}, &d0);
  tr.update({det(0, 0.9f, 10, 10, 50, 50), det(1, 0.3f, 100, 100, 120, 120)}, &d1);
  for (const char* k :
       {"tracks_before", "tracks_after", "dets", "iou1", "assign1", "iou2", "assign2"}) {
    EXPECT_TRUE(d1.contains(k)) << k;
  }
  EXPECT_TRUE(d0.at("tracks_before").empty());
  EXPECT_EQ(d1.at("tracks_before").size(), 1u);
  EXPECT_EQ(d1.at("assign1"), json::parse("[[0, 0]]"));
  EXPECT_EQ(d1.at("iou2").size(), 0u);  // eşleşmeyen iz yok: 0 × 1
  EXPECT_EQ(d1.at("dets").at(1).at("index"), 1);
}

TEST(Tracker, FixtureParity) {
  const auto root = bt::fixtures_root();
  BAYKUS_REQUIRE_FILES({root / "tracks" / "dev_dets.json.gz",
                        root / "tracks" / "dev_tracks_py.json.gz", tracker_json()});
  const auto cfg = baykus::TrackerConfig::load(tracker_json().string());
  const auto dets = read_gz_json(root / "tracks" / "dev_dets.json.gz");
  const auto want = read_gz_json(root / "tracks" / "dev_tracks_py.json.gz");
  ASSERT_EQ(want.at("cfg"), cfg.to_json());
  std::map<std::string, json> py;
  for (const auto& s : want.at("sequences")) py[s.at("seq").get<std::string>()] = s;
  ASSERT_EQ(dets.at("sequences").size(), py.size());
  baykus::ByteTrack tracker(cfg);
  for (const auto& s : dets.at("sequences")) {
    const auto seq = s.at("seq").get<std::string>();
    SCOPED_TRACE(seq);
    ASSERT_TRUE(py.count(seq));
    const auto& py_frames = py[seq].at("frames");
    ASSERT_EQ(py_frames.size(), s.at("frames").size());
    tracker.reset();
    std::map<int, int> fwd, back;
    std::size_t compared = 0;
    int first_diff = -1;
    for (std::size_t k = 0; k < py_frames.size(); ++k) {
      const auto& fr = s.at("frames")[k];
      ASSERT_EQ(fr.at("frame"), py_frames[k].at("frame"));
      std::vector<baykus::Det> in;
      for (const auto& d : fr.at("dets")) {
        if (d.at("score_raw").get<double>() < 0.1) continue;  // takip katmanı
        const auto& b = d.at("box");
        in.push_back(det(d.at("index").get<int>(), d.at("score_raw").get<float>(),
                         b[0].get<double>(), b[1].get<double>(), b[2].get<double>(),
                         b[3].get<double>()));
      }
      if (!frame_ok(py_frames[k].at("tracks"), tracker.update(in), fwd, back)) {
        first_diff = fr.at("frame").get<int>();
        break;
      }
      ++compared;
    }
    EXPECT_EQ(first_diff, -1) << "ilk farklı kare; karşılaştırılan " << compared << "/"
                              << py_frames.size();
  }
}

TEST(Tracker, TrackFileMatchesPythonRun) {  // baykus_track'in yolu (iz dosyası biçimi + Kabul 4)
  const auto root = bt::fixtures_root();
  BAYKUS_REQUIRE_FILES({root / "tracks" / "dev_dets.json.gz",
                        root / "tracks" / "dev_tracks_py.json.gz", tracker_json()});
  const auto cfg = baykus::TrackerConfig::load(tracker_json().string());
  const auto dets = read_gz_json(root / "tracks" / "dev_dets.json.gz");
  const auto want = read_gz_json(root / "tracks" / "dev_tracks_py.json.gz");
  const json out = baykus::track_file(dets, cfg, {37}, false);
  EXPECT_EQ(out.at("cfg"), cfg.to_json());
  ASSERT_EQ(out.at("sequences").size(), want.at("sequences").size());
  for (std::size_t i = 0; i < want.at("sequences").size(); ++i) {
    const auto& py = want.at("sequences")[i];
    const auto& cpp = out.at("sequences")[i];
    SCOPED_TRACE(py.at("seq").get<std::string>());
    EXPECT_EQ(cpp.at("seq"), py.at("seq"));
    EXPECT_EQ(cpp.at("split"), py.at("split"));
    ASSERT_EQ(cpp.at("frames").size(), py.at("frames").size());
    std::map<int, int> fwd, back;
    for (std::size_t k = 0; k < py.at("frames").size(); ++k) {
      const auto& fc = cpp.at("frames")[k];
      ASSERT_EQ(fc.at("frame"), py.at("frames")[k].at("frame"));
      std::vector<std::pair<int, int>> got;
      for (const auto& t : fc.at("tracks")) {
        got.emplace_back(t.at("index").get<int>(), t.at("track_id").get<int>());
      }
      ASSERT_TRUE(frame_ok(py.at("frames")[k].at("tracks"), got, fwd, back))
          << "kare " << fc.at("frame");
    }
    const auto& dump = cpp.at("dumps").at("37");
    for (const char* k :
         {"tracks_before", "tracks_after", "dets", "iou1", "assign1", "iou2", "assign2"}) {
      EXPECT_TRUE(dump.contains(k)) << k;
    }
  }
}

TEST(Tracker, TrackFileRefusesTestSplit) {
  const json rec = {{"seq", "2-01"}, {"split", "test"}, {"frames", json::array()}};
  EXPECT_THROW(baykus::track_file(rec, baykus::TrackerConfig{}, {}, false), std::invalid_argument);
  EXPECT_EQ(baykus::track_file(rec, baykus::TrackerConfig{}, {}, true).at("sequences").size(), 1u);
}

TEST(Tracker, TrackFileSingleSequenceAppliesTrackFloor) {  // `baykus-engine detect` çıktısı
  const json d_hi = {{"index", 3},
                     {"cls_id", 0},
                     {"score_raw", 0.9},
                     {"score", 0.9},
                     {"box", {10.0, 10.0, 50.0, 50.0}}};
  const json d_lo = {{"index", 5},
                     {"cls_id", 1},
                     {"score_raw", 0.05},
                     {"score", 0.05},
                     {"box", {100.0, 100.0, 120.0, 120.0}}};
  const json rec = {
      {"seq", "1_08"},
      {"split", "val"},
      {"floor", 0.001},
      {"frames", {{{"frame", 1}, {"dets", {d_hi, d_lo}}}, {{"frame", 2}, {"dets", {d_hi}}}}}};
  const json out = baykus::track_file(rec, baykus::TrackerConfig{}, {}, false);
  const auto& frames = out.at("sequences").at(0).at("frames");
  EXPECT_EQ(frames.at(0).at("tracks"), json::parse(R"([{"index": 3, "track_id": -1}])"));
  EXPECT_EQ(frames.at(1).at("tracks"), json::parse(R"([{"index": 3, "track_id": 0}])"));
  EXPECT_TRUE(out.at("sequences").at(0).at("dumps").empty());
}

TEST(Files, GzipRoundTrip) {
  const auto p = (fs::temp_directory_path() / "baykus_roundtrip.json.gz").string();
  const std::string text = "{\"a\": [1, 2, 3], \"ş\": \"ğüİ\"}";
  baykus::write_gzip(p, text);
  EXPECT_EQ(baykus::read_gzip(p), text);
  EXPECT_THROW(baykus::read_gzip(p + ".yok"), std::runtime_error);
}
