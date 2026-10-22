// Kabul 2: hareket birim eşliği. Ego Python'dan sabit (`maps_val.json.gz`), izler C++ ByteTrack
// (`dev_dets`, Faz 3 takip eşliği birebir); beklenen `motion_val.json` (Python `MotionState`).
// Kimlikler tespit indeksi üzerinden eşlenir. Serbest satır: iki taraftan birinin v'si 0,03 ya da
// 0,10'a 1e-6'dan yakın; orada karar ayrışırsa iz kararlar yeniden eşitlenene kadar serbest.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "baykus/ego.hpp"
#include "baykus/files.hpp"
#include "baykus/motion.hpp"
#include "baykus/motion_config.hpp"
#include "baykus/tracker.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr double kSpeedTol = 1e-6;
constexpr double kXyTol = 1e-4;
constexpr double kBand = 1e-6;

fs::path configs() {
  const char* c = std::getenv("BAYKUS_CONFIGS");
  return fs::path(c && *c ? c : "/configs") / "engine";
}

json read_gz(const fs::path& p) { return json::parse(baykus::read_gzip(p.string())); }

std::optional<cv::Matx23d> mat6(const json& j) {
  if (j.is_null()) return std::nullopt;
  return cv::Matx23d(j[0].get<double>(), j[1].get<double>(), j[2].get<double>(), j[3].get<double>(),
                     j[4].get<double>(), j[5].get<double>());
}

baykus::EgoResult ego_from_json(const json& e) {
  baykus::EgoResult r;
  r.ok = e.at("ok").get<bool>();
  r.source = e.at("source").get<std::string>();
  r.map_gen = e.at("map_gen").get<int>();
  r.provisional = e.at("provisional").get<bool>();
  r.frame_to_map = mat6(e.at("frame_to_map"));
  if (!e.at("inliers").is_null()) r.inliers = e.at("inliers").get<int>();
  if (!e.at("inlier_ratio").is_null()) r.inlier_ratio = e.at("inlier_ratio").get<double>();
  r.keyframe = e.at("keyframe").get<bool>();
  r.relinked = e.at("relinked").get<bool>();
  r.relocalized = e.at("relocalized").get<bool>();
  if (!e.at("merge").is_null()) {
    const auto& m = e.at("merge");
    r.merge =
        baykus::Merge{m.at("from").get<int>(), m.at("to").get<int>(), *mat6(m.at("transform"))};
  }
  return r;
}

bool near_band(const std::optional<double>& v, double lo, double hi) {
  return v && (std::abs(*v - lo) < kBand || std::abs(*v - hi) < kBand);
}

}  // namespace

TEST(Motion, ValParity) {
  const auto root = bt::fixtures_root();
  const auto maps_p = root / "motion" / "maps_val.json.gz";
  const auto dets_p = root / "tracks" / "dev_dets.json.gz";
  const auto py_tracks_p = root / "tracks" / "dev_tracks_py.json.gz";
  const auto want_p = root / "expected" / "motion_val.json";
  BAYKUS_REQUIRE_FILES(
      {maps_p, dets_p, py_tracks_p, want_p, configs() / "motion.json", configs() / "tracker.json"});
  const auto mcfg = baykus::MotionCfg::load((configs() / "motion.json").string());
  const auto tcfg = baykus::TrackerConfig::load((configs() / "tracker.json").string());
  const json maps = read_gz(maps_p);
  const json dets = read_gz(dets_p);
  const json py_tracks = read_gz(py_tracks_p);
  const json want = bt::read_json(want_p);
  const auto& sm = maps.at("sequences");
  const auto& sd = dets.at("sequences");
  const auto& st = py_tracks.at("sequences");
  const auto& sw = want.at("sequences");
  ASSERT_EQ(sm.size(), sd.size());
  ASSERT_EQ(sm.size(), st.size());
  ASSERT_EQ(sm.size(), sw.size());

  const double lo = mcfg.motion.stationary_max;
  const double hi = mcfg.motion.moving_min;
  json evidence = json::object();
  int reported = 0;
  for (std::size_t q = 0; q < sm.size(); ++q) {
    const std::string seq = sm[q].at("seq").get<std::string>();
    SCOPED_TRACE(seq);
    ASSERT_EQ(sd[q].at("seq"), seq);
    ASSERT_EQ(st[q].at("seq"), seq);
    ASSERT_EQ(sw[q].at("seq"), seq);
    const auto& fm = sm[q].at("frames");
    const auto& fd = sd[q].at("frames");
    const auto& ft = st[q].at("frames");
    const auto& fw = sw[q].at("frames");
    ASSERT_EQ(fm.size(), fd.size());
    ASSERT_EQ(fm.size(), ft.size());
    ASSERT_EQ(fm.size(), fw.size());

    baykus::ByteTrack tracker(tcfg);
    baykus::MotionState ms(mcfg.motion);
    std::set<int> spreading;  // serbest satırdaki ayrışmanın taşındığı py izleri
    long rows = 0, compared = 0, freed = 0, mismatched = 0;
    double max_speed = 0.0, max_xy = 0.0;
    for (std::size_t k = 0; k < fm.size(); ++k) {
      const int frame = fm[k].at("frame").get<int>();
      ASSERT_EQ(fd[k].at("frame").get<int>(), frame);
      ASSERT_EQ(fw[k].at("frame").get<int>(), frame);
      std::vector<baykus::Det> in;
      std::map<int, baykus::Box> boxes;
      for (const auto& d : fd[k].at("dets")) {
        if (d.at("score_raw").get<double>() < 0.1) continue;  // takip katmanı
        const auto& b = d.at("box");
        baykus::Det det{
            d.at("index").get<int>(),    d.at("cls_id").get<int>(), d.at("score_raw").get<float>(),
            d.at("score").get<double>(), b[0].get<double>(),        b[1].get<double>(),
            b[2].get<double>(),          b[3].get<double>()};
        boxes[det.index] = baykus::Box{det.x1, det.y1, det.x2, det.y2};
        in.push_back(det);
      }
      std::map<int, int> cpp_of_index;
      std::vector<baykus::TrackObs> obs;
      for (const auto& [index, id] : tracker.update(in)) {
        if (id < 0) continue;
        cpp_of_index[index] = id;
        obs.push_back(baykus::TrackObs{id, boxes.at(index)});
      }
      const auto got = ms.update(ego_from_json(fm[k].at("ego")), obs);
      std::map<int, int> cpp_of_py;
      for (const auto& t : ft[k].at("tracks")) {
        const int py_id = t.at("track_id").get<int>();
        if (py_id < 0) continue;
        const auto it = cpp_of_index.find(t.at("index").get<int>());
        if (it != cpp_of_index.end()) cpp_of_py[py_id] = it->second;
      }

      for (const auto& row : fw[k].at("tracks")) {
        ++rows;
        const int py_id = row[0].get<int>();
        const std::string motion = row[1].get<std::string>();
        const std::optional<double> speed =
            row[2].is_null() ? std::nullopt : std::optional<double>(row[2].get<double>());
        const auto cid = cpp_of_py.find(py_id);
        const auto g = cid == cpp_of_py.end() ? got.end() : got.find(cid->second);
        if (g == got.end()) {
          ++mismatched;
          if (reported++ < 10)
            ADD_FAILURE() << "kare " << frame << " py " << py_id << ": C++ izi yok";
          continue;
        }
        const baykus::TrackMotion& m = g->second;
        const bool band = near_band(speed, lo, hi) || near_band(m.speed_rel, lo, hi);
        const bool same_motion = m.motion == motion;
        if (spreading.count(py_id)) {
          if (same_motion) {
            spreading.erase(py_id);
          } else {
            ++freed;
            continue;
          }
        }
        if (band && !same_motion) {
          spreading.insert(py_id);
          ++freed;
          continue;
        }
        if (band) ++freed;
        ++compared;
        bool ok = same_motion && speed.has_value() == m.speed_rel.has_value() &&
                  row[3].is_null() == !m.map_xy.has_value();
        if (ok && speed) {
          const double d = std::abs(*speed - *m.speed_rel);
          max_speed = std::max(max_speed, d);
          ok = d <= kSpeedTol;
        }
        if (ok && m.map_xy) {
          const double d = std::max(std::abs(row[3].get<double>() - m.map_xy->x),
                                    std::abs(row[4].get<double>() - m.map_xy->y));
          max_xy = std::max(max_xy, d);
          ok = d <= kXyTol;
        }
        if (!ok) {
          ++mismatched;
          if (reported++ < 10) {
            ADD_FAILURE() << "kare " << frame << " py " << py_id << ": beklenen " << row.dump()
                          << ", C++ " << m.motion
                          << " v=" << (m.speed_rel ? std::to_string(*m.speed_rel) : "null");
          }
        }
      }
    }
    EXPECT_EQ(mismatched, 0);
    ASSERT_GT(rows, 0);
    EXPECT_GE(static_cast<double>(compared) / static_cast<double>(rows), 0.99)
        << compared << "/" << rows;
    evidence[seq] = {{"rows", rows},
                     {"compared", compared},
                     {"free", freed},
                     {"mismatched", mismatched},
                     {"max_speed_diff", max_speed},
                     {"max_xy_diff", max_xy}};
  }
  if (const char* dir = std::getenv("BAYKUS_EVIDENCE_DIR"); dir && *dir) {
    std::ofstream(fs::path(dir) / "parity_motion.json") << evidence.dump(2) << "\n";
  }
}
