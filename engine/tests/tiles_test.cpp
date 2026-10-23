// F4 Görev 15D: parçalı çıkarım (`tiles=4`) Python referansıyla eşlik (fixtures/expected/tiles_*).
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "baykus/models.hpp"
#include "baykus/parity.hpp"
#include "baykus/pipeline.hpp"
#include "baykus/tiles.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
namespace bp = baykus::parity;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

fs::path cases_path() { return bt::fixtures_root() / "expected" / "tiles_cases.json"; }

fs::path manifest_path() {
  const char* c = std::getenv("BAYKUS_CONFIGS");
  return fs::path(c && *c ? c : "/configs") / "engine" / "models.json";
}

std::vector<std::vector<int>> rects(const std::vector<cv::Rect>& w) {
  std::vector<std::vector<int>> out;
  for (const auto& r : w) out.push_back({r.x, r.y, r.width, r.height});
  return out;
}

}  // namespace

TEST(Tiles, WindowsMatchPython) {
  BAYKUS_REQUIRE_FILES({cases_path()});
  for (const auto& c : bt::read_json(cases_path()).at("windows")) {
    const auto got = rects(baykus::tile_windows(c.at("w").get<int>(), c.at("h").get<int>()));
    EXPECT_EQ(json(got), c.at("windows")) << c.dump();
  }
}

TEST(Tiles, MergeMatchesPython) {
  BAYKUS_REQUIRE_FILES({cases_path()});
  const auto wins = baykus::tile_windows(640, 512);
  int n = 0;
  for (const auto& c : bt::read_json(cases_path()).at("merges")) {
    SCOPED_TRACE("durum " + std::to_string(n++));
    std::vector<std::vector<baykus::Det>> per_tile;
    for (const auto& t : c.at("per_tile")) per_tile.push_back(bt::dets_from_json(t));
    const auto got = baykus::merge_tiles(per_tile, wins);
    const auto want = bt::dets_from_json(c.at("merged"));
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      EXPECT_EQ(got[i].index, want[i].index);
      EXPECT_EQ(got[i].cls_id, want[i].cls_id);
      EXPECT_EQ(got[i].score_raw, want[i].score_raw);
      EXPECT_NEAR(got[i].x1, want[i].x1, 1e-9);
      EXPECT_NEAR(got[i].y1, want[i].y1, 1e-9);
      EXPECT_NEAR(got[i].x2, want[i].x2, 1e-9);
      EXPECT_NEAR(got[i].y2, want[i].y2, 1e-9);
    }
  }
}

TEST(Tiles, TinyFrame) {
  for (const auto& r : baykus::tile_windows(33, 17)) {
    EXPECT_GT(r.width, 0);
    EXPECT_GT(r.height, 0);
    EXPECT_GE(r.x, 0);
    EXPECT_GE(r.y, 0);
    EXPECT_LE(r.x + r.width, 33);
    EXPECT_LE(r.y + r.height, 17);
  }
}

TEST(Tiles, Tiles4FixtureParity) {
  const auto m = bt::models_root();
  BAYKUS_REQUIRE_FILES({bt::fixtures_root() / "index.json", m / "e3c.onnx", manifest_path()});
  const auto models = baykus::load_models(m.string(), manifest_path().string());
  baykus::Pipeline pipe(models, /*hash_tensor=*/true);
  const double display_boundary = models.calibration.display_boundary(0.25);
  const auto root = bt::fixtures_root();
  const auto entries = bt::fixtures_of("tiles");
  ASSERT_FALSE(entries.empty());
  for (const auto& e : entries) {
    SCOPED_TRACE(e.at("name").get<std::string>());
    const auto exp = bt::read_json(root / e.at("path").get<std::string>());
    const auto& in = exp.at("inputs");
    const auto r = pipe.run(bt::read_file(root / in.at("rgb").get<std::string>()),
                            bt::read_file(root / in.at("ir").get<std::string>()), "off", 0.001f,
                            nullptr, nullptr, 4);
    EXPECT_EQ(r.tensor_sha256, exp.at("tensor_sha256").get<std::string>());
    std::vector<baykus::Det> track;
    for (const auto& d : r.dets_all) {
      if (d.score_raw >= 0.1f) track.push_back(d);
    }
    const auto display = baykus::filter_display(track, 0.25);
    const struct {
      const char* name;
      const std::vector<baykus::Det>& got;
      double boundary;
    } layers[] = {
        {"eval", r.dets_all, 0.001}, {"track", track, 0.1}, {"display", display, display_boundary}};
    for (const auto& l : layers) {
      const auto c =
          bp::compare_layer(bt::dets_from_json(exp.at("dets").at(l.name)), l.got, l.boundary);
      EXPECT_TRUE(c.ok) << l.name << ": " << (c.errors.empty() ? "" : c.errors.front());
    }
  }
}
