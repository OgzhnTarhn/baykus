// Kabul 1 a ve b (modül düzeyi): `modules` bloğu (kapı off, özgün pikseller). Ham referans
// BAYKUS_RAW_REF ile aynı makinede üretilmiş dosyadan alınabilir (CI; fixtures.hpp).
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "baykus/calibration.hpp"
#include "baykus/detections.hpp"
#include "baykus/detector.hpp"
#include "baykus/image_io.hpp"
#include "baykus/parity.hpp"
#include "baykus/preprocess.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
namespace bp = baykus::parity;

namespace {

std::vector<bp::Row> raw_rows(const std::vector<float>& out) {
  std::vector<bp::Row> rows(baykus::kRows);
  for (int i = 0; i < baykus::kRows; ++i) {
    for (int j = 0; j < baykus::kCols; ++j) {
      rows[i][j] = out[i * baykus::kCols + j];
    }
  }
  return rows;
}

std::vector<bp::Row> raw_rows(const bt::json& j) {
  std::vector<bp::Row> rows;
  for (const auto& r : j) {
    bp::Row row{};
    for (int k = 0; k < baykus::kCols; ++k) {
      row[k] = static_cast<float>(r.at(k).get<double>());
    }
    rows.push_back(row);
  }
  return rows;
}

struct Frame {
  std::string name;
  bt::json expected;
  std::vector<float> out;
  baykus::Letterbox lb;
};

std::vector<Frame> run_frames(baykus::Detector& det) {
  const auto root = bt::fixtures_root();
  std::vector<Frame> frames;
  for (const auto& e : bt::fixtures_of("frame")) {
    Frame f{e.at("name").get<std::string>(),
            bt::read_json(root / e.at("path").get<std::string>()),
            {},
            {}};
    const auto& in = f.expected.at("inputs");
    auto p = baykus::decode_pair(bt::read_file(root / in.at("rgb").get<std::string>()),
                                 bt::read_file(root / in.at("ir").get<std::string>()));
    f.out = det.infer(baykus::letterbox(baykus::stack_rgbt(p.bgr, p.ir), det.imgsz(), &f.lb));
    frames.push_back(std::move(f));
  }
  return frames;
}

}  // namespace

class DetectorParity : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto m = bt::models_root();
    BAYKUS_REQUIRE_FILES(
        {bt::fixtures_root() / "index.json", m / "e3c.onnx", m / "calibration.json"});
  }
};

TEST_F(DetectorParity, RawOutputParity) {
  if (!bt::raw_reference_path().empty()) BAYKUS_REQUIRE_FILES({bt::raw_reference_path()});
  baykus::Detector det((bt::models_root() / "e3c.onnx").string());
  for (const auto& f : run_frames(det)) {
    SCOPED_TRACE(f.name);
    const auto r =
        bp::compare_raw(raw_rows(bt::raw_reference(f.name, f.expected)), raw_rows(f.out));
    EXPECT_TRUE(r.ok) << (r.errors.empty() ? "" : r.errors.front());
  }
}

TEST_F(DetectorParity, LayerParity) {
  baykus::Detector det((bt::models_root() / "e3c.onnx").string());
  const auto cal = baykus::Calibration::load((bt::models_root() / "calibration.json").string());
  for (const auto& f : run_frames(det)) {
    SCOPED_TRACE(f.name);
    auto eval = baykus::decode_rows(f.out, f.lb, 0.001f);
    for (auto& d : eval) d.score = cal.apply(d.score_raw);
    baykus::order(eval);
    std::vector<baykus::Det> track;
    for (const auto& d : eval) {
      if (d.score_raw >= 0.1f) track.push_back(d);
    }
    const auto display = baykus::filter_display(track, 0.25);
    const auto& want = f.expected.at("modules").at("dets");
    const struct {
      const char* name;
      const std::vector<baykus::Det>& got;
      double boundary;
    } layers[] = {{"eval", eval, 0.001},
                  {"track", track, 0.1},
                  {"display", display, cal.display_boundary(0.25)}};
    for (const auto& l : layers) {
      const auto r = bp::compare_layer(bt::dets_from_json(want.at(l.name)), l.got, l.boundary);
      EXPECT_TRUE(r.ok) << l.name << ": " << (r.errors.empty() ? "" : r.errors.front());
    }
  }
}
