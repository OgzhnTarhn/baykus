// Kabul 1-2 kanıt dosyası: her kare fikstüründe ham çıktı (a), üç katman (b), letterbox (Kabul 2)
// karşılaştırmasının sayıları → results/engine/parity_detect.json (Faz 3 plan Görev 19).
// Kullanım: baykus_parity_report <çıktı.json>   (BAYKUS_FIXTURES, BAYKUS_MODELS ortamdan)
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../tests/fixtures.hpp"
#include "baykus/calibration.hpp"
#include "baykus/detections.hpp"
#include "baykus/detector.hpp"
#include "baykus/image_io.hpp"
#include "baykus/parity.hpp"
#include "baykus/preprocess.hpp"
#include "baykus/version.hpp"

namespace bt = baykus::test;
namespace bp = baykus::parity;
using json = nlohmann::json;

namespace {

json layer_json(const bp::LayerCompare& r) {
  return {{"ok", r.ok},         {"py_rows", r.py_rows},       {"cpp_rows", r.cpp_rows},
          {"compared", r.compared}, {"free_rows", r.free_rows}, {"swaps", r.swaps},
          {"ratio", r.ratio},   {"max_raw_diff", r.max_raw_diff}, {"min_iou", r.min_iou},
          {"errors", r.errors}};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "kullanım: baykus_parity_report <çıktı.json>\n";
    return 2;
  }
  const auto root = bt::fixtures_root();
  const auto models = bt::models_root();
  baykus::Detector det((models / "e3c.onnx").string());
  const auto cal = baykus::Calibration::load((models / "calibration.json").string());
  const double display_boundary = cal.display_boundary(0.25);
  json frames = json::array();
  bool all_ok = true;
  for (const auto& e : bt::fixtures_of("frame")) {
    const auto exp = bt::read_json(root / e.at("path").get<std::string>());
    const auto& in = exp.at("inputs");
    auto p = baykus::decode_pair(bt::read_file(root / in.at("rgb").get<std::string>()),
                                 bt::read_file(root / in.at("ir").get<std::string>()));
    baykus::Letterbox lb{};
    const auto x = baykus::letterbox(baykus::stack_rgbt(p.bgr, p.ir), det.imgsz(), &lb);
    const auto& mod = exp.at("modules");
    const bool lb_ok = bt::sha256_hex(x.data(), x.size() * sizeof(float)) ==
                           mod.at("letterbox").at("sha256").get<std::string>() &&
                       lb.gain == mod.at("letterbox").at("gain").get<double>();
    const auto out = det.infer(x);

    std::vector<bp::Row> py_raw, cpp_raw(baykus::kRows);
    for (const auto& r : mod.at("raw")) {
      bp::Row row{};
      for (int k = 0; k < baykus::kCols; ++k) row[k] = static_cast<float>(r.at(k).get<double>());
      py_raw.push_back(row);
    }
    for (int i = 0; i < baykus::kRows; ++i)
      for (int k = 0; k < baykus::kCols; ++k) cpp_raw[i][k] = out[i * baykus::kCols + k];
    const auto raw = bp::compare_raw(py_raw, cpp_raw);

    auto eval = baykus::decode_rows(out, lb, 0.001f);
    for (auto& d : eval) d.score = cal.apply(d.score_raw);
    baykus::order(eval);
    std::vector<baykus::Det> track;
    for (const auto& d : eval)
      if (d.score_raw >= 0.1f) track.push_back(d);
    const auto display = baykus::filter_display(track, 0.25);
    const auto& want = mod.at("dets");
    const auto le = bp::compare_layer(bt::dets_from_json(want.at("eval")), eval, 0.001);
    const auto lt = bp::compare_layer(bt::dets_from_json(want.at("track")), track, 0.1);
    const auto ld = bp::compare_layer(bt::dets_from_json(want.at("display")), display, display_boundary);
    const bool ok = lb_ok && raw.ok && le.ok && lt.ok && ld.ok;
    all_ok = all_ok && ok;
    frames.push_back({{"name", e.at("name")},
                      {"ok", ok},
                      {"letterbox_sha_equal", lb_ok},
                      {"raw", {{"ok", raw.ok}, {"max_diff", raw.max_diff}, {"swaps", raw.swaps}, {"errors", raw.errors}}},
                      {"layers", {{"eval", layer_json(le)}, {"track", layer_json(lt)}, {"display", layer_json(ld)}}}});
  }
  const json rec = {
      {"kabul", "1 (a, b modül düzeyi) + 2"},
      {"passed", all_ok},
      {"n_frames", frames.size()},
      {"display_boundary_raw", display_boundary},
      {"opencv_version", baykus::opencv_version()},
      {"onnxruntime_version", baykus::onnxruntime_version()},
      {"tolerances",
       {{"raw", bp::kRawTol}, {"raw_tie", bp::kRawTieTol}, {"layer_iou", bp::kLayerIoU},
        {"layer_raw", bp::kLayerRawTol}, {"boundary", bp::kBoundaryTol}, {"min_ratio", bp::kMinRatio}}},
      {"frames", frames},
  };
  std::ofstream(argv[1]) << rec.dump(2) << "\n";
  std::cout << argv[1] << ": " << frames.size() << " kare, Kabul 1-2 " << (all_ok ? "geçti" : "GEÇMEDİ")
            << "\n";
  return all_ok ? 0 : 1;
}
