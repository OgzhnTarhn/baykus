// Kabul 3 b: zaman kuralı birim senaryoları (`temporal_unit.json`; Python'da üretilmiş 50 karelik
// diziler: sabit kayma, zıplayan kestirim, ortada başlayan kayma, 3 px sınırı, çift sayıda S,
// sıfırlama, kayıp/gürültü önceliği).
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "baykus/health.hpp"
#include "baykus/temporal.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
using json = nlohmann::json;

TEST(Temporal, UnitSequences) {
  const auto m = bt::models_root();
  BAYKUS_REQUIRE_FILES({bt::fixtures_root() / "index.json", m / "gate_model.json"});
  const auto gm = baykus::GateModel::load((m / "gate_model.json").string());
  const auto entries = bt::fixtures_of("temporal_unit");
  ASSERT_EQ(entries.size(), 1u);
  const auto scenarios =
      bt::read_json(bt::fixtures_root() / entries[0].at("path").get<std::string>());
  ASSERT_FALSE(scenarios.empty());
  for (const auto& sc : scenarios) {
    SCOPED_TRACE(sc.at("name").get<std::string>());
    baykus::MisalignWindow w(gm.window, gm.min_frames, gm.tol_px);
    int frame = 0;
    for (const auto& st : sc.at("steps")) {
      if (st.contains("reset")) {
        w.reset();
        continue;
      }
      ++frame;
      SCOPED_TRACE("kare " + std::to_string(frame));
      const json got = w.update(st.at("input"));
      const auto& want = st.at("expected");
      EXPECT_EQ(got.at("pair").at("misaligned"), want.at("misaligned"));
      EXPECT_EQ(got.at("gate"), want.at("gate"));
      EXPECT_EQ(got.at("reason_code"), want.at("reason_code"));
      EXPECT_EQ(got.at("reason"), want.at("reason"));
    }
    EXPECT_EQ(frame, 50);
  }
}

namespace {

json frame(bool flagged, json dx, json dy) {
  return {{"rgb", {{"flags", json::array()}}},
          {"ir", {{"flags", json::array()}}},
          {"pair", {{"misaligned_frame", flagged}, {"dx_px", dx}, {"dy_px", dy}}}};
}

}  // namespace

TEST(Temporal, FlaggedFrameWithoutVectorThrows) {
  baykus::MisalignWindow w(5, 3, 3.0);
  EXPECT_THROW(w.update(frame(true, nullptr, 0.0)), std::invalid_argument);
  EXPECT_NO_THROW(w.update(frame(false, nullptr, nullptr)));
}

TEST(Temporal, InputIsNotModified) {
  baykus::MisalignWindow w(5, 3, 3.0);
  const json in = frame(true, 20.0, 0.0);
  for (int i = 0; i < 3; ++i) w.update(in);
  const json out = w.update(in);
  EXPECT_FALSE(in.at("pair").contains("misaligned"));
  EXPECT_EQ(out.at("pair").at("misaligned"), true);
  EXPECT_EQ(out.at("reason"), "IR hizasız");
}
