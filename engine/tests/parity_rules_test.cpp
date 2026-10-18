// Kabul 1 kurallarının kendisi (sentetik listeler): kurallar yanlışsa eşlik testi bir şey
// kanıtlamaz.
#include <gtest/gtest.h>

#include <vector>

#include "baykus/parity.hpp"

namespace bp = baykus::parity;
using baykus::Det;

namespace {

Det d(int index, float raw, double score, double x = 0.0) {
  return Det{index, 0, raw, score, x, 0.0, x + 10.0, 10.0};
}

bp::Row row(float x, float score) { return bp::Row{x, 0.0f, x + 10.0f, 10.0f, score, 0.0f}; }

}  // namespace

TEST(ParityRules, RawExactPasses) {
  std::vector<bp::Row> py{row(0, 0.9f), row(20, 0.8f)};
  auto r = bp::compare_raw(py, py);
  EXPECT_TRUE(r.ok);
  EXPECT_EQ(r.swaps, 0);
}

TEST(ParityRules, RawDiffAboveTolFails) {
  std::vector<bp::Row> py{row(0, 0.9f)};
  std::vector<bp::Row> cpp{row(0, 0.9f + 2e-4f)};
  EXPECT_FALSE(bp::compare_raw(py, cpp).ok);
}

TEST(ParityRules, RawSwapWithinTieIsMatchedByIoU) {
  std::vector<bp::Row> py{row(0, 0.5f), row(50, 0.5f + 5e-6f), row(100, 0.1f)};
  std::vector<bp::Row> cpp{row(50, 0.5f + 5e-6f), row(0, 0.5f), row(100, 0.1f)};
  auto r = bp::compare_raw(py, cpp);
  EXPECT_TRUE(r.ok);
  EXPECT_EQ(r.swaps, 1);
}

TEST(ParityRules, RawSwapOutsideTieFails) {
  std::vector<bp::Row> py{row(0, 0.5f), row(50, 0.5f + 5e-5f)};
  std::vector<bp::Row> cpp{row(50, 0.5f + 5e-5f), row(0, 0.5f)};
  EXPECT_FALSE(bp::compare_raw(py, cpp).ok);
}

TEST(ParityRules, LayerIdenticalPasses) {
  std::vector<Det> py{d(3, 0.9f, 0.95), d(1, 0.5f, 0.6)};
  auto r = bp::compare_layer(py, py, 0.1);
  EXPECT_TRUE(r.ok);
  EXPECT_EQ(r.compared, 2);
}

static std::vector<Det> many(int n) {
  std::vector<Det> v;
  for (int i = 0; i < n; ++i) v.push_back(d(i, 0.9f - i * 1e-3f, 0.9 - i * 1e-3, i * 20.0));
  return v;
}

TEST(ParityRules, BoundaryRowIsFree) {
  auto py = many(200);
  const auto cpp = py;
  py.push_back(d(500, 0.10005f, 0.2, 9000.0));  // sınıra 1e-4'ten yakın: kümede olmayabilir
  auto r = bp::compare_layer(py, cpp, 0.1);
  EXPECT_TRUE(r.ok) << (r.errors.empty() ? "" : r.errors[0]);
  EXPECT_EQ(r.free_rows, 1);
}

TEST(ParityRules, MissingRowFarFromBoundaryFails) {
  std::vector<Det> py{d(3, 0.9f, 0.95), d(7, 0.3f, 0.3)};
  std::vector<Det> cpp{d(3, 0.9f, 0.95)};
  EXPECT_FALSE(bp::compare_layer(py, cpp, 0.1).ok);
}

TEST(ParityRules, SwapOfEqualCalibratedNeighboursAllowed) {
  std::vector<Det> py{d(1, 0.30005f, 0.3, 0), d(2, 0.3f, 0.3, 50)};
  std::vector<Det> cpp{d(2, 0.3f, 0.3, 50), d(1, 0.30005f, 0.3, 0)};
  auto r = bp::compare_layer(py, cpp, 0.1);
  EXPECT_TRUE(r.ok);
  EXPECT_EQ(r.swaps, 1);
}

TEST(ParityRules, UnallowedOrderChangeFails) {
  std::vector<Det> py{d(1, 0.5f, 0.6, 0), d(2, 0.3f, 0.3, 50)};
  std::vector<Det> cpp{d(2, 0.3f, 0.3, 50), d(1, 0.5f, 0.6, 0)};
  EXPECT_FALSE(bp::compare_layer(py, cpp, 0.1).ok);
}

TEST(ParityRules, LowIoUFails) {
  std::vector<Det> py{d(1, 0.5f, 0.6, 0)};
  std::vector<Det> cpp{d(1, 0.5f, 0.6, 1)};  // 1 px kayma: IoU 9/11
  EXPECT_FALSE(bp::compare_layer(py, cpp, 0.1).ok);
}

TEST(ParityRules, ComparedRatioBelow99Fails) {
  std::vector<Det> py, cpp;
  for (int i = 0; i < 50; ++i) {
    py.push_back(d(i, 0.5f - i * 1e-3f, 0.5, i * 20.0));
    cpp.push_back(py.back());
  }
  py.push_back(d(99, 0.10003f, 0.2, 2000.0));  // serbest, ama oran 50/51 < %99
  auto r = bp::compare_layer(py, cpp, 0.1);
  EXPECT_FALSE(r.ok);
  EXPECT_LT(r.ratio, bp::kMinRatio);
}

TEST(ParityRules, BoundaryRowPresentOnBothSidesIsCompared) {
  auto py = many(200);
  py.push_back(d(500, 0.10005f, 0.2, 9000.0));  // sınıra yakın ama iki tarafta da var
  const auto cpp = py;
  auto r = bp::compare_layer(py, cpp, 0.1);
  EXPECT_TRUE(r.ok);
  EXPECT_EQ(r.free_rows, 0);
  EXPECT_EQ(r.compared, 201);
}

TEST(ParityRules, BoundaryRowPresentOnBothSidesStillChecksIoU) {
  auto py = many(200);
  py.push_back(d(500, 0.10005f, 0.2, 9000.0));
  auto cpp = py;
  cpp.back().x1 += 5.0;
  EXPECT_FALSE(bp::compare_layer(py, cpp, 0.1).ok);
}
