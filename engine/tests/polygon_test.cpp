#include <gtest/gtest.h>

#include <vector>

#include "baykus/polygon.hpp"

namespace {

using baykus::is_simple;
using baykus::point_in_polygon;
using baykus::polygon_area;
using baykus::segments_intersect;

using P = cv::Point2d;

}  // namespace

TEST(Polygon, SimpleCheck) {
  const std::vector<P> square = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
  EXPECT_TRUE(is_simple(square));
  // İçbükey L.
  const std::vector<P> l_shape = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 4.0},
                                  {4.0, 4.0}, {4.0, 10.0}, {0.0, 10.0}};
  EXPECT_TRUE(is_simple(l_shape));
  // Düz geçen doğrusal köşe basitliği bozmaz.
  const std::vector<P> straight = {{0.0, 0.0}, {5.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}};
  EXPECT_TRUE(is_simple(straight));
  // Papyon: kendini keser.
  const std::vector<P> bowtie = {{0.0, 0.0}, {10.0, 10.0}, {10.0, 0.0}, {0.0, 10.0}};
  EXPECT_FALSE(is_simple(bowtie));
  // Geri katlanan sivri uç.
  const std::vector<P> spike = {{0.0, 0.0}, {10.0, 0.0}, {5.0, 0.0}, {5.0, 5.0}};
  EXPECT_FALSE(is_simple(spike));
  // Yinelenen ardışık nokta (sıfır uzunluklu kenar).
  const std::vector<P> dup = {{0.0, 0.0}, {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}};
  EXPECT_FALSE(is_simple(dup));
  // İki nokta çokgen değil.
  EXPECT_FALSE(is_simple({{0.0, 0.0}, {10.0, 0.0}}));
}

TEST(Polygon, SegmentIntersect) {
  // Çapraz kesişim.
  EXPECT_TRUE(segments_intersect({0.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}, {10.0, 0.0}));
  // Uç noktada dokunma.
  EXPECT_TRUE(segments_intersect({0.0, 0.0}, {10.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}));
  // T dokunma.
  EXPECT_TRUE(segments_intersect({0.0, 0.0}, {10.0, 0.0}, {5.0, 0.0}, {5.0, 10.0}));
  // Doğrusal çakışma.
  EXPECT_TRUE(segments_intersect({0.0, 0.0}, {10.0, 0.0}, {5.0, 0.0}, {15.0, 0.0}));
  // Doğrusal ayrık.
  EXPECT_FALSE(segments_intersect({0.0, 0.0}, {10.0, 0.0}, {20.0, 0.0}, {30.0, 0.0}));
  // Paralel.
  EXPECT_FALSE(segments_intersect({0.0, 0.0}, {10.0, 0.0}, {0.0, 5.0}, {10.0, 5.0}));
  // Kesişim yalnızca uzantıda kalır.
  EXPECT_FALSE(segments_intersect({0.0, 0.0}, {10.0, 0.0}, {5.0, 5.0}, {5.0, 10.0}));
}

TEST(Polygon, PointInPolygon) {
  const std::vector<P> square = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
  EXPECT_TRUE(point_in_polygon({5.0, 5.0}, square));    // iç
  EXPECT_FALSE(point_in_polygon({15.0, 5.0}, square));  // dış
  EXPECT_TRUE(point_in_polygon({5.0, 0.0}, square));    // kenar üstü
  EXPECT_TRUE(point_in_polygon({0.0, 0.0}, square));    // köşe üstü
  // İçbükey L: girinti bölgesi dışarıda.
  const std::vector<P> l_shape = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 4.0},
                                  {4.0, 4.0}, {4.0, 10.0}, {0.0, 10.0}};
  EXPECT_FALSE(point_in_polygon({7.0, 7.0}, l_shape));
  EXPECT_TRUE(point_in_polygon({2.0, 2.0}, l_shape));
}

TEST(Polygon, Area) {
  const std::vector<P> square = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
  EXPECT_NEAR(polygon_area(square), 100.0, 1e-9);
  const std::vector<P> reversed = {{0.0, 0.0}, {0.0, 10.0}, {10.0, 10.0}, {10.0, 0.0}};
  EXPECT_NEAR(polygon_area(reversed), 100.0, 1e-9);
}
