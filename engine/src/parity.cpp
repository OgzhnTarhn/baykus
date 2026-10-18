#include "baykus/parity.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace baykus::parity {

double iou(double ax1, double ay1, double ax2, double ay2, double bx1, double by1, double bx2,
           double by2) {
  const double iw = std::min(ax2, bx2) - std::max(ax1, bx1);
  const double ih = std::min(ay2, by2) - std::max(ay1, by1);
  if (iw <= 0 || ih <= 0) return 0.0;
  const double inter = iw * ih;
  const double uni = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
  return uni > 0 ? inter / uni : 0.0;
}

namespace {

double row_iou(const Row& a, const Row& b) {
  return iou(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
}

double row_diff(const Row& a, const Row& b) {
  double d = 0.0;
  for (int k = 0; k < kCols; ++k) d = std::max(d, std::fabs(static_cast<double>(a[k]) - b[k]));
  return d;
}

}  // namespace

RawCompare compare_raw(const std::vector<Row>& py, const std::vector<Row>& cpp) {
  RawCompare r;
  r.rows = static_cast<int>(py.size());
  if (py.size() != cpp.size()) {
    r.ok = false;
    r.errors.push_back("satır sayısı farklı");
    return r;
  }
  const std::size_t n = py.size();
  std::size_t i = 0;
  while (i < n) {
    // Ardışık Python skorları 1e-5 içindeyse bir grup: ORT'nin sıralaması işlemciye göre
    // değişebilir
    std::size_t j = i + 1;
    while (j < n && std::fabs(static_cast<double>(py[j][4]) - py[j - 1][4]) < kRawTieTol) ++j;
    std::vector<bool> used(j - i, false);
    for (std::size_t a = i; a < j; ++a) {
      std::size_t best = a;
      if (j - i > 1) {
        double best_iou = -1.0;
        for (std::size_t b = i; b < j; ++b) {
          if (used[b - i]) continue;
          const double v = row_iou(py[a], cpp[b]);
          if (v > best_iou) best_iou = v, best = b;
        }
      }
      used[best - i] = true;
      if (best != a) ++r.swaps;
      const double d = row_diff(py[a], cpp[best]);
      r.max_diff = std::max(r.max_diff, d);
      if (d > kRawTol) {
        std::ostringstream ss;
        ss << "satır " << a << ": fark " << d << " > " << kRawTol;
        r.errors.push_back(ss.str());
      }
    }
    i = j;
  }
  r.swaps /= 2;  // her yer değiştirme iki satırı taşır
  r.ok = r.errors.empty();
  return r;
}

LayerCompare compare_layer(const std::vector<Det>& py, const std::vector<Det>& cpp,
                           double boundary_raw) {
  LayerCompare r;
  r.py_rows = static_cast<int>(py.size());
  r.cpp_rows = static_cast<int>(cpp.size());
  std::map<int, const Det*> pm, cm;
  for (const auto& d : py) pm[d.index] = &d;
  for (const auto& d : cpp) cm[d.index] = &d;
  auto near = [&](float raw) {
    return std::fabs(static_cast<double>(raw) - boundary_raw) < kBoundaryTol;
  };
  // Serbestlik yalnız üyelikte: sınıra yakın satır bir tarafta olmayabilir. İki tarafta da varsa
  // normal karşılaştırılır (tasarım §6.1 Kabul 1 b: "kümede olup olmamakta serbest").
  std::set<int> free;
  for (const auto& d : py)
    if (near(d.score_raw) && !cm.count(d.index)) free.insert(d.index);
  for (const auto& d : cpp)
    if (near(d.score_raw) && !pm.count(d.index)) free.insert(d.index);
  r.free_rows = static_cast<int>(free.size());

  auto err = [&](const std::string& s) { r.errors.push_back(s); };
  for (const auto& [idx, d] : pm) {
    (void)d;
    if (!free.count(idx) && !cm.count(idx))
      err("satır " + std::to_string(idx) + " yalnız Python'da");
  }
  for (const auto& [idx, d] : cm) {
    (void)d;
    if (!free.count(idx) && !pm.count(idx)) err("satır " + std::to_string(idx) + " yalnız C++'ta");
  }
  std::vector<int> py_seq, cpp_seq;
  for (const auto& d : py)
    if (!free.count(d.index) && cm.count(d.index)) py_seq.push_back(d.index);
  for (const auto& d : cpp)
    if (!free.count(d.index) && pm.count(d.index)) cpp_seq.push_back(d.index);
  for (int idx : py_seq) {
    const Det& a = *pm[idx];
    const Det& b = *cm[idx];
    const double v = iou(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2);
    const double dr = std::fabs(static_cast<double>(a.score_raw) - b.score_raw);
    r.min_iou = std::min(r.min_iou, v);
    r.max_raw_diff = std::max(r.max_raw_diff, dr);
    if (v < kLayerIoU) err("satır " + std::to_string(idx) + ": IoU " + std::to_string(v));
    if (dr > kLayerRawTol) err("satır " + std::to_string(idx) + ": ham fark " + std::to_string(dr));
  }
  r.compared = static_cast<int>(py_seq.size());
  for (std::size_t k = 0; k < py_seq.size() && k < cpp_seq.size();) {
    if (py_seq[k] == cpp_seq[k]) {
      ++k;
      continue;
    }
    const bool adjacent =
        k + 1 < py_seq.size() && py_seq[k] == cpp_seq[k + 1] && py_seq[k + 1] == cpp_seq[k];
    const Det& a = *pm[py_seq[k]];
    const Det& b = *pm[adjacent ? py_seq[k + 1] : py_seq[k]];
    if (adjacent && a.score == b.score &&
        std::fabs(static_cast<double>(a.score_raw) - b.score_raw) <= kSwapRawTol) {
      ++r.swaps;
      k += 2;
      continue;
    }
    err("sıra farkı, konum " + std::to_string(k));
    break;
  }
  const int total = std::max(r.py_rows, r.cpp_rows);
  r.ratio = total > 0 ? static_cast<double>(r.compared) / total : 1.0;
  if (r.ratio < kMinRatio) {
    std::ostringstream ss;
    ss << "karşılaştırılan oran " << r.ratio << " < " << kMinRatio;
    err(ss.str());
  }
  r.ok = r.errors.empty();
  return r;
}

HealthCompare compare_health(const nlohmann::json& py, const nlohmann::json& cpp) {
  using json = nlohmann::json;
  HealthCompare r;
  auto fail = [&r](const std::string& msg) { r.errors.push_back(msg); };
  auto field = [](const json& j, const char* ptr) -> const json* {
    const json::json_pointer p(ptr);
    return j.contains(p) ? &j.at(p) : nullptr;
  };
  enum class Kind { kRel, kPx, kResp };
  auto number = [&](const char* ptr, Kind kind) {
    const json* a = field(py, ptr);
    const json* b = field(cpp, ptr);
    if (a == nullptr || b == nullptr) return fail(std::string(ptr) + ": alan yok");
    if (a->is_null() || b->is_null()) {
      if (a->is_null() != b->is_null()) {
        fail(std::string(ptr) + ": py=" + a->dump() + " cpp=" + b->dump());
      }
      return;
    }
    const double x = a->get<double>();
    const double y = b->get<double>();
    const double d = std::fabs(x - y);
    bool ok = true;
    if (kind == Kind::kRel) {
      const double scale = std::max(std::fabs(x), std::fabs(y));
      r.max_rel = std::max(r.max_rel, scale > 0 ? d / scale : 0.0);
      ok = d <= kHealthRel * scale + 1e-12;
    } else if (kind == Kind::kPx) {
      r.max_px = std::max(r.max_px, d);
      ok = d <= kHealthPx;
    } else {
      r.max_resp = std::max(r.max_resp, d);
      ok = d <= kHealthResp;
    }
    if (!ok) {
      std::ostringstream ss;
      ss.precision(17);
      ss << ptr << ": py=" << x << " cpp=" << y;
      fail(ss.str());
    }
  };
  auto same = [&](const char* ptr) {
    const json* a = field(py, ptr);
    const json* b = field(cpp, ptr);
    if (a == nullptr || b == nullptr) return fail(std::string(ptr) + ": alan yok");
    if (*a != *b) fail(std::string(ptr) + ": py=" + a->dump() + " cpp=" + b->dump());
  };
  for (const char* p : {"/rgb/std", "/rgb/noise", "/rgb/blur", "/rgb/brightness", "/ir/std",
                        "/ir/noise", "/ir/blur"}) {
    number(p, Kind::kRel);
  }
  for (const char* p : {"/pair/misalign_px", "/pair/dx_px", "/pair/dy_px"}) number(p, Kind::kPx);
  number("/pair/response", Kind::kResp);
  for (const char* p :
       {"/rgb/state", "/rgb/flags", "/ir/state", "/ir/flags", "/pair/misaligned_frame",
        "/pair/misaligned", "/gate", "/reason_code", "/reason"}) {
    same(p);
  }
  if (py.contains("applied")) same("/applied");
  r.ok = r.errors.empty();
  return r;
}

}  // namespace baykus::parity
