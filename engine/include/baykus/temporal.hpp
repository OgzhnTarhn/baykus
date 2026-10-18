#pragma once

// Hizasızlığın zaman kuralı (Faz 3 tasarımı §4.2): Python `baykus.health.temporal.MisalignWindow`.
// Pencere oturumun son `window` karesi (geçerli dahil); S = penceredeki `misaligned_frame`
// kareleri; v̄ = S'nin (dx_px, dy_px) vektörlerinin bileşen bazında ortancası (çift sayıda ortadaki
// ikisinin ortalaması); C = {i ∈ S : ‖vᵢ − v̄‖₂ ≤ tol_px} (sınır dahil, double). `misaligned` = |C|
// ≥ min_frames ve iki kamera da kayıp değil; kapı ve neden yeniden hesaplanır.
#include <deque>
#include <nlohmann/json.hpp>

namespace baykus {

class MisalignWindow {
 public:
  MisalignWindow(int window, int min_frames, double tol_px);

  // Tek kare sağlık sözlüğü (`assess` çıktısı) → zaman kuralı uygulanmış kopyası. İşaretli karede
  // kayma vektörü yoksa std::invalid_argument.
  nlohmann::json update(const nlohmann::json& h);
  void reset();

 private:
  struct Entry {
    bool flagged;
    double dx;
    double dy;
  };
  int window_;
  int min_frames_;
  double tol_px_;
  std::deque<Entry> buf_;
};

}  // namespace baykus
