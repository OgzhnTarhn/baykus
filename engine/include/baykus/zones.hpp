#pragma once

// Faz 4 bölgeler, kurallar, alarmlar (docs/design/zones_rules.md). Davranış yalnız şartname +
// C++ testleri; parametreler `ZonesCfg` (configs/engine/motion.json "zones").
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "baykus/ego.hpp"
#include "baykus/motion.hpp"
#include "baykus/motion_config.hpp"

namespace baykus {

// Bölge tanımı; `kind` = area | restricted | dwell | line.
struct ZoneDef {
  std::string id;
  std::string name;
  std::string kind;
  int gen;
  std::vector<cv::Point2d> points;
  std::optional<int> dwell_frames;
  std::string count;
};

// Bölge olayı; `kind` = enter | exit | exit_lost | stopped | line_cross | alarm_start |
// alarm_clear | alarm_ack.
struct ZoneEvent {
  int n;
  int frame;
  std::string kind;
  std::string zone_id;
  std::optional<int> track_id;
  std::optional<std::string> dir;
  int gen;
  std::optional<std::string> alarm_id;
  nlohmann::json to_json() const;
};

// Açık alarm; `kind` = intrusion | dwell_exceeded.
struct Alarm {
  std::string id;
  std::string kind;
  std::string zone_id;
  std::vector<int> track_ids;
  int since_n;
  int since_frame;
  bool acked;
  nlohmann::json to_json() const;
};

// Bölge durumu, geçişler ve alarmlar (şartname §Aktiflik, §Gözlem, §Çokgen, §Hat, §Alarmlar).
class Zones {
 public:
  explicit Zones(const ZonesCfg& cfg);

  // Doğrulama ilk hatada std::invalid_argument; hata varsa hiçbir durum değişmez. Değişmeyen
  // bölgenin durumu korunur; değişen / çıkarılan bölgenin açık alarmları `cleared`'a yazılır.
  void set(std::vector<ZoneDef> defs, int max_gen, std::vector<ZoneEvent>* cleared = nullptr);

  std::vector<ZoneEvent> update(int n, int frame, const EgoResult& ego,
                                const std::vector<TrackOut>& tracks,
                                const std::map<int, TrackMotion>& motion);

  // Açık alarm yoksa false; acked false ise true yapılır ve `out`'a alarm_ack eklenir.
  bool ack(const std::string& alarm_id, int n, int frame, std::vector<ZoneEvent>* out);

  std::vector<Alarm> alarms() const;
  nlohmann::json counts() const;
  nlohmann::json defs_json() const;
  // Bölge tanımı; yoksa nullptr.
  const ZoneDef* find(const std::string& id) const;
  void reset_tracks();
  void reset();

 private:
  // İz–bölge durumu (şartname §Gözlem / §Çokgen / §Hat).
  struct TrackZone {
    std::string state = "none";  // none | inside | outside (çokgen türleri)
    // Koşu (in/out ya da hat tarafı): taraf, uzunluk, ilk gözlem n / frame / konumu.
    int run_side = 0;
    int run_len = 0;
    int run_n = 0;
    int run_frame = 0;
    cv::Point2d run_pos{0.0, 0.0};
    int gap = 0;       // art arda `yok` gözlem sayısı
    int absent = 0;    // bu karede `tracks`'te yok sayısı
    int absent_n = 0;  // ilk yok karesinin n / frame'i
    int absent_frame = 0;
    int stopped_run = 0;  // durağan koşusu
    int stopped_n = 0;    // durağan koşusunun ilk gözlemi
    int stopped_frame = 0;
    int moving_run = 0;  // hareketli koşusu
    bool stopped_fired = false;
    int prev_side = 0;    // hat: önceki gözlemin tarafı
    int stable_side = 0;  // hat: kararlı taraf (0 = yok)
    cv::Point2d last_stable_pos{0.0, 0.0};
    std::string last_motion = "unknown";  // son karedeki hareket
  };

  // Bölge çalışma durumu.
  struct ZoneState {
    ZoneDef def;
    bool active = false;
    std::map<int, TrackZone> tracks;
    std::set<int> entered;   // restricted: girmiş izler
    std::set<int> exceeded;  // dwell: eşiği aşmış izler
    int a_to_b = 0;
    int b_to_a = 0;
  };

  // Askıya alma: açık alarmlar için alarm_clear, iz durumları ve sayaçlar silinir.
  void suspend_(ZoneState& z, int n, int frame, std::vector<ZoneEvent>& out);
  // Bütün "art arda" sayaçları (in/out, durağan, hareketli, hat taraf koşusu) sıfırlar.
  static void reset_runs_(TrackZone& tz);
  // Çokgen türü gözlemi (şartname §Çokgen türleri).
  void observe_polygon_(ZoneState& z, int track_id, TrackZone& tz, const cv::Point2d& p,
                        const std::string& motion, int n, int frame, std::vector<ZoneEvent>& out);
  // Hat gözlemi (şartname §Hat).
  void observe_line_(ZoneState& z, int track_id, TrackZone& tz, const cv::Point2d& p, int n,
                     int frame, std::vector<ZoneEvent>& out);
  // Aktif bölgenin alarmları (şartname §Alarmlar): önce temizlemeler, sonra açılışlar.
  void reconcile_alarms_(ZoneState& z, int n, int frame, std::vector<ZoneEvent>& out);

  ZonesCfg cfg_;
  std::vector<ZoneState> zones_;
  std::vector<Alarm> alarms_;  // açık alarmlar, açılış sırasıyla
  int last_n_ = 0;
  int last_frame_ = 0;
  bool has_ego_ = false;  // hiç update yoksa askıda
  int ego_gen_ = 0;       // son görülen ego'nun map_gen'i
  bool ego_provisional_ = false;
};

// Bölge gövdesini ayrıştırır (docs/design/events_rules.md 'Bölge gövdesi ayrıştırma'). Tür hatası →
// std::invalid_argument (Türkçe, bölge sırası / id ile). Değer sınırları `Zones::set`'te.
std::vector<ZoneDef> parse_zone_defs(const nlohmann::json& body);

}  // namespace baykus
