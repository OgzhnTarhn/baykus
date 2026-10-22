#pragma once

// Faz 4 olay günlüğü (docs/design/events_rules.md §EventLog). Kayıtlar JSON olarak tutulur;
// `max_events` aşılırsa en eski kayıt düşer ve `dropped` artar.
#include <deque>
#include <string>

#include <nlohmann/json.hpp>

#include "baykus/motion_config.hpp"
#include "baykus/zones.hpp"

namespace baykus {

class EventLog {
 public:
  explicit EventLog(const LogCfg& cfg);

  // Kaydı ekler (seq = son seq + 1) ve kaydı döner; taşan en eski kayıt düşer.
  nlohmann::json append(ZoneEvent e, const std::string& zone_name, const std::string& zone_kind);

  // `seq`'ten büyük kayıtlar (artan) + son seq + düşen sayısı.
  nlohmann::json since(long seq) const;

  // `seq`'ten büyük kayıtlar CSV (RFC 4180, satır sonu "\r\n", başlık dahil).
  std::string csv_since(long seq) const;

  long dropped() const { return dropped_; }
  long last_seq() const { return last_seq_; }
  void reset();

 private:
  int max_events_;
  std::deque<nlohmann::json> records_;
  long last_seq_ = 0;
  long dropped_ = 0;
};

}  // namespace baykus
