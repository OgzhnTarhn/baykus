#include "baykus/event_log.hpp"

#include <cstddef>
#include <optional>
#include <utility>

namespace baykus {
namespace {

// Boş dize → null; değilse dize.
nlohmann::json str_or_null(const std::string& s) {
  if (s.empty()) return nullptr;
  return s;
}

// İsteğe bağlı değer → JSON (yoksa null).
template <typename T>
nlohmann::json opt_or_null(const std::optional<T>& v) {
  if (!v) return nullptr;
  return *v;
}

// CSV alanı: null → boş; virgül / tırnak / CR / LF içeren alan çift tırnakla sarılır.
std::string csv_field(const nlohmann::json& v) {
  if (v.is_null()) return std::string();
  std::string s = v.is_string() ? v.get<std::string>() : v.dump();
  bool quote = false;
  for (char c : s) {
    if (c == ',' || c == '"' || c == '\r' || c == '\n') {
      quote = true;
      break;
    }
  }
  if (!quote) return s;
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') out += '"';
    out += c;
  }
  out += '"';
  return out;
}

}  // namespace

EventLog::EventLog(const LogCfg& cfg) : max_events_(cfg.max_events) {}

nlohmann::json EventLog::append(ZoneEvent e, const std::string& zone_name,
                                const std::string& zone_kind) {
  nlohmann::json rec;
  rec["seq"] = ++last_seq_;
  rec["n"] = e.n;
  rec["frame"] = e.frame;
  rec["kind"] = e.kind;
  rec["zone_id"] = str_or_null(e.zone_id);
  rec["zone_name"] = str_or_null(zone_name);
  rec["zone_kind"] = str_or_null(zone_kind);
  rec["track_id"] = opt_or_null(e.track_id);
  rec["dir"] = opt_or_null(e.dir);
  rec["gen"] = e.gen;
  rec["alarm_id"] = opt_or_null(e.alarm_id);
  records_.push_back(rec);
  while (static_cast<int>(records_.size()) > max_events_) {
    records_.pop_front();
    ++dropped_;
  }
  return rec;
}

nlohmann::json EventLog::since(long seq) const {
  nlohmann::json events = nlohmann::json::array();
  for (const auto& rec : records_) {
    if (rec["seq"].get<long>() > seq) events.push_back(rec);
  }
  nlohmann::json out;
  out["events"] = std::move(events);
  out["last_seq"] = last_seq_;
  out["dropped"] = dropped_;
  return out;
}

std::string EventLog::csv_since(long seq) const {
  static const char* const kHeader =
      "seq,n,frame,kind,zone_id,zone_name,zone_kind,track_id,dir,gen,alarm_id";
  static const char* const kFields[] = {"seq",     "n",         "frame",     "kind",
                                        "zone_id", "zone_name", "zone_kind", "track_id",
                                        "dir",     "gen",       "alarm_id"};
  std::string out = kHeader;
  out += "\r\n";
  for (const auto& rec : records_) {
    if (rec["seq"].get<long>() <= seq) continue;
    for (std::size_t i = 0; i < sizeof(kFields) / sizeof(kFields[0]); ++i) {
      if (i > 0) out += ',';
      out += csv_field(rec[kFields[i]]);
    }
    out += "\r\n";
  }
  return out;
}

void EventLog::reset() {
  records_.clear();
  last_seq_ = 0;
  dropped_ = 0;
}

}  // namespace baykus
