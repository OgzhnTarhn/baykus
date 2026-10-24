// Sabit Türkçe metin tabloları (Faz 4 tasarımı §5.5, §5.13): motor olay / alarm / bölge türlerini
// İngilizce kodla verir; Türkçe metin yalnız ekranda ve yalnız burada.
export const EVENT_TEXT: Record<string, string> = {
  enter: "bölgeye girdi",
  exit: "bölgeden çıktı",
  exit_lost: "bölgedeyken kayboldu",
  stopped: "bölgede durdu",
  line_cross: "hattı geçti",
  alarm_start: "alarm başladı",
  alarm_ack: "alarm onaylandı",
  alarm_clear: "alarm kapandı",
  map_provisional: "konum aranıyor",
  map_relocalized: "yeniden konumlandı",
  map_merged: "harita birleşti",
  map_new_generation: "yeni harita",
  zones_updated: "bölgeler güncellendi",
  session_reset_tracks: "izler sıfırlandı",
};

export const ALARM_TEXT_ZONE: Record<string, string> = {
  intrusion: "Yasak bölgeye giriş",
  dwell_exceeded: "Bekleme sınırı aşıldı",
};

export const ZONE_KIND_TEXT: Record<string, string> = {
  area: "Alan",
  restricted: "Yasak bölge",
  dwell: "Bekleme sınırı",
  line: "Sayım hattı",
};

export const MOTION_TEXT: Record<string, string> = {
  moving: "hareketli",
  stationary: "duran",
  unknown: "bilinmiyor",
};

export const DIR_TEXT: Record<string, string> = {
  a_to_b: "A → B",
  b_to_a: "B → A",
};

// Bilinmeyen kod ekranda kodun kendisiyle görünür (yeni motor türü sessizce kaybolmasın).
export function text(table: Record<string, string>, code: string | null | undefined): string {
  if (code === null || code === undefined) return "";
  return table[code] ?? code;
}
