// Ürün varsayılanları (Faz 3 tasarımı §4.2): kapı `advise` — sağlık ölçülür, uyarı gösterilir, piksel
// kesilmez; `auto` kullanıcı seçimi (temiz veride zarar verdiği ölçüldü).
import type { DetectSettings } from "@/lib/client";

export const DEFAULT_SETTINGS: Omit<DetectSettings, "session"> = {
  conf: 0.25,
  track: true,
  gate: "advise",
};
