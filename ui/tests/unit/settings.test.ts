// Ürün varsayılanları (Faz 3 tasarımı §4.2: gate=advise; auto kullanıcı seçimi).
import { describe, expect, it } from "vitest";

import { DEFAULT_SETTINGS } from "@/lib/settings";

describe("settings", () => {
  it("default gate is advise, conf 0.25, tracking on", () => {
    expect(DEFAULT_SETTINGS).toEqual({ conf: 0.25, track: true, gate: "advise" });
  });
});
