import { describe, expect, it } from "vitest";

import { isCurrent, nextEpoch, resetScope, type Stamp } from "@/lib/session";

describe("response stamp (S12)", () => {
  it("Console ignores stale response", () => {
    // istek 1_08 sırasında gönderildi; yanıt gelmeden dizi değişti (yeni dönem)
    const sent: Stamp = { session: "s", epoch: 3 };
    const now: Stamp = { session: "s", epoch: nextEpoch(3) };
    expect(isCurrent(sent, now)).toBe(false);
  });

  it("same session and epoch is current", () => {
    expect(isCurrent({ session: "s", epoch: 2 }, { session: "s", epoch: 2 })).toBe(true);
  });

  it("other session is never current", () => {
    expect(isCurrent({ session: "a", epoch: 2 }, { session: "b", epoch: 2 })).toBe(false);
  });

  it("epochs only grow", () => {
    expect(nextEpoch(0)).toBe(1);
    expect(nextEpoch(41)).toBe(42);
  });
});

describe("reset scope", () => {
  it("seek within sequence resets tracks only", () => {
    // ileri atlama > 1, geri adım, aynı kareyi yeniden isteme, döngü başı: yalnız takipçi sıfırlanır
    expect(resetScope({ seq: "a", index: 3 }, { seq: "a", index: 5 })).toBe("tracks");
    expect(resetScope({ seq: "a", index: 3 }, { seq: "a", index: 2 })).toBe("tracks");
    expect(resetScope({ seq: "a", index: 3 }, { seq: "a", index: 3 })).toBe("tracks");
    expect(resetScope({ seq: "a", index: 14 }, { seq: "a", index: 0 })).toBe("tracks");
    // ardışık kare ve ilk kare: sıfırlama yok
    expect(resetScope({ seq: "a", index: 3 }, { seq: "a", index: 4 })).toBeNull();
    expect(resetScope(null, { seq: "a", index: 0 })).toBeNull();
  });

  it("sequence change resets all", () => {
    expect(resetScope({ seq: "a", index: 3 }, { seq: "b", index: 4 })).toBe("all");
    expect(resetScope({ seq: "a", index: 3 }, { seq: "b", index: 0 })).toBe("all");
  });
});
