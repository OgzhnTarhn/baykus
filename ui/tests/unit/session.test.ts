import { describe, expect, it } from "vitest";

import { isCurrent, nextEpoch, type Stamp } from "@/lib/session";

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
