// Gecikme penceresi (Faz 3 tasarımı §4.4 FR-U3; plan Görev 32): son 100 kare, p50 / p95 (motorun
// baykus_bench'iyle aynı yüzdelik: numpy linear), aşama kırılımı.
import { describe, expect, it } from "vitest";

import { LatencyWindow, percentile, STAGES } from "@/lib/latency";

const timing = (total: number) => {
  const t: Record<string, number> = { decode: 1, health: 2, pre: 3, infer: total - 10, post: 3 };
  t.track = 1;
  t.total = total;
  return t;
};

describe("latency", () => {
  it("percentile matches numpy linear", () => {
    expect(percentile([1, 2, 3, 4], 50)).toBe(2.5);
    expect(percentile([10], 95)).toBe(10);
    expect(percentile([3, 1, 2], 0)).toBe(1);
    expect(percentile([3, 1, 2], 100)).toBe(3);
    expect(() => percentile([], 50)).toThrow();
  });

  it("rolling p50 p95 over last 100", () => {
    const w = new LatencyWindow();
    expect(w.stats()).toBeNull();
    for (let i = 1; i <= 150; i++) w.push(timing(i + 20));
    const s = w.stats()!;
    expect(s.n).toBe(100);
    // pencere 71..170: p50 = (120 + 121) / 2; p95 sıra 94,05 → 165 + 0,05
    expect(s.p50).toBeCloseTo(120.5, 9);
    expect(s.p95).toBeCloseTo(165.05, 9);
    expect(w.totals()).toHaveLength(100);
    expect(w.totals()[0]).toBe(71);
  });

  it("stage breakdown sums to total", () => {
    const w = new LatencyWindow();
    for (const t of [40, 80, 120]) w.push(timing(t));
    const b = w.breakdown();
    expect(b.map((x) => x.stage)).toEqual([...STAGES]);
    const sum = b.reduce((a, x) => a + x.ms, 0);
    expect(sum).toBeCloseTo(w.meanTotal()!, 9);
    expect(w.meanTotal()).toBeCloseTo(80, 9);
  });

  it("reset empties the window", () => {
    const w = new LatencyWindow();
    w.push(timing(50));
    w.reset();
    expect(w.stats()).toBeNull();
    expect(w.breakdown().every((x) => x.ms === 0)).toBe(true);
  });
});
