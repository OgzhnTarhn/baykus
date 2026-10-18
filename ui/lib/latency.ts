// Canlı gecikme (Faz 3 tasarımı §4.4 FR-U3): son 100 karede motorun timing_ms'i; p50 / p95
// baykus_bench'teki yüzdelikle aynı (numpy linear, engine/src/bench.cpp). Aşama kırılımı pencere
// ortalaması; motor total'ı aşamaların toplamı olduğu için kırılım toplamı ortalama total'a eşit.
export const STAGES = ["decode", "health", "pre", "infer", "post", "track"] as const;
export type Stage = (typeof STAGES)[number];

export const LATENCY_WINDOW = 100;

export function percentile(values: number[], q: number): number {
  if (values.length === 0) throw new Error("boş dizide yüzdelik yok");
  const v = [...values].sort((a, b) => a - b);
  const idx = (q / 100) * (v.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, v.length - 1);
  const t = idx - lo;
  const a = v[lo];
  const b = v[hi];
  return t >= 0.5 ? b - (b - a) * (1 - t) : a + (b - a) * t; // numpy _lerp
}

export class LatencyWindow {
  private rows: Record<string, number>[] = [];

  push(timing: Record<string, number>): void {
    this.rows.push(timing);
    if (this.rows.length > LATENCY_WINDOW) this.rows.shift();
  }

  totals(): number[] {
    return this.rows.map((r) => r.total);
  }

  stats(): { p50: number; p95: number; n: number } | null {
    const t = this.totals();
    if (t.length === 0) return null;
    return { p50: percentile(t, 50), p95: percentile(t, 95), n: t.length };
  }

  meanTotal(): number | null {
    const t = this.totals();
    return t.length === 0 ? null : t.reduce((a, b) => a + b, 0) / t.length;
  }

  breakdown(): { stage: Stage; ms: number }[] {
    const n = this.rows.length;
    return STAGES.map((stage) => ({
      stage,
      ms: n === 0 ? 0 : this.rows.reduce((a, r) => a + (r[stage] ?? 0), 0) / n,
    }));
  }

  reset(): void {
    this.rows = [];
  }
}
