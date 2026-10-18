"use client";

// Canlı gecikme grafiği (FR-U3): son 100 karenin motor süresi (düz canvas, kütüphane yok), p50 / p95
// çizgileri ve aşama kırılımı (pencere ortalaması).
import { useEffect, useRef } from "react";

import type { Stage } from "@/lib/latency";

export type LatencyView = {
  totals: number[];
  stats: { p50: number; p95: number; n: number } | null;
  breakdown: { stage: Stage; ms: number }[];
};

const STAGE_COLOR: Record<Stage, string> = {
  decode: "#8a9aab",
  health: "#4cc38a",
  pre: "#b48cf0",
  infer: "#67c6e3",
  post: "#f07fa8",
  track: "#f0a23b",
};

const STAGE_TEXT: Record<Stage, string> = {
  decode: "çözme",
  health: "sağlık",
  pre: "ön işlem",
  infer: "çıkarım",
  post: "son işlem",
  track: "takip",
};

const css = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";

function draw(canvas: HTMLCanvasElement, v: LatencyView) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (v.totals.length === 0 || !v.stats) return;
  const top = Math.max(...v.totals, v.stats.p95) * 1.15;
  const y = (ms: number) => h - (ms / top) * h;
  const x = (i: number) => (i / 99) * w;

  ctx.font = "10px var(--font-geist-mono), monospace";
  for (const [ms, label] of [
    [v.stats.p95, "p95"],
    [v.stats.p50, "p50"],
  ] as const) {
    ctx.strokeStyle = css("--muted");
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, y(ms));
    ctx.lineTo(w, y(ms));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = css("--muted");
    ctx.fillText(`${label} ${ms.toFixed(1)}`, 4, y(ms) - 3);
  }

  ctx.strokeStyle = css("--rgb");
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const start = 100 - v.totals.length; // sağa yaslı: en yeni kare sağda
  v.totals.forEach((ms, i) => {
    const px = x(start + i);
    if (i === 0) ctx.moveTo(px, y(ms));
    else ctx.lineTo(px, y(ms));
  });
  ctx.stroke();
}

export function LatencyChart({ view }: { view: LatencyView }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (ref.current) draw(ref.current, view);
  }, [view]);
  const sum = view.breakdown.reduce((a, b) => a + b.ms, 0);
  return (
    <section className="flex flex-col gap-2 rounded border border-line bg-panel p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h2 className="whitespace-nowrap text-sm font-semibold">Motor gecikmesi</h2>
        <span className="num text-xs text-muted">
          {view.stats
            ? `p50 ${view.stats.p50.toFixed(1)} · p95 ${view.stats.p95.toFixed(1)} ms · ${view.stats.n} kare`
            : "veri yok"}
        </span>
      </div>
      <canvas
        ref={ref}
        className="h-24 w-full"
        role="img"
        aria-label={
          view.stats
            ? `Son ${view.stats.n} karede motor süresi; p50 ${view.stats.p50.toFixed(1)} ms, p95 ${view.stats.p95.toFixed(1)} ms`
            : "Henüz ölçüm yok"
        }
      />
      {sum > 0 && (
        <>
          <div className="flex h-2 w-full overflow-hidden rounded" aria-hidden>
            {view.breakdown.map((b) => (
              <span
                key={b.stage}
                style={{ width: `${(b.ms / sum) * 100}%`, background: STAGE_COLOR[b.stage] }}
              />
            ))}
          </div>
          <dl className="grid grid-cols-3 gap-x-3 gap-y-0.5 text-xs">
            {view.breakdown.map((b) => (
              <div key={b.stage} className="flex items-center justify-between gap-1">
                <dt className="flex items-center gap-1 text-muted">
                  <span
                    className="h-2 w-2 rounded-sm"
                    style={{ background: STAGE_COLOR[b.stage] }}
                    aria-hidden
                  />
                  {STAGE_TEXT[b.stage]}
                </dt>
                <dd className="num">{b.ms.toFixed(1)}</dd>
              </div>
            ))}
          </dl>
        </>
      )}
    </section>
  );
}
