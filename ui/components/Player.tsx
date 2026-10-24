// Oynatıcı (FR-U1, FR-U2): oynat / durdur / kare kare / konum / hız; RGB, IR ya da yan yana görünüm.
// Görüntü ve kutular aynı yanıttan gelir (kare önce yüklenir, sonra birlikte gösterilir).
import type { ReactNode } from "react";

import type { DetectResponse } from "@/lib/client";
import type { PlayerSnapshot } from "@/lib/player";
import { visible } from "@/lib/motion";
import { Overlay } from "@/components/Overlay";

export type FrameView = {
  seq: string;
  frame: number;
  rgbSrc: string;
  irSrc: string;
  response: DetectResponse;
};

export type View = "rgb" | "ir" | "both";

const VIEWS: { value: View; label: string }[] = [
  { value: "rgb", label: "RGB" },
  { value: "ir", label: "IR" },
  { value: "both", label: "Yan yana" },
];

export const SPEEDS = [1, 2, 5, 10, 15, 30];

function Pane({
  title,
  color,
  src,
  fv,
  size,
  onlyMoving,
}: {
  title: string;
  color: string;
  src: string | null;
  fv: FrameView | null;
  size: [number, number];
  onlyMoving: boolean;
}) {
  return (
    <figure className="min-w-0 flex-1">
      <div
        className="relative w-full overflow-hidden rounded border border-line bg-black"
        style={{ aspectRatio: `${size[0]} / ${size[1]}` }}
      >
        {src && (
          // eslint-disable-next-line @next/next/no-img-element -- kare baytları API'den; optimizasyon yok
          <img src={src} alt={`${title} kare ${fv?.frame}`} className="h-full w-full" />
        )}
        {fv && (
          <Overlay
            detections={visible(fv.response.detections, onlyMoving)}
            width={size[0]}
            height={size[1]}
          />
        )}
        <figcaption
          className="absolute left-2 top-2 rounded bg-black/60 px-1.5 text-xs font-medium"
          style={{ color }}
        >
          {title}
        </figcaption>
      </div>
    </figure>
  );
}

export type PlayerProps = {
  badge?: ReactNode;
  snap: PlayerSnapshot<FrameView>;
  size: [number, number];
  view: View;
  onlyMoving?: boolean;
  onView(v: View): void;
  onPlay(): void;
  onPause(): void;
  onStep(): void;
  onSeek(index: number): void;
  onSpeed(fps: number): void;
};

export function Player(p: PlayerProps) {
  const { snap } = p;
  const fv = snap.result;
  const playing = snap.state === "playing";
  const total = snap.frames.length;
  const onlyMoving = p.onlyMoving ?? false;
  return (
    <section className="flex min-w-0 flex-col gap-3" aria-label="Oynatıcı">
      <div className="flex min-h-6 items-center gap-2">{p.badge}</div>
      <div className="flex flex-col gap-3 lg:flex-row">
        {p.view !== "ir" && (
          <Pane
            title="RGB"
            color="var(--rgb)"
            src={fv?.rgbSrc ?? null}
            fv={fv}
            size={p.size}
            onlyMoving={onlyMoving}
          />
        )}
        {p.view !== "rgb" && (
          <Pane
            title="IR"
            color="var(--ir)"
            src={fv?.irSrc ?? null}
            fv={fv}
            size={p.size}
            onlyMoving={onlyMoving}
          />
        )}
      </div>

      {snap.error && (
        <p role="alert" className="rounded border border-alarm/60 bg-alarm/10 px-3 py-2 text-sm">
          {snap.error}. Oynatma durdu; motor açılınca yeniden oynatın.
        </p>
      )}

      <div className="flex flex-wrap items-center gap-3 text-sm">
        <button
          type="button"
          className="w-24 rounded bg-rgb px-3 py-1.5 font-medium text-bg disabled:opacity-40"
          onClick={playing ? p.onPause : p.onPlay}
          disabled={total === 0}
        >
          {playing ? "Durdur" : "Oynat"}
        </button>
        <button
          type="button"
          className="rounded border border-line px-3 py-1.5 disabled:opacity-40"
          onClick={p.onStep}
          disabled={total === 0 || snap.busy}
        >
          Sonraki kare
        </button>
        <label className="flex min-w-48 flex-1 items-center gap-2">
          <span className="sr-only">Konum</span>
          <input
            type="range"
            className="flex-1"
            min={0}
            max={Math.max(0, total - 1)}
            value={Math.max(0, snap.index)}
            onChange={(e) => p.onSeek(Number(e.target.value))}
            disabled={total === 0}
          />
          <span className="num whitespace-nowrap text-muted">
            {snap.frame ?? "–"} · {snap.index + 1}/{total}
          </span>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-muted">Hız üst sınırı</span>
          <select
            className="rounded border border-line bg-panel-2 px-2 py-1"
            value={snap.fps}
            onChange={(e) => p.onSpeed(Number(e.target.value))}
          >
            {SPEEDS.map((v) => (
              <option key={v} value={v}>
                {v} kare/sn
              </option>
            ))}
          </select>
        </label>
        <div className="flex gap-1" role="group" aria-label="Görünüm">
          {VIEWS.map((v) => (
            <button
              key={v.value}
              type="button"
              aria-pressed={p.view === v.value}
              className={`rounded border px-2 py-1 ${
                p.view === v.value ? "border-rgb text-rgb" : "border-line text-muted"
              }`}
              onClick={() => p.onView(v.value)}
            >
              {v.label}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
