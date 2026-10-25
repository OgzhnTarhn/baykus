"use client";

// Göreli mozaik harita paneli (Faz 4 tasarımı §5.4): karo küçük resimleri, kameranın görüş alanı, iz
// kuyrukları ve işaretçiler; sürükle-kaydır, imleç çevresinde yakınlaştır, sığdır, kamerayı izle.
// Harita gerçek konum değildir: fiziksel ölçek, kuzey oku ve metre yoktur.
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";

import type { Affine, Motion, Track } from "@/lib/client";
import { MAP_CAPTION, apply, compose, hitTest, type Mosaic } from "@/lib/mosaic";
import { Viewport } from "@/lib/viewport";

const BG = "#0b0d10";
const TEXT = "#d8e1ea";
const MOVING_COLOR = "#d8e1ea";
const STATIONARY_COLOR = "#8a9aab";
const UNKNOWN_COLOR = "#6b7a8a";
const ZOOM_STEP = 1.1;
const HIT_PX = 12;
const FIT_PAD = 24;
const DRAG_SLOP = 2;

function trailColor(motion: Motion): string {
  if (motion === "moving") return MOVING_COLOR;
  if (motion === "stationary") return STATIONARY_COLOR;
  return UNKNOWN_COLOR;
}

// Tuvali panel arka planıyla boyar (tuval pikseli uzayı).
function fillBackground(ctx: CanvasRenderingContext2D): void {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

// NaN noktadan sonra yeni alt yol açar (kuyruk kesintisi).
function strokeTrail(
  ctx: CanvasRenderingContext2D,
  view: Affine,
  pts: [number, number][],
): void {
  ctx.beginPath();
  let pen = false;
  for (const p of pts) {
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) {
      pen = false;
      continue;
    }
    const [x, y] = apply(view, p);
    if (pen) ctx.lineTo(x, y);
    else {
      ctx.moveTo(x, y);
      pen = true;
    }
  }
  ctx.stroke();
}

// Saf çizim: karo katmanı + görüş alanı + iz kuyrukları + işaretçiler. `view` harita → ekran (CSS px)
// dönüşümüdür; çağıranın o anki dönüşümü (ör. devicePixelRatio) taban olarak korunur. Task 21 tam kapsamı
// çevrimdışı tuvale aynı fonksiyonla çizer.
export function drawMap(
  ctx: CanvasRenderingContext2D,
  mosaic: Mosaic,
  gen: number,
  view: Affine,
  opts: { tracks: Track[]; selected: number | null; motionOf: (id: number) => Motion },
): void {
  const base = ctx.getTransform();
  const baseAffine: Affine = [base.a, base.b, base.c, base.d, base.e, base.f];
  fillBackground(ctx);

  const layer = mosaic.layer(gen);
  for (const tile of layer.tiles) {
    ctx.setTransform(...compose(baseAffine, compose(view, tile.m)));
    ctx.drawImage(tile.thumb, 0, 0);
  }
  ctx.setTransform(...baseAffine);

  // Görüş alanı (kare dörtgeni): beyaz, ince.
  if (layer.footprint !== null && layer.footprint.length > 0) {
    ctx.beginPath();
    layer.footprint.forEach((p, i) => {
      const [x, y] = apply(view, p);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    ctx.stroke();
  }

  // İz kuyrukları: hareketli düz kalın, duran kesikli ince, bilinmeyen düz ince gri.
  for (const [id, pts] of layer.trails) {
    if (id === opts.selected) continue;
    const motion = opts.motionOf(id);
    ctx.strokeStyle = trailColor(motion);
    ctx.lineWidth = motion === "moving" ? 2 : 1;
    ctx.setLineDash(motion === "stationary" ? [5, 4] : []);
    strokeTrail(ctx, view, pts);
  }

  // Seçili iz en üstte: beyaz hale + kalın kuyruk.
  const selectedPts = opts.selected === null ? undefined : layer.trails.get(opts.selected);
  if (selectedPts !== undefined && opts.selected !== null) {
    const motion = opts.motionOf(opts.selected);
    ctx.setLineDash([]);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 6;
    strokeTrail(ctx, view, selectedPts);
    ctx.strokeStyle = trailColor(motion);
    ctx.lineWidth = 4;
    ctx.setLineDash(motion === "stationary" ? [5, 4] : []);
    strokeTrail(ctx, view, selectedPts);
  }
  ctx.setLineDash([]);

  // Geçerli izlerin işaretçisi ve iz numarası.
  ctx.font = "10px var(--font-geist-mono), monospace";
  ctx.textBaseline = "middle";
  for (const t of opts.tracks) {
    if (t.map_xy === null) continue;
    const [x, y] = apply(view, t.map_xy);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const sel = t.track_id === opts.selected;
    ctx.beginPath();
    ctx.arc(x, y, sel ? 4.5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = trailColor(t.motion);
    ctx.fill();
    ctx.lineWidth = sel ? 2 : 1;
    ctx.strokeStyle = sel ? "#ffffff" : BG;
    ctx.stroke();
    const label = `${t.track_id}`;
    ctx.lineWidth = 3;
    ctx.strokeStyle = BG;
    ctx.strokeText(label, x + 6, y - 6);
    ctx.fillStyle = TEXT;
    ctx.fillText(label, x + 6, y - 6);
  }
}

export function MapPanel({
  mosaic,
  version,
  tracks,
  selected,
  onSelect,
  follow,
}: {
  mosaic: Mosaic;
  version: number;
  tracks: Track[];
  selected: number | null;
  onSelect(id: number | null): void;
  follow?: number;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewRef = useRef<Viewport | null>(null);
  const dragRef = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const lastFollow = useRef<number | undefined>(follow);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [percent, setPercent] = useState(100);
  const [following, setFollowing] = useState(true);
  const [hover, setHover] = useState<{ id: number; x: number; y: number } | null>(null);

  const gen = mosaic.currentGen();
  const layer = gen === null ? null : mosaic.layer(gen);

  const motionOf = useCallback(
    (id: number): Motion => tracks.find((t) => t.track_id === id)?.motion ?? "unknown",
    [tracks],
  );

  const ensureViewport = useCallback((): Viewport | null => {
    const canvas = canvasRef.current;
    if (canvas === null) return null;
    if (viewRef.current === null) {
      viewRef.current = new Viewport(canvas.clientWidth, canvas.clientHeight);
    }
    return viewRef.current;
  }, []);

  const draw = useCallback((): void => {
    const canvas = canvasRef.current;
    const vp = ensureViewport();
    if (canvas === null || vp === null) return;
    const ctx = canvas.getContext("2d");
    if (ctx === null) return;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const g = mosaic.currentGen();
    if (g === null) {
      fillBackground(ctx);
    } else {
      drawMap(ctx, mosaic, g, vp.matrix(), { tracks, selected, motionOf });
    }
    setPercent(vp.percent());
  }, [ensureViewport, mosaic, tracks, selected, motionOf]);

  // Tuval boyutu: kapsayıcıya göre, devicePixelRatio ile.
  useEffect(() => {
    const wrap = wrapRef.current;
    if (wrap === null) return;
    const ro = new ResizeObserver(() => {
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;
      viewRef.current?.resize(w, h);
      setSize({ w, h });
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  // Her itmede (version artar) yeniden çiz; izleme açıksa görüş alanını ortada tut.
  useEffect(() => {
    void version; // yalnız yeniden çizim tetiği
    const vp = ensureViewport();
    const g = mosaic.currentGen();
    if (vp !== null && g !== null && vp.following) vp.follow(mosaic.layer(g).footprint);
    draw();
  }, [version, size, ensureViewport, draw, mosaic]);

  // "Haritada göster": sayaç değişince seçili izin son sonlu kuyruk noktasını ortala.
  useEffect(() => {
    if (follow === undefined || follow === lastFollow.current) return;
    lastFollow.current = follow;
    const vp = ensureViewport();
    const g = mosaic.currentGen();
    if (vp === null || g === null || selected === null) return;
    const pts = mosaic.layer(g).trails.get(selected) ?? [];
    let last: [number, number] | null = null;
    for (const p of pts) if (Number.isFinite(p[0]) && Number.isFinite(p[1])) last = p;
    if (last === null) return;
    vp.offsetX = vp.width / 2 - vp.scale * last[0];
    vp.offsetY = vp.height / 2 - vp.scale * last[1];
    draw();
  }, [follow, selected, ensureViewport, draw, mosaic]);

  // Tekerlek: imleç çevresinde yakınlaştır (passive olmayan dinleyici, varsayılan kaydırma engellenir).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const onWheel = (e: WheelEvent): void => {
      e.preventDefault();
      const vp = ensureViewport();
      if (vp === null) return;
      const r = canvas.getBoundingClientRect();
      const factor = e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
      vp.zoomAt(e.clientX - r.left, e.clientY - r.top, factor);
      draw();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [ensureViewport, draw]);

  const zoomCenter = (factor: number): void => {
    const vp = ensureViewport();
    if (vp === null) return;
    vp.zoomAt(vp.width / 2, vp.height / 2, factor);
    draw();
  };

  const fit = (): void => {
    const vp = ensureViewport();
    const g = mosaic.currentGen();
    if (vp === null || g === null) return;
    const b = mosaic.bounds(g);
    if (b === null) return;
    vp.fit(b, vp.width, vp.height, FIT_PAD);
    draw();
  };

  const toggleFollow = (): void => {
    const vp = ensureViewport();
    if (vp === null) return;
    vp.following = !vp.following;
    setFollowing(vp.following);
    if (vp.following) {
      const g = mosaic.currentGen();
      if (g !== null) vp.follow(mosaic.layer(g).footprint);
    }
    draw();
  };

  const pick = (x: number, y: number): number | null => {
    const vp = ensureViewport();
    const g = mosaic.currentGen();
    if (vp === null) return null;
    const p = vp.toMap({ x, y });
    const trails = g === null ? new Map<number, [number, number][]>() : mosaic.layer(g).trails;
    return hitTest([p.x, p.y], tracks, trails, HIT_PX / vp.scale);
  };

  const onPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>): void => {
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = { x: e.clientX, y: e.clientY, moved: false };
  };

  const onPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>): void => {
    const drag = dragRef.current;
    if (drag !== null) {
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > DRAG_SLOP) drag.moved = true;
      drag.x = e.clientX;
      drag.y = e.clientY;
      const vp = ensureViewport();
      if (vp !== null && (dx !== 0 || dy !== 0)) {
        vp.pan(dx, dy);
        setFollowing(vp.following);
        draw();
      }
      return;
    }
    const r = e.currentTarget.getBoundingClientRect();
    const id = pick(e.clientX - r.left, e.clientY - r.top);
    setHover(id === null ? null : { id, x: e.clientX - r.left, y: e.clientY - r.top });
  };

  const onPointerUp = (e: ReactPointerEvent<HTMLCanvasElement>): void => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    if (drag === null || drag.moved) return;
    const r = e.currentTarget.getBoundingClientRect();
    onSelect(pick(e.clientX - r.left, e.clientY - r.top));
  };

  const onPointerLeave = (): void => {
    dragRef.current = null;
    setHover(null);
  };

  return (
    <section className="flex flex-col gap-2 rounded border border-line bg-panel p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2 className="text-sm font-semibold">Harita</h2>
        <span className="text-xs text-muted">{MAP_CAPTION}</span>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <button
          type="button"
          className="rounded border border-line px-2 py-1"
          onClick={() => zoomCenter(1 / ZOOM_STEP)}
          aria-label="Uzaklaştır"
        >
          −
        </button>
        <button
          type="button"
          className="rounded border border-line px-2 py-1"
          onClick={() => zoomCenter(ZOOM_STEP)}
          aria-label="Yakınlaştır"
        >
          +
        </button>
        <button
          type="button"
          className="rounded border border-line px-2 py-1 disabled:opacity-40"
          onClick={fit}
          disabled={gen === null}
        >
          Sığdır
        </button>
        <button
          type="button"
          aria-pressed={following}
          className={`rounded border px-2 py-1 ${
            following ? "border-rgb text-rgb" : "border-line text-muted"
          }`}
          onClick={toggleFollow}
        >
          Kamerayı izle
        </button>
        <span className="num ml-auto text-xs text-muted">{percent}%</span>
      </div>
      {layer !== null && layer.provisional && (
        <span className="inline-flex w-fit items-center gap-1.5 rounded-full border border-warn px-2 py-0.5 text-xs text-warn">
          <span className="h-1.5 w-1.5 rounded-full bg-warn" aria-hidden />
          konum aranıyor…
        </span>
      )}
      {layer !== null && !layer.provisional && gen !== null && gen > 1 && (
        <span className="inline-flex w-fit items-center gap-1.5 rounded-full border border-warn px-2 py-0.5 text-xs text-warn">
          <span className="h-1.5 w-1.5 rounded-full bg-warn" aria-hidden />
          {`yeni harita (nesil ${gen}) — önceki bölgeler askıda`}
        </span>
      )}
      <div
        ref={wrapRef}
        className="relative w-full overflow-hidden rounded border border-line"
        style={{ aspectRatio: "4 / 3" }}
      >
        <canvas
          ref={canvasRef}
          className="h-full w-full cursor-grab touch-none"
          role="img"
          aria-label="Göreli harita: kameranın görüş alanı, iz kuyrukları ve araç işaretçileri"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={onPointerLeave}
        />
        {gen === null && (
          <p className="pointer-events-none absolute inset-0 flex items-center justify-center p-4 text-center text-sm text-muted">
            Harita, ego izlenen ilk karelerle oluşur.
          </p>
        )}
        {hover !== null && (
          <span
            className="num pointer-events-none absolute rounded border border-line bg-panel-2 px-1.5 py-0.5 text-xs"
            style={{ left: hover.x + 10, top: hover.y + 10 }}
          >
            {`#${hover.id}`}
          </span>
        )}
      </div>
    </section>
  );
}
