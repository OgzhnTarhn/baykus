// Anlık görüntü (Faz 4 §5.13): her bölge olayında ve alarm başlangıcında iz kutusu çevresinde
// 128 × 128 kırpma; bellekte tutulur, en çok 500 tanesi saklanır; dışa aktarmaya girmez.
import type { Box } from "@/lib/client";
import type { Pixels } from "@/lib/faults";

export const SNAPSHOT_SIZE = 128;
export const SNAPSHOT_MAX = 500;

const clamp = (v: number, lo: number, hi: number) =>
  v < lo ? lo : v > hi ? hi : v;

// Kutunun merkezine oturan size×size pencere; görüntü dışına taşarsa içeri kaydırılır, eksen
// görüntüden küçükse o eksende görüntü boyu kullanılır. Dönen: [x0, y0, w, h].
export function cropWindow(
  box: Box,
  width: number,
  height: number,
  size: number = SNAPSHOT_SIZE,
): [number, number, number, number] {
  const w = Math.min(size, width);
  const h = Math.min(size, height);
  const cx = (box.x1 + box.x2) / 2;
  const cy = (box.y1 + box.y2) / 2;
  const x0 = clamp(Math.round(cx - w / 2), 0, width - w);
  const y0 = clamp(Math.round(cy - h / 2), 0, height - h);
  return [x0, y0, w, h];
}

// Görüntüden (x0, y0) köşeli w×h bölgeyi kopyalar; sınırlar görüntüye kırpılır.
export function crop(
  p: Pixels,
  x0: number,
  y0: number,
  w: number,
  h: number,
): Pixels {
  const x1 = clamp(x0, 0, p.width);
  const y1 = clamp(y0, 0, p.height);
  const x2 = clamp(x0 + w, x1, p.width);
  const y2 = clamp(y0 + h, y1, p.height);
  const cw = x2 - x1;
  const ch = y2 - y1;
  const data = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    const src = ((y1 + y) * p.width + x1) * 4;
    data.set(p.data.subarray(src, src + cw * 4), y * cw * 4);
  }
  return { width: cw, height: ch, data };
}

export type Snapshot = { rgb: string; ir: string };

// Olay anı kırpmaları: RGB ve IR aynı pencereyle kırpılır, kodlayıcıyla metne çevrilip bellekte
// tutulur. Kapasite aşılınca en eski anı düşer.
export class Snapshots {
  private readonly items = new Map<number, Snapshot>();

  constructor(
    private readonly encode: (p: Pixels) => string,
    private readonly max: number = SNAPSHOT_MAX,
  ) {}

  capture(id: number, rgb: Pixels, ir: Pixels, box: Box): void {
    const [x0, y0, w, h] = cropWindow(box, rgb.width, rgb.height);
    this.items.delete(id);
    while (this.items.size >= this.max) {
      const oldest = this.items.keys().next().value;
      if (oldest === undefined) break;
      this.items.delete(oldest);
    }
    this.items.set(id, {
      rgb: this.encode(crop(rgb, x0, y0, w, h)),
      ir: this.encode(crop(ir, x0, y0, w, h)),
    });
  }

  get(id: number): Snapshot | null {
    return this.items.get(id) ?? null;
  }

  size(): number {
    return this.items.size;
  }

  clear(): void {
    this.items.clear();
  }
}
