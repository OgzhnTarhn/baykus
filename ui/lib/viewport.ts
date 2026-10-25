// Harita görünümü (Faz 4 §5.4): kaydır, imleç çevresinde yakınlaştır, sığdır, kamerayı izle;
// kullanıcı kaydırınca izleme kapanır. Dönüşüm: ekran = ölçek * harita + kaydırma.
import type { Affine } from "@/lib/client";

export const ZOOM_MIN = 0.25;
export const ZOOM_MAX = 8;

export type Point = { x: number; y: number };
export type Bounds = { x: number; y: number; w: number; h: number };

function clampScale(s: number): number {
  return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, s));
}

export class Viewport {
  width: number;
  height: number;
  scale = 1;
  offsetX = 0;
  offsetY = 0;
  following = true;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
  }

  // Yüzde olarak ölçek (1× = 100).
  percent(): number {
    return Math.round(this.scale * 100);
  }

  // [a, b, tx, c, d, ty]
  matrix(): Affine {
    return [this.scale, 0, this.offsetX, 0, this.scale, this.offsetY];
  }

  toScreen(p: Point): Point {
    return { x: this.scale * p.x + this.offsetX, y: this.scale * p.y + this.offsetY };
  }

  toMap(p: Point): Point {
    return { x: (p.x - this.offsetX) / this.scale, y: (p.y - this.offsetY) / this.scale };
  }

  // İmlecin altındaki harita noktası yerinde kalacak şekilde yakınlaştırır.
  zoomAt(x: number, y: number, factor: number): void {
    if (!(factor > 0)) return;
    const anchor = this.toMap({ x, y });
    this.scale = clampScale(this.scale * factor);
    this.offsetX = x - this.scale * anchor.x;
    this.offsetY = y - this.scale * anchor.y;
  }

  // Ekran pikseliyle kaydırır; kullanıcı kaydırınca izleme kapanır.
  pan(dx: number, dy: number): void {
    this.offsetX += dx;
    this.offsetY += dy;
    this.following = false;
  }

  // Sınırları dolgulu görünüme sığdırır, en-boy oranını korur ve ortalar.
  fit(bounds: Bounds, width: number, height: number, pad: number): void {
    this.width = width;
    this.height = height;
    if (!(bounds.w > 0) || !(bounds.h > 0)) return;
    const s = clampScale(Math.min((width - 2 * pad) / bounds.w, (height - 2 * pad) / bounds.h));
    this.scale = s;
    this.offsetX = width / 2 - s * (bounds.x + bounds.w / 2);
    this.offsetY = height / 2 - s * (bounds.y + bounds.h / 2);
  }

  // Kamerayı izler: ayak izinin ağırlık merkezini görünümün ortasına getirir, ölçeği korur.
  follow(footprint: [number, number][] | null): void {
    if (!footprint || footprint.length === 0 || !this.following) return;
    let cx = 0;
    let cy = 0;
    for (const [x, y] of footprint) {
      cx += x;
      cy += y;
    }
    cx /= footprint.length;
    cy /= footprint.length;
    this.offsetX = this.width / 2 - this.scale * cx;
    this.offsetY = this.height / 2 - this.scale * cy;
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }
}
