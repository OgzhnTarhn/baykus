// Göreli mozaik harita (Faz 4 tasarımı §5.4); gerçek konum değildir, ölçek / kuzey / metre yok.
// Küçük resimler opak CanvasImageSource değerleridir; bu modül DOM'a dokunmaz.
import type { Affine, Ego, Track } from "@/lib/client";

export const TILE_EVERY = 5;
export const LAYER_MAX_TILES = 400;
export const TRAIL_MAX = 150;
export const MAP_CAPTION =
  "göreli harita (kamera hareketinden; ilk kareye göre; kanal değişince yeniden konumlanır)";

export type Rect = { x: number; y: number; w: number; h: number };

// Nesil katmanı: küçük resimler (karo), iz kuyrukları ve kameranın görüş alanı.
export type Layer = {
  gen: number;
  provisional: boolean;
  tiles: { m: Affine; thumb: CanvasImageSource }[];
  // kuyruk son 150 kare (tasarım §5.4); o süre görünmeyen izin kuyruğu düşer.
  trails: Map<number, [number, number][]>;
  seen: Map<number, number>;
  footprint: [number, number][] | null;
};

// a ∘ b: önce b, sonra a.
export function compose(a: Affine, b: Affine): Affine {
  return [
    a[0] * b[0] + a[1] * b[3],
    a[0] * b[1] + a[1] * b[4],
    a[0] * b[2] + a[1] * b[5] + a[2],
    a[3] * b[0] + a[4] * b[3],
    a[3] * b[1] + a[4] * b[4],
    a[3] * b[2] + a[4] * b[5] + a[5],
  ];
}

export function apply(m: Affine, p: [number, number]): [number, number] {
  return [m[0] * p[0] + m[1] * p[1] + m[2], m[3] * p[0] + m[4] * p[1] + m[5]];
}

// Tuval sırası farklıdır: setTransform(a, b, c, d, e, f) x' = a·x + c·y + e, y' = b·x + d·y + f.
export type CanvasMatrix = [number, number, number, number, number, number];

export function toCanvas(m: Affine): CanvasMatrix {
  return [m[0], m[3], m[1], m[4], m[2], m[5]];
}

export function fromCanvas(m: {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
}): Affine {
  return [m.a, m.c, m.e, m.b, m.d, m.f];
}

function emptyLayer(gen: number): Layer {
  return {
    gen,
    provisional: false,
    tiles: [],
    trails: new Map(),
    seen: new Map(),
    footprint: null,
  };
}

// Dolunca eşit aralıkla seyrelt: çift sıralar kalır, harita kapsamı korunur.
function thin(tiles: Layer["tiles"]): Layer["tiles"] {
  const kept: Layer["tiles"] = [];
  for (let i = 0; i < tiles.length; i += 2) kept.push(tiles[i]);
  return kept;
}

export class Mosaic {
  private frame: [number, number];
  private thumbScale: number;
  private layers = new Map<number, Layer>();
  private gen: number | null = null;
  private tileCount = 0;
  private broken = new Set<number>();
  private epochNo = 0;
  private counter = 0;

  constructor(frame: [number, number], thumbScale: number) {
    this.frame = frame;
    this.thumbScale = thumbScale;
  }

  // Yanıt damgası (oturum/dönem): geçerli değilse itme yok sayılır.
  reset(epoch: number): void {
    this.layers.clear();
    this.gen = null;
    this.tileCount = 0;
    this.broken.clear();
    this.epochNo = epoch;
    this.counter = 0;
  }

  epoch(): number {
    return this.epochNo;
  }

  currentGen(): number | null {
    return this.gen;
  }

  layer(gen: number): Layer {
    return this.layers.get(gen) ?? emptyLayer(gen);
  }

  // Sıradaki iyi kare karo ister mi? (her TILE_EVERY. iyi kare)
  needsThumb(ego: Ego | null): boolean {
    return ego !== null && ego.ok && this.tileCount % TILE_EVERY === 0;
  }

  push(
    frame: number,
    ego: Ego | null,
    tracks: Track[],
    thumb: CanvasImageSource | null,
    epoch: number = this.epochNo,
  ): void {
    if (epoch !== this.epochNo) return; // önceki dönemin geç gelen yanıtı
    this.counter += 1;
    if (ego === null) {
      this.breakTrails(tracks);
      this.prune(this.gen === null ? undefined : this.layers.get(this.gen));
      return;
    }
    const l = this.layerOf(ego);
    l.provisional = ego.provisional;
    if (ego.merge !== null) this.merge(ego.merge.from, ego.merge.to, ego.merge.transform);
    if (!ego.ok || ego.frame_to_map === null) {
      l.footprint = null;
      this.breakTrails(tracks);
      this.prune(l);
      return;
    }
    const m = ego.frame_to_map;
    if (this.tileCount % TILE_EVERY === 0 && thumb !== null) {
      const s = this.thumbScale;
      l.tiles.push({ m: [m[0] * s, m[1] * s, m[2], m[3] * s, m[4] * s, m[5]], thumb });
      if (l.tiles.length > LAYER_MAX_TILES) l.tiles = thin(l.tiles);
    }
    this.tileCount += 1;
    l.footprint = [
      apply(m, [0, 0]),
      apply(m, [this.frame[0], 0]),
      apply(m, [this.frame[0], this.frame[1]]),
      apply(m, [0, this.frame[1]]),
    ];
    for (const t of tracks) {
      if (t.map_xy === null) continue;
      const pts = l.trails.get(t.track_id) ?? [];
      pts.push([t.map_xy[0], t.map_xy[1]]);
      if (pts.length > TRAIL_MAX) pts.splice(0, pts.length - TRAIL_MAX);
      l.trails.set(t.track_id, pts);
      l.seen.set(t.track_id, this.counter);
      this.broken.delete(t.track_id);
    }
    this.prune(l);
  }

  // TRAIL_MAX itmeden uzun süre görülmeyen izlerin kuyruğu düşer.
  private prune(l: Layer | undefined): void {
    if (l === undefined) return;
    for (const [id, at] of l.seen) {
      if (at < this.counter - TRAIL_MAX) {
        l.trails.delete(id);
        l.seen.delete(id);
      }
    }
  }

  bounds(gen: number): Rect | null {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    const add = (p: [number, number]): void => {
      if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) return;
      minX = Math.min(minX, p[0]);
      minY = Math.min(minY, p[1]);
      maxX = Math.max(maxX, p[0]);
      maxY = Math.max(maxY, p[1]);
    };
    const l = this.layers.get(gen);
    if (l === undefined) return null;
    // Karo köşeleri küçük resim dikdörtgenidir (tile.m ölçeği zaten içerir).
    const tw = this.frame[0] / this.thumbScale;
    const th = this.frame[1] / this.thumbScale;
    for (const tile of l.tiles) {
      add(apply(tile.m, [0, 0]));
      add(apply(tile.m, [tw, th]));
    }
    for (const pts of l.trails.values()) for (const p of pts) add(p);
    if (l.footprint !== null) for (const p of l.footprint) add(p);
    return Number.isFinite(minX) && Number.isFinite(minY)
      ? { x: minX, y: minY, w: maxX - minX, h: maxY - minY }
      : null;
  }

  // Geçerli katmanı bulur / kurar; son ego kazanır.
  private layerOf(ego: Ego): Layer {
    if (this.gen !== ego.map_gen) {
      // İyi kare sayacı nesiller arası sürer; yalnız reset() sıfırlar.
      this.gen = ego.map_gen;
      this.broken.clear();
    }
    let l = this.layers.get(ego.map_gen);
    if (l === undefined) {
      l = emptyLayer(ego.map_gen);
      this.layers.set(ego.map_gen, l);
    }
    return l;
  }

  // Birleşme: kaynağın karo, iz ve görüş alanı dönüşümle hedef katmana taşınır.
  private merge(from: number, to: number, transform: Affine): void {
    const src = this.layers.get(from);
    if (src === undefined) return;
    const dst = this.layers.get(to) ?? emptyLayer(to);
    for (const tile of src.tiles) {
      dst.tiles.push({ m: compose(transform, tile.m), thumb: tile.thumb });
    }
    if (dst.tiles.length > LAYER_MAX_TILES) dst.tiles = thin(dst.tiles);
    for (const [id, pts] of src.trails) {
      const moved: [number, number][] = pts.map((p) =>
        Number.isFinite(p[0]) && Number.isFinite(p[1]) ? apply(transform, p) : [NaN, NaN],
      );
      const old = dst.trails.get(id) ?? [];
      dst.trails.set(id, old.concat(moved));
    }
    for (const [id, at] of src.seen) {
      dst.seen.set(id, Math.max(dst.seen.get(id) ?? -Infinity, at));
    }
    if (src.footprint !== null) {
      dst.footprint = src.footprint.map((p) => apply(transform, p));
    }
    this.layers.set(to, dst);
    this.layers.delete(from);
  }

  // ego iyi değilken ya da ego yokken kuyruklar kesilir; arka arkaya iki kesme işareti yoktur.
  private breakTrails(tracks: Track[]): void {
    const l = this.gen === null ? undefined : this.layers.get(this.gen);
    if (l === undefined) return;
    for (const t of tracks) {
      if (t.map_xy === null || this.broken.has(t.track_id)) continue;
      const pts = l.trails.get(t.track_id);
      if (pts === undefined || pts.length === 0) continue;
      pts.push([NaN, NaN]);
      if (pts.length > TRAIL_MAX) pts.splice(0, pts.length - TRAIL_MAX);
      this.broken.add(t.track_id);
    }
  }
}

// İmleç altındaki iz: en yakın sonlu kuyruk noktası ya da geçerli takip konumu (harita birimi).
export function hitTest(
  p: [number, number],
  tracks: Track[],
  trails: Map<number, [number, number][]>,
  radius: number,
): number | null {
  let best: number | null = null;
  let bestD = radius;
  const offer = (id: number, q: [number, number]): void => {
    if (!Number.isFinite(q[0]) || !Number.isFinite(q[1])) return;
    const d = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (d <= bestD) {
      bestD = d;
      best = id;
    }
  };
  for (const t of tracks) if (t.map_xy !== null) offer(t.track_id, t.map_xy);
  for (const [id, pts] of trails) for (const q of pts) offer(id, q);
  return best;
}
