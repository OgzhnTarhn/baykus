import { describe, expect, it } from "vitest";

import type { Affine, Ego, Track } from "@/lib/client";
import { Mosaic, hitTest } from "@/lib/mosaic";

// Küçük yardımcılar: Ego (ok, map_gen, provisional, frame_to_map, merge) ve Track (track_id, map_xy).
function ego(over: Partial<Ego> = {}): Ego {
  return {
    ok: true,
    source: "rgb",
    map_gen: 1,
    provisional: false,
    frame_to_map: [1, 0, 0, 0, 1, 0],
    inliers: 40,
    inlier_ratio: 0.9,
    keyframe: false,
    relinked: false,
    relocalized: false,
    merge: null,
    ...over,
  };
}

function track(track_id: number, map_xy: [number, number] | null): Track {
  return {
    track_id,
    cls_id: 0,
    box: { x1: 0, y1: 0, x2: 16, y2: 16 },
    score_raw: 0.9,
    score: 0.9,
    motion: "moving",
    speed_rel: null,
    map_xy,
  };
}

// DOM yok: küçük resim düz nesne, yalnız karşılaştırmada kullanılıyor.
function thumb(tag: string): CanvasImageSource {
  return { tag } as unknown as CanvasImageSource;
}

describe("mozaik katmanları", () => {
  it("adds a tile every fifth ok frame", () => {
    const m = new Mosaic([64, 48], 4);
    const t = thumb("t");
    const at = (f: number) => ego({ map_gen: 1, frame_to_map: [1, 0, f * 10, 0, 1, 0] });

    // 1. iyi kare karo ister ve depolar
    expect(m.needsThumb(at(0))).toBe(true);
    m.push(0, at(0), [], t);
    expect(m.layer(1).tiles).toHaveLength(1);
    expect(m.needsThumb(at(1))).toBe(false);

    // 2..5. iyi kare karo değil
    for (let f = 1; f < 5; f++) m.push(f, at(f), [], t);
    expect(m.layer(1).tiles).toHaveLength(1);

    // 6. iyi kare yeniden karo ister
    expect(m.needsThumb(at(5))).toBe(true);
    m.push(5, at(5), [], t);
    for (let f = 6; f < 10; f++) m.push(f, at(f), [], t);
    expect(m.layer(1).tiles).toHaveLength(2);

    // 11. iyi kare karo karesi ama küçük resim yok -> depolanmaz
    m.push(10, at(10), [], null);
    expect(m.layer(1).tiles).toHaveLength(2);
    m.push(11, at(11), [], t);
    expect(m.layer(1).tiles).toHaveLength(2);

    // m = frame_to_map ∘ thumbScale: [a*s, b*s, tx, c*s, d*s, ty]
    expect(m.layer(1).tiles.map((x: { m: Affine }) => x.m)).toEqual([
      [4, 0, 0, 0, 4, 0],
      [4, 0, 50, 0, 4, 0],
    ]);
    expect(m.layer(1).tiles[0].thumb).toBe(t);
    expect(m.currentGen()).toBe(1);

    // iyi olmayan ego ve ego yok: sıradaki itme karo depolamaz
    const fresh = new Mosaic([64, 48], 4);
    expect(fresh.needsThumb(null)).toBe(false);
    expect(fresh.needsThumb(ego({ ok: false }))).toBe(false);
    expect(fresh.needsThumb(ego({ map_gen: 1 }))).toBe(true);
  });

  it("thins tiles evenly at the layer cap", () => {
    const m = new Mosaic([64, 48], 4);
    const t = thumb("t");
    for (let f = 0; f < 2000; f++) {
      m.push(f, ego({ map_gen: 1, frame_to_map: [1, 0, f, 0, 1, 0] }), [], t);
    }
    // 2000 iyi kare -> 400 karo (1., 6., ..., 1996. iyi kare)
    expect(m.layer(1).tiles).toHaveLength(400);

    // 401. karo: çift sıralar (0, 2, 4, ...) kalır, sonra yenisi eklenir
    m.push(2000, ego({ map_gen: 1, frame_to_map: [1, 0, 2000, 0, 1, 0] }), [], t);
    const tiles = m.layer(1).tiles;
    expect(tiles).toHaveLength(201);
    // karo k'nın tx değeri 5k; tutulanlar 0, 2, ... -> 0, 10, ..., 1998
    expect(tiles.slice(0, 200).map((x: { m: Affine }) => x.m[2])).toEqual(
      Array.from({ length: 200 }, (_, j) => 10 * j),
    );
    expect(tiles[200].m).toEqual([4, 0, 2000, 0, 4, 0]);
  });

  it("skips tiles and breaks trails when ego not ok", () => {
    const m = new Mosaic([64, 48], 4);
    const t = thumb("t");
    for (let f = 0; f < 5; f++) m.push(f, ego({ map_gen: 1 }), [track(7, [f, f])], t);

    // ego iyi değil: karo yok, iz kesmesi
    m.push(5, ego({ map_gen: 1, ok: false }), [track(7, [5, 5])], t);
    expect(m.layer(1).footprint).toBeNull();
    // ego yok: karo yok, ikinci kesme işareti art arda gelmez
    m.push(6, null, [track(7, [6, 6])], t);
    expect(m.layer(1).footprint).toBeNull();

    // 6. iyi kare -> karo; map_xy'si olmayan takip hiç kaydedilmez
    m.push(7, ego({ map_gen: 1 }), [track(7, [7, 7]), track(8, null)], t);
    const l = m.layer(1);
    expect(l.tiles).toHaveLength(2);
    const pts = l.trails.get(7) ?? [];
    expect(pts).toHaveLength(7);
    expect(pts.slice(0, 5)).toEqual([
      [0, 0],
      [1, 1],
      [2, 2],
      [3, 3],
      [4, 4],
    ]);
    expect(pts[5]).toEqual([NaN, NaN]);
    expect(pts[6]).toEqual([7, 7]);
    expect(l.trails.has(8)).toBe(false);
    expect(l.footprint).toEqual([
      [0, 0],
      [64, 0],
      [64, 48],
      [0, 48],
    ]);
    // sınırlar sonlu noktalardır: kesme işaretçisi (NaN) dışarıda kalır
    expect(m.bounds(1)).toEqual({ x: 0, y: 0, w: 64, h: 48 });

    // iz başına en fazla 150 nokta: en eskisi atılır
    const m2 = new Mosaic([64, 48], 4);
    for (let f = 0; f < 155; f++) m2.push(f, ego({ map_gen: 1 }), [track(3, [f, 0])], t);
    const kept = m2.layer(1).trails.get(3) ?? [];
    expect(kept).toHaveLength(150);
    expect(kept[0]).toEqual([5, 0]);
    expect(kept[149]).toEqual([154, 0]);
  });

  it("merge moves tiles and trails into the target layer", () => {
    const m = new Mosaic([64, 48], 4);
    const t = thumb("t");
    const T: Affine = [2, 0, 100, 0, 2, 200];

    // 1. kuşak: bir karo + bir kesme işareti
    m.push(0, ego({ map_gen: 1, frame_to_map: [1, 0, 0, 0, 1, 0] }), [track(1, [10, 20])], t);
    m.push(1, ego({ map_gen: 1, ok: false }), [track(1, [30, 40])], t);
    // 2. kuşak
    m.push(2, ego({ map_gen: 2, frame_to_map: [1, 0, 5, 0, 1, 6] }), [track(9, [1, 1])], t);

    // birleşme: önce 1. kuşak taşınır, sonra geçerli kare işlenir
    m.push(
      3,
      ego({
        map_gen: 2,
        frame_to_map: [1, 0, 7, 0, 1, 8],
        merge: { from: 1, to: 2, transform: T },
      }),
      [track(9, [2, 2])],
      t,
    );

    const gone = m.layer(1);
    expect(gone.tiles).toEqual([]);
    expect(gone.trails.size).toBe(0);
    expect(gone.footprint).toBeNull();
    expect(m.bounds(1)).toBeNull();

    const l2 = m.layer(2);
    expect(l2.tiles).toHaveLength(1);
    // m' = T ∘ m
    expect(l2.tiles[0].m).toEqual([8, 0, 100, 0, 8, 200]);
    // iz noktaları T(p), kesme işareti olduğu gibi kalır
    expect(l2.trails.get(1)).toEqual([
      [120, 240],
      [NaN, NaN],
    ]);
    expect(l2.trails.get(9)).toEqual([
      [1, 1],
      [2, 2],
    ]);
    expect(m.currentGen()).toBe(2);
  });

  it("provisional generation is flagged", () => {
    const m = new Mosaic([64, 48], 4);
    const t = thumb("t");
    m.push(0, ego({ map_gen: 1, provisional: true }), [], t);
    expect(m.layer(1).provisional).toBe(true);

    // son ego belirler
    m.push(1, ego({ map_gen: 1, provisional: false }), [], t);
    expect(m.layer(1).provisional).toBe(false);
    m.push(2, ego({ map_gen: 1, provisional: true }), [], t);
    expect(m.layer(1).provisional).toBe(true);

    // iyi olmayan itme de bayrağı günceller
    m.push(3, ego({ map_gen: 1, ok: false, provisional: false }), [], t);
    expect(m.layer(1).provisional).toBe(false);

    // bilinmeyen kuşak: boş katman, bayrak yok
    const empty = m.layer(99);
    expect(empty.provisional).toBe(false);
    expect(empty.tiles).toEqual([]);
    expect(empty.footprint).toBeNull();
  });

  it("footprint of identity is the frame rectangle", () => {
    const m = new Mosaic([64, 48], 4);
    m.push(0, ego({ map_gen: 1, frame_to_map: [1, 0, 0, 0, 1, 0] }), [], null);
    expect(m.layer(1).footprint).toEqual([
      [0, 0],
      [64, 0],
      [64, 48],
      [0, 48],
    ]);

    // öteleme + ölçek: köşeler son iyi frame_to_map'ten geçer
    m.push(1, ego({ map_gen: 1, frame_to_map: [2, 0, 10, 0, 2, 20] }), [], null);
    expect(m.layer(1).footprint).toEqual([
      [10, 20],
      [138, 20],
      [138, 116],
      [10, 116],
    ]);

    // son itme iyi değilse ayak izi yok
    m.push(2, ego({ map_gen: 1, ok: false }), [], null);
    expect(m.layer(1).footprint).toBeNull();
  });

  it("hit test picks the nearest trail within radius", () => {
    const trails = new Map<number, [number, number][]>([
      [1, [[0, 0], [10, 0]]],
      [2, [[100, 100], [NaN, NaN]]],
    ]);
    const tracks = [track(1, [20, 0]), track(2, [200, 200]), track(3, [5, 5])];

    // iz noktasına yakın: iz 1
    expect(hitTest([11, 1], tracks, trails, 5)).toBe(1);
    // geçerli takip konumu daha yakın: iz 3
    expect(hitTest([5, 6], tracks, trails, 5)).toBe(3);
    // yarıçap (harita birimi) dışında kimse yok
    expect(hitTest([50, 50], tracks, trails, 5)).toBeNull();
    // NaN kesme işareti yarışmayı bozmaz: sonlu nokta yine bulunur
    expect(hitTest([101, 101], tracks, trails, 5)).toBe(2);
    // yarıçap sınırı kapsar
    expect(hitTest([15, 0], tracks, trails, 5)).toBe(1);
  });

  it("mosaic ignores response from previous session", () => {
    const m = new Mosaic([64, 48], 4);
    const t = thumb("t");
    m.push(0, ego({ map_gen: 1 }), [track(1, [0, 0])], t);
    m.reset(2);
    expect(m.epoch()).toBe(2);
    expect(m.currentGen()).toBeNull();
    expect(m.bounds(1)).toBeNull();

    // eski dönem yanıtı: hiçbir şey değişmez
    m.push(1, ego({ map_gen: 1 }), [track(1, [1, 1])], t, 1);
    expect(m.currentGen()).toBeNull();
    expect(m.layer(1).tiles).toEqual([]);
    expect(m.layer(1).trails.size).toBe(0);

    // geçerli dönem uygulanır
    m.push(1, ego({ map_gen: 1 }), [track(1, [1, 1])], t, 2);
    expect(m.currentGen()).toBe(1);
    expect(m.layer(1).tiles).toHaveLength(1);
    expect(m.layer(1).trails.get(1)).toEqual([[1, 1]]);
  });
});
