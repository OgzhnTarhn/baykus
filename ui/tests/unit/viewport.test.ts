import { describe, expect, it } from "vitest";

import { Viewport } from "@/lib/viewport";

describe("map viewport (Faz 4)", () => {
  it("zoom is clamped and keeps the cursor point fixed", () => {
    const v = new Viewport(800, 600);
    // imlecin altındaki harita noktası yakınlaştırma boyunca yerinde kalır
    const cursor = { x: 300, y: 200 };
    const before = v.toMap(cursor);

    v.zoomAt(cursor.x, cursor.y, 2);
    expect(v.percent()).toBe(200);
    const after = v.toMap(cursor);
    expect(after.x).toBeCloseTo(before.x, 9);
    expect(after.y).toBeCloseTo(before.y, 9);

    // üst sınır 8×
    v.zoomAt(cursor.x, cursor.y, 100);
    expect(v.percent()).toBe(800);
    const atMax = v.toMap(cursor);
    expect(atMax.x).toBeCloseTo(before.x, 9);
    expect(atMax.y).toBeCloseTo(before.y, 9);

    // alt sınır 0,25×
    v.zoomAt(cursor.x, cursor.y, 0.0001);
    expect(v.percent()).toBe(25);
    const atMin = v.toMap(cursor);
    expect(atMin.x).toBeCloseTo(before.x, 9);
    expect(atMin.y).toBeCloseTo(before.y, 9);
  });

  it("pan disables follow", () => {
    const v = new Viewport(800, 600);
    expect(v.following).toBe(true);

    const p = { x: 100, y: 100 };
    const s0 = v.toScreen(p);
    v.pan(10, -5);
    const s1 = v.toScreen(p);
    // kaydırma ekran pikseliyle: harita noktası dx, dy kadar kayar
    expect(s1.x - s0.x).toBeCloseTo(10, 9);
    expect(s1.y - s0.y).toBeCloseTo(-5, 9);
    expect(v.following).toBe(false);

    // kullanıcı takibi yeniden açabilir
    v.following = true;
    expect(v.following).toBe(true);
  });

  it("fit keeps aspect and pads", () => {
    const v = new Viewport(400, 400);
    const bounds = { x: 0, y: 0, w: 100, h: 50 };
    v.fit(bounds, 400, 400, 8);

    // (400 - 2*8) x (400 - 2*8) kutusuna sığar: ölçek = min(384/100, 384/50)
    expect(v.percent()).toBe(384);
    const [a, b, , d, e] = v.matrix();
    expect(b).toBe(0);
    expect(d).toBe(0);
    // en-boy oranı korunur: x ve y ölçeği aynı
    expect(a).toBeCloseTo(e, 9);

    // sınırlar dolgulu kutunun içinde ve ortada
    const tl = v.toScreen({ x: bounds.x, y: bounds.y });
    const br = v.toScreen({ x: bounds.x + bounds.w, y: bounds.y + bounds.h });
    expect(tl.x).toBeCloseTo(8, 9);
    expect(tl.y).toBeCloseTo(104, 9);
    expect(br.x).toBeCloseTo(392, 9);
    expect(br.y).toBeCloseTo(296, 9);
    expect((tl.x + br.x) / 2).toBeCloseTo(200, 9);
    expect((tl.y + br.y) / 2).toBeCloseTo(200, 9);

    // ölçek [0,25, 8] aralığına kırpılır
    v.fit({ x: 0, y: 0, w: 100000, h: 100000 }, 400, 400, 8);
    expect(v.percent()).toBe(25);
    v.fit({ x: 0, y: 0, w: 0.001, h: 0.001 }, 400, 400, 8);
    expect(v.percent()).toBe(800);
  });

  it("follow centers the footprint", () => {
    const v = new Viewport(800, 600);
    const footprint: [number, number][] = [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
    ];
    v.follow(footprint);
    // ağırlık merkezi (5, 5) görünümün ortasında
    const center = v.toScreen({ x: 5, y: 5 });
    expect(center.x).toBeCloseTo(400, 9);
    expect(center.y).toBeCloseTo(300, 9);
    // takip ölçeği değiştirmez
    expect(v.percent()).toBe(100);

    // null hiçbir şey yapmaz
    const before = v.matrix();
    v.follow(null);
    expect(v.matrix()).toEqual(before);

    // takip kapalıyken de hiçbir şey yapmaz
    v.pan(10, 10);
    const panned = v.matrix();
    v.follow([
      [100, 100],
      [200, 200],
    ]);
    expect(v.matrix()).toEqual(panned);

    // following = true takibi yeniden açar
    v.following = true;
    v.follow([
      [100, 100],
      [200, 200],
    ]);
    const recentered = v.toScreen({ x: 150, y: 150 });
    expect(recentered.x).toBeCloseTo(400, 9);
    expect(recentered.y).toBeCloseTo(300, 9);

    // yeniden boyutlandırma yeni merkezi kullanır
    v.resize(1000, 400);
    v.follow(footprint);
    const resized = v.toScreen({ x: 5, y: 5 });
    expect(resized.x).toBeCloseTo(500, 9);
    expect(resized.y).toBeCloseTo(200, 9);
  });

  it("toMap inverts toScreen", () => {
    const v = new Viewport(640, 480);
    // başlangıç: ölçek 1, kaydırma 0
    expect(v.percent()).toBe(100);
    expect(v.toScreen({ x: 10, y: 20 })).toEqual({ x: 10, y: 20 });

    v.zoomAt(100, 50, 3);
    v.pan(-20, 15);

    const p = { x: 123.5, y: -42.25 };
    const back = v.toMap(v.toScreen(p));
    expect(back.x).toBeCloseTo(p.x, 9);
    expect(back.y).toBeCloseTo(p.y, 9);

    // matris [s, 0, ox, 0, s, oy] ve ekran = ölçek * harita + kaydırma
    const [a, b, c, d, e, f] = v.matrix();
    expect(b).toBe(0);
    expect(d).toBe(0);
    expect(a).toBeCloseTo(e, 9);
    const s = v.toScreen(p);
    expect(s.x).toBeCloseTo(a * p.x + c, 9);
    expect(s.y).toBeCloseTo(e * p.y + f, 9);
  });
});
