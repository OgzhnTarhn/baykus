// Olay anı görüntü kırpmaları (Faz 4 tasarımı §5.13): kutu merkezli size×size pencere, kenarda
// görüntü içinde kalacak biçimde kaydırılır; RGB ve IR aynı pencereyle kırpılır, en fazla `max`
// anı saklanır (en eskisi düşer).
import { describe, expect, it } from "vitest";

import type { Box } from "@/lib/client";
import type { Pixels } from "@/lib/faults";
import { Snapshots, cropWindow } from "@/lib/snapshots";

// Her pikselin R kanalı x, G kanalı y: kırpmanın nereye oturduğu ilk pikselden okunur.
function pattern(width: number, height: number): Pixels {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      data[o] = x;
      data[o + 1] = y;
      data[o + 2] = 0;
      data[o + 3] = 255;
    }
  }
  return { width, height, data };
}

// IR deseni: R kanalı 255 − x (RGB'den ayırt edilebilsin diye).
function patternIr(width: number, height: number): Pixels {
  const img = pattern(width, height);
  for (let i = 0; i < img.data.length; i += 4) img.data[i] = 255 - img.data[i];
  return img;
}

// Kırpmanın boyutu ve sol üst pikseli: "128x128:86,86".
const encode = (p: Pixels) => `${p.width}x${p.height}:${p.data[0]},${p.data[1]}`;

const box = (x1: number, y1: number, x2: number, y2: number): Box => ({ x1, y1, x2, y2 });

describe("snapshots", () => {
  it("snapshot cap drops oldest", () => {
    const snaps = new Snapshots(encode, 2);
    const img = pattern(200, 200);
    const b = box(90, 90, 110, 110);

    snaps.capture(1, img, img, b);
    snaps.capture(2, img, img, b);
    expect(snaps.size()).toBe(2);

    snaps.capture(3, img, img, b);
    expect(snaps.size()).toBe(2);
    expect(snaps.get(1)).toBeNull();
    expect(snaps.get(2)).not.toBeNull();
    expect(snaps.get(3)).not.toBeNull();

    snaps.clear();
    expect(snaps.size()).toBe(0);
    expect(snaps.get(2)).toBeNull();
    expect(snaps.get(3)).toBeNull();
  });

  it("snapshot crop shifts at image edge", () => {
    // Sağ/alt kenardaki kutu: pencere görüntü içinde kalacak biçimde sola/yukarı kayar.
    expect(cropWindow(box(180, 180, 200, 200), 200, 200)).toEqual([72, 72, 128, 128]);
    // Sol/üst kenardaki kutu: pencere 0'a yapışır.
    expect(cropWindow(box(0, 0, 20, 20), 200, 200)).toEqual([0, 0, 128, 128]);
    // Eksen görüntüden büyükse o eksende 0..boyut.
    expect(cropWindow(box(20, 140, 30, 160), 50, 300)).toEqual([0, 86, 50, 128]);

    const snaps = new Snapshots(encode);
    const img = pattern(200, 200);
    snaps.capture(7, img, img, box(180, 180, 200, 200));
    expect(snaps.get(7)).toEqual({ rgb: "128x128:72,72", ir: "128x128:72,72" });
  });

  it("snapshot crop is centered on the box", () => {
    // Kutu merkezi (150, 150), pencere 128 → sol üst (86, 86).
    expect(cropWindow(box(140, 140, 160, 160), 300, 300)).toEqual([86, 86, 128, 128]);
    // size verilirse pencere o boyutta ve yine merkezli.
    expect(cropWindow(box(140, 140, 160, 160), 300, 300, 64)).toEqual([118, 118, 64, 64]);

    const snaps = new Snapshots(encode);
    const rgb = pattern(300, 300);
    const ir = patternIr(300, 300);
    snaps.capture(11, rgb, ir, box(140, 140, 160, 160));
    // Aynı pencere iki kanala da uygulanır; IR'in R kanalı 255 − x.
    expect(snaps.get(11)).toEqual({ rgb: "128x128:86,86", ir: "128x128:169,86" });
  });
});
