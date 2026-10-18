// Arıza simülasyonu (Faz 3 tasarımı §4.4; plan Görev 31): tanımlar baykus.robustness.corruptions
// ile aynı. Kaydırma ve karartma Python fikstürleriyle piksel piksel karşılaştırılır
// (fixtures/synthetic: darken_rgb_ir.png = özgün IR, shift_ir_rgb.png = özgün RGB).
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import {
  blackout,
  blur,
  buildFaultForm,
  darken,
  faultedChannels,
  mulberry32,
  noise,
  rintEven,
  shiftIr,
  NO_FAULTS,
  type Pixels,
} from "@/lib/faults";
import { decodePng } from "./png";

const SYN = join(__dirname, "..", "..", "..", "fixtures", "synthetic");

function px(width: number, height: number, values: number[]): Pixels {
  const data = new Uint8ClampedArray(width * height * 4);
  values.forEach((v, i) => {
    data[i * 4] = data[i * 4 + 1] = data[i * 4 + 2] = v;
    data[i * 4 + 3] = 255;
  });
  return { width, height, data };
}

function firstDiff(a: Pixels, b: Pixels): string | null {
  if (a.width !== b.width || a.height !== b.height) return "boyut farklı";
  for (let i = 0; i < a.data.length; i++) {
    if (a.data[i] !== b.data[i]) {
      const p = i >> 2;
      return `piksel (${p % a.width}, ${Math.floor(p / a.width)}) kanal ${i & 3}: ${a.data[i]} ≠ ${b.data[i]}`;
    }
  }
  return null;
}

describe("faults", () => {
  it("shift matches python", () => {
    const ir = decodePng(join(SYN, "darken_rgb_ir.png"));
    const expected = decodePng(join(SYN, "shift_ir_ir.png"));
    expect(firstDiff(shiftIr(ir, 20), expected)).toBeNull();
  });

  it("darken matches python", () => {
    const rgb = decodePng(join(SYN, "shift_ir_rgb.png"));
    const expected = decodePng(join(SYN, "darken_rgb_rgb.png"));
    // fikstürde .5 sınırına düşen pikseller var (v ≡ 2 mod 4): yuvarlama kuralı gerçekten sınanıyor
    let halves = 0;
    for (let i = 0; i < rgb.data.length; i++) if ((i & 3) !== 3 && rgb.data[i] % 4 === 2) halves++;
    expect(halves).toBeGreaterThan(0);
    expect(firstDiff(darken(rgb, 0.25), expected)).toBeNull();
  });

  it("darken rounds half to even", () => {
    expect(rintEven(2.5)).toBe(2);
    expect(rintEven(3.5)).toBe(4);
    expect(rintEven(-2.5)).toBe(-2);
    expect(rintEven(2.4)).toBe(2);
    expect(rintEven(2.6)).toBe(3);
    const out = darken(px(2, 1, [5, 7]), 0.5);
    expect([out.data[0], out.data[4]]).toEqual([2, 4]);
  });

  it("blackout is zeros", () => {
    const out = blackout(px(2, 2, [10, 200, 30, 255]));
    for (let i = 0; i < out.data.length; i++) expect(out.data[i]).toBe((i & 3) === 3 ? 255 : 0);
  });

  it("shift fills the vacated edge with zeros", () => {
    const out = shiftIr(px(4, 1, [1, 2, 3, 4]), 2);
    expect([0, 1, 2, 3].map((x) => out.data[x * 4])).toEqual([0, 0, 1, 2]);
  });

  it("ir noise keeps the image gray and is reproducible", () => {
    const img = px(8, 8, Array(64).fill(128));
    const a = noise(img, 20, mulberry32(1));
    const b = noise(img, 20, mulberry32(1));
    expect(a.data).toEqual(b.data);
    let changed = 0;
    for (let i = 0; i < a.data.length; i += 4) {
      expect(a.data[i + 1]).toBe(a.data[i]);
      expect(a.data[i + 2]).toBe(a.data[i]);
      if (a.data[i] !== 128) changed++;
    }
    expect(changed).toBeGreaterThan(32);
  });

  it("blur keeps a flat image flat and smooths an edge", () => {
    const flat = blur(px(5, 5, Array(25).fill(90)), 3);
    expect(Array.from(flat.data.filter((_, i) => (i & 3) !== 3))).toEqual(Array(75).fill(90));
    const edge = blur(px(4, 1, [0, 0, 255, 255]), 3);
    expect(edge.data[4]).toBeGreaterThan(0);
    expect(edge.data[8]).toBeLessThan(255);
  });

  it("only faulted channel is sent", async () => {
    expect(faultedChannels(NO_FAULTS)).toEqual({ rgb: false, ir: false });
    expect(faultedChannels({ ...NO_FAULTS, shift: 20 })).toEqual({ rgb: false, ir: true });
    expect(faultedChannels({ ...NO_FAULTS, darken: 0.25 })).toEqual({ rgb: true, ir: false });
    expect(faultedChannels({ ...NO_FAULTS, blur: 7 })).toEqual({ rgb: true, ir: true });
    expect(faultedChannels({ ...NO_FAULTS, offRgb: true })).toEqual({ rgb: true, ir: false });

    const meta = { seq: "a", frame: 3, conf: 0.25, track: true, gate: "auto", session: "s" };
    const ir = new Blob([new Uint8Array([1, 2, 3])], { type: "image/png" });
    const form = buildFaultForm({ ...meta, fault: { shift: 20 } }, { ir });
    expect(form.get("rgb")).toBeNull();
    const sent = form.get("ir");
    expect(sent).toBeInstanceOf(Blob);
    expect((sent as Blob).type).toBe("image/png");
    expect(new Uint8Array(await (sent as Blob).arrayBuffer())).toEqual(new Uint8Array([1, 2, 3]));
    expect(JSON.parse(form.get("meta") as string)).toEqual({ ...meta, fault: { shift: 20 } });
  });
});
