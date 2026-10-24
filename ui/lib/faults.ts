// Arıza simülasyonu (Faz 3 tasarımı §4.4, FR-U4): tarayıcı bozulmuş kanalı canvas'ta üretir ve PNG
// olarak gönderir. Tanımlar spec §12.3 ve baykus.robustness.corruptions ile aynı:
// - darken: float32 çarpım, yarıda çifte yuvarlama (np.rint), [0, 255] kırpma (Python ile birebir);
// - shiftIr: içerik sağa dx px, boşalan kenar 0 (Python ile birebir);
// - noise: Gauss σ, yarıda çifte yuvarlama; rastgele sayılar numpy ile aynı değil (demo, ölçüm değil);
// - blur: Gauss çekirdeği (OpenCV sigma=0 çekirdeği, kenar REFLECT_101; kayan nokta, birebir değil);
// - blackout: kamerayı kapat = sıfır görüntü.
// Pikseller ImageData düzeninde (RGBA, alfa 255); IR gri olduğu için R = G = B korunur.
import type { Freeze } from "@/lib/client";

export type Pixels = { width: number; height: number; data: Uint8ClampedArray };

export type FaultConfig = {
  darken: number; // RGB çarpanı; 1 = kapalı
  noise: number; // IR σ; 0 = kapalı
  shift: number; // IR sağa kaydırma px; 0 = kapalı
  blur: number; // iki kanal, çekirdek; 0 = kapalı
  offRgb: boolean;
  offIr: boolean;
  freezeRgb: boolean;
  freezeIr: boolean;
};

export const NO_FAULTS: FaultConfig = {
  darken: 1,
  noise: 0,
  shift: 0,
  blur: 0,
  offRgb: false,
  offIr: false,
  freezeRgb: false,
  freezeIr: false,
};

// Spec §12.3 şiddetleri (baykus.robustness.corruptions.SUITE)
export const LEVELS = {
  darken: [0.5, 0.25, 0.1],
  noise: [5, 10, 20],
  shift: [5, 10, 20, 40],
  blur: [3, 7, 11],
} as const;

// np.rint: en yakın tam sayı, yarıda çift olana.
export function rintEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

const clamp = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : v);

function copy(img: Pixels): Pixels {
  return { width: img.width, height: img.height, data: new Uint8ClampedArray(img.data) };
}

export function darken(img: Pixels, factor: number): Pixels {
  const out = copy(img);
  const f = Math.fround(factor);
  for (let i = 0; i < out.data.length; i++) {
    if ((i & 3) === 3) continue;
    out.data[i] = clamp(rintEven(Math.fround(img.data[i] * f)));
  }
  return out;
}

export function shiftIr(img: Pixels, dx: number): Pixels {
  const { width: w, height: h } = img;
  const out: Pixels = { width: w, height: h, data: new Uint8ClampedArray(img.data.length) };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      const sx = x - dx;
      if (sx >= 0 && sx < w) {
        const s = (y * w + sx) * 4;
        out.data[o] = img.data[s];
        out.data[o + 1] = img.data[s + 1];
        out.data[o + 2] = img.data[s + 2];
      }
      out.data[o + 3] = 255;
    }
  }
  return out;
}

// Tohumlu düzgün dağılım [0, 1) (tekrarlanabilir demo).
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// IR gürültüsü: piksel başına tek örnek (gri kalır), Box-Muller.
export function noise(img: Pixels, sigma: number, rng: () => number): Pixels {
  const out = copy(img);
  for (let i = 0; i < out.data.length; i += 4) {
    const u = 1 - rng();
    const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
    const v = clamp(rintEven(Math.fround(img.data[i] + Math.fround(g * sigma))));
    out.data[i] = out.data[i + 1] = out.data[i + 2] = v;
  }
  return out;
}

// OpenCV getGaussianKernel(k, 0): k ≤ 7 için sabit tablolar, üstü sigma = 0,3((k−1)/2 − 1) + 0,8.
function gaussianKernel(k: number): number[] {
  const table: Record<number, number[]> = {
    1: [1],
    3: [0.25, 0.5, 0.25],
    5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
    7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
  };
  if (table[k]) return table[k];
  const sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8;
  const r = (k - 1) / 2;
  const w = Array.from({ length: k }, (_, i) => Math.exp(-((i - r) ** 2) / (2 * sigma * sigma)));
  const s = w.reduce((a, b) => a + b, 0);
  return w.map((v) => v / s);
}

// BORDER_REFLECT_101: ... 2 1 | 0 1 2 ... n−1 | n−2 ...
function reflect101(i: number, n: number): number {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

export function blur(img: Pixels, k: number): Pixels {
  if (k <= 1) return copy(img);
  if (k % 2 === 0) throw new Error(`blur: çekirdek tek sayı olmalı, ${k} verildi`);
  const { width: w, height: h } = img;
  const kern = gaussianKernel(k);
  const r = (k - 1) / 2;
  const tmp = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let j = -r; j <= r; j++) s += kern[j + r] * img.data[(y * w + reflect101(x + j, w)) * 4 + c];
        tmp[(y * w + x) * 3 + c] = s;
      }
    }
  }
  const out: Pixels = { width: w, height: h, data: new Uint8ClampedArray(img.data.length) };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let j = -r; j <= r; j++) s += kern[j + r] * tmp[(reflect101(y + j, h) * w + x) * 3 + c];
        out.data[(y * w + x) * 4 + c] = clamp(rintEven(s));
      }
      out.data[(y * w + x) * 4 + 3] = 255;
    }
  }
  return out;
}

export function blackout(img: Pixels): Pixels {
  const out: Pixels = { width: img.width, height: img.height, data: new Uint8ClampedArray(img.data.length) };
  for (let i = 3; i < out.data.length; i += 4) out.data[i] = 255;
  return out;
}

// kareyi dondur (Faz 4 §5.6): o kanalın tuttuğu karenin özgün baytları diskten yeniden gönderilir
// (Next rotası meta.freeze). Donma açıkken ilk kare numarası tutulur, kapanınca bırakılır.
export function holdFrames(held: Freeze, f: FaultConfig, frame: number): Freeze {
  const out: Freeze = {};
  if (f.freezeRgb) out.rgb = held.rgb ?? frame;
  if (f.freezeIr) out.ir = held.ir ?? frame;
  return out;
}

export function faultedChannels(f: FaultConfig): { rgb: boolean; ir: boolean } {
  return {
    rgb: !f.freezeRgb && (f.offRgb || f.darken < 1 || f.blur > 0),
    ir: !f.freezeIr && (f.offIr || f.noise > 0 || f.shift > 0 || f.blur > 0),
  };
}

// Bir kanala açık arızaları uygular. Sıra: kamerayı kapat (gerisi anlamsız), karartma, bulanıklık,
// kaydırma, gürültü (sensör gürültüsü en son, görüntünün üstüne).
export function applyFaults(
  img: Pixels,
  channel: "rgb" | "ir",
  f: FaultConfig,
  rng: () => number,
): Pixels {
  if (channel === "rgb") {
    if (f.offRgb) return blackout(img);
    let out = f.darken < 1 ? darken(img, f.darken) : img;
    if (f.blur > 0) out = blur(out, f.blur);
    return out;
  }
  if (f.offIr) return blackout(img);
  let out = f.blur > 0 ? blur(img, f.blur) : img;
  if (f.shift > 0) out = shiftIr(out, f.shift);
  if (f.noise > 0) out = noise(out, f.noise, rng);
  return out;
}

// Kayıt ve gösterim için yalnız açık arızalar.
export function faultSummary(f: FaultConfig): Record<string, number | boolean> {
  const out: Record<string, number | boolean> = {};
  if (f.darken < 1) out.darken = f.darken;
  if (f.noise > 0) out.noise = f.noise;
  if (f.shift > 0) out.shift = f.shift;
  if (f.blur > 0) out.blur = f.blur;
  if (f.offRgb) out.off_rgb = true;
  if (f.offIr) out.off_ir = true;
  if (f.freezeRgb) out.freeze_rgb = true;
  if (f.freezeIr) out.freeze_ir = true;
  return out;
}

// multipart: `meta` (JSON) + yalnız bozulan kanallar; eksik kanalı Next sunucusu diskten tamamlar.
export function buildFaultForm(
  meta: Record<string, unknown>,
  parts: { rgb?: Blob; ir?: Blob },
): FormData {
  const form = new FormData();
  form.set("meta", JSON.stringify(meta));
  if (parts.rgb) form.set("rgb", parts.rgb, "rgb.png");
  if (parts.ir) form.set("ir", parts.ir, "ir.png");
  return form;
}

// Tarayıcıda: canvas → PNG (kayıpsız; JPEG yeniden sıkıştırması ek bozulma eklemesin).
export function encodePng(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG kodlanamadı"))), "image/png"),
  );
}
