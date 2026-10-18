// Yalnız testler için küçük PNG çözücü (8 bit gri ya da RGB, araya serpiştirme yok): fikstür
// PNG'lerini tarayıcıdaki ImageData düzenine (RGBA) çevirir.
import { readFileSync } from "node:fs";
import { inflateSync } from "node:zlib";

import type { Pixels } from "@/lib/faults";

export function decodePng(path: string): Pixels {
  const b = readFileSync(path);
  let pos = 8;
  let width = 0;
  let height = 0;
  let colorType = 0;
  const idat: Buffer[] = [];
  while (pos < b.length) {
    const len = b.readUInt32BE(pos);
    const type = b.toString("ascii", pos + 4, pos + 8);
    const data = b.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      if (data[8] !== 8 || data[12] !== 0) throw new Error("yalnız 8 bit, serpiştirmesiz PNG");
      colorType = data[9];
    } else if (type === "IDAT") {
      idat.push(data);
    }
    pos += 12 + len;
  }
  const ch = colorType === 0 ? 1 : colorType === 2 ? 3 : 0;
  if (ch === 0) throw new Error(`desteklenmeyen PNG renk tipi: ${colorType}`);
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * ch;
  const cur = new Uint8Array(stride);
  const prev = new Uint8Array(stride);
  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const off = y * (stride + 1);
    const filter = raw[off];
    for (let x = 0; x < stride; x++) {
      const v = raw[off + 1 + x];
      const a = x >= ch ? cur[x - ch] : 0;
      const up = prev[x];
      const c = x >= ch ? prev[x - ch] : 0;
      let p: number;
      if (filter === 0) p = 0;
      else if (filter === 1) p = a;
      else if (filter === 2) p = up;
      else if (filter === 3) p = (a + up) >> 1;
      else {
        const pa = Math.abs(up - c);
        const pb = Math.abs(a - c);
        const pc = Math.abs(a + up - 2 * c);
        p = pa <= pb && pa <= pc ? a : pb <= pc ? up : c;
      }
      cur[x] = (v + p) & 0xff;
    }
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      if (ch === 1) {
        out[o] = out[o + 1] = out[o + 2] = cur[x];
      } else {
        out[o] = cur[x * 3];
        out[o + 1] = cur[x * 3 + 1];
        out[o + 2] = cur[x * 3 + 2];
      }
      out[o + 3] = 255;
    }
    prev.set(cur);
  }
  return { width, height, data: out };
}
