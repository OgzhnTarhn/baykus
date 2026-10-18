// Tespit (Faz 3 tasarımı §4.4). İki biçim:
// - JSON {"seq","frame","conf","track","gate","session"}: iki kanalın özgün baytları diskten
//   (tarayıcıda yeniden kodlama pikselleri Python referansından uzaklaştırmasın).
// - multipart: `meta` (aynı JSON + isteğe bağlı "fault") ve bozulan kanallar `rgb` / `ir`
//   (image/png, kare boyutunda, ≤ 10 MB); gönderilmeyen kanal diskten tamamlanır.
import { errorResponse, framePath, HttpError, loadIndex, readFrame } from "@/lib/data";
import { detect, type DetectParams, type Part } from "@/lib/engine";

const MAX_PART_BYTES = 10 * 1024 * 1024; // motorla aynı sınır
const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

type Meta = DetectParams & { seq: string; frame: number };

function parseMeta(raw: unknown): Meta {
  const m = (raw ?? {}) as Record<string, unknown>;
  const frame = typeof m.frame === "string" ? Number(m.frame) : m.frame;
  if (typeof m.seq !== "string" || typeof frame !== "number" || !Number.isInteger(frame)) {
    throw new HttpError(400, "seq ve frame zorunlu");
  }
  if (typeof m.conf !== "number" || typeof m.track !== "boolean" || typeof m.gate !== "string") {
    throw new HttpError(400, "conf (sayı), track (true/false) ve gate zorunlu");
  }
  const session = typeof m.session === "string" ? m.session : "";
  return { seq: m.seq, frame, conf: m.conf, track: m.track, gate: m.gate, session };
}

// PNG başlığından (IHDR) genişlik ve yükseklik; PNG değilse null.
function pngSize(b: Uint8Array): [number, number] | null {
  if (b.length < 24 || PNG_SIGNATURE.some((v, i) => b[i] !== v)) return null;
  if (String.fromCharCode(...b.subarray(12, 16)) !== "IHDR") return null;
  const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  return [view.getUint32(16), view.getUint32(20)];
}

async function readJson(req: Request): Promise<unknown> {
  try {
    return await req.json();
  } catch {
    throw new HttpError(400, "geçersiz JSON gövdesi");
  }
}

async function faultParts(form: FormData, meta: Meta): Promise<Partial<Record<"rgb" | "ir", Part>>> {
  const index = await loadIndex(meta.seq);
  const out: Partial<Record<"rgb" | "ir", Part>> = {};
  for (const key of ["rgb", "ir"] as const) {
    const v = form.get(key);
    if (v === null) continue;
    if (typeof v === "string") throw new HttpError(400, `${key} dosya olmalı`);
    if (v.size > MAX_PART_BYTES) throw new HttpError(413, `${key} parçası 10 MB'tan büyük`);
    if (v.type !== "image/png") throw new HttpError(400, `${key} image/png olmalı`);
    const bytes = new Uint8Array(await v.arrayBuffer());
    const size = pngSize(bytes);
    if (!size) throw new HttpError(400, `${key} geçerli bir PNG değil`);
    if (size[0] !== index.size[0] || size[1] !== index.size[1]) {
      throw new HttpError(400, `${key} ${index.size[0]}×${index.size[1]} olmalı`);
    }
    out[key] = { bytes, type: "image/png" };
  }
  return out;
}

export async function POST(req: Request) {
  try {
    const ct = req.headers.get("content-type") ?? "";
    let meta: Meta;
    let parts: Partial<Record<"rgb" | "ir", Part>> = {};
    if (ct.startsWith("application/json")) {
      meta = parseMeta(await readJson(req));
    } else if (ct.startsWith("multipart/form-data")) {
      let form: FormData;
      try {
        form = await req.formData();
      } catch {
        throw new HttpError(400, "geçersiz multipart gövdesi");
      }
      const raw = form.get("meta");
      if (typeof raw !== "string") throw new HttpError(400, "meta alanı zorunlu");
      try {
        meta = parseMeta(JSON.parse(raw));
      } catch (e) {
        throw e instanceof HttpError ? e : new HttpError(400, "meta geçerli JSON değil");
      }
      parts = await faultParts(form, meta);
    } else {
      throw new HttpError(400, "application/json ya da multipart/form-data bekleniyor");
    }
    await framePath(meta.seq, meta.frame, "rgb"); // dizi ve kare doğrulaması (400 / 404)
    const rgb = parts.rgb ?? {
      bytes: await readFrame(meta.seq, meta.frame, "rgb"),
      type: "image/jpeg",
    };
    const ir = parts.ir ?? { bytes: await readFrame(meta.seq, meta.frame, "ir"), type: "image/png" };
    return await detect(meta, rgb, ir);
  } catch (e) {
    return errorResponse(e);
  }
}
