// Demo verisi (Faz 3 tasarımı §4.4): /data/sequences/<dizi>/{rgb/<kare:06d>.jpg, ir/<kare:06d>.png,
// index.json}. Test bölümü hiç gösterilmez (demo paketine zaten girmez; bu ikinci emniyet). Dizi
// adı ve kare numarası doğrulanmadan dosya yolu kurulmaz (yol geçişi yok).
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

export type SequenceIndex = {
  name: string;
  split: string;
  frames: number[];
  fps: number;
  size: [number, number];
  source: string;
};

export type Modality = "rgb" | "ir";

export class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

const SEQ_RE = /^[A-Za-z0-9_-]+$/;
const FRAME_RE = /^\d+$/;

export function dataDir(): string {
  return process.env.DATA_DIR ?? "/data";
}

function seqDir(seq: string): string {
  if (!SEQ_RE.test(seq)) throw new HttpError(400, `geçersiz dizi adı: ${seq}`);
  return join(dataDir(), "sequences", seq);
}

async function readIndex(dir: string): Promise<SequenceIndex | null> {
  try {
    return JSON.parse(await readFile(join(dir, "index.json"), "utf-8")) as SequenceIndex;
  } catch {
    return null;
  }
}

export async function listSequences(): Promise<SequenceIndex[]> {
  const root = join(dataDir(), "sequences");
  let names: string[] = [];
  try {
    names = await readdir(root);
  } catch {
    return [];
  }
  const out: SequenceIndex[] = [];
  for (const name of names.filter((n) => SEQ_RE.test(n)).sort()) {
    const index = await readIndex(join(root, name));
    if (index && index.split !== "test") out.push(index);
  }
  return out;
}

export async function loadIndex(seq: string): Promise<SequenceIndex> {
  const index = await readIndex(seqDir(seq));
  if (!index || index.split === "test") throw new HttpError(404, `dizi yok: ${seq}`);
  return index;
}

export function parseFrame(frame: string | number): number {
  const s = String(frame);
  if (!FRAME_RE.test(s)) throw new HttpError(400, `geçersiz kare numarası: ${s}`);
  return Number(s);
}

export function parseModality(mod: string): Modality {
  if (mod !== "rgb" && mod !== "ir") throw new HttpError(400, `geçersiz kanal: ${mod}`);
  return mod;
}

// Doğrulanmış yol: dizi adı deseni, kare tam sayı ve index.frames içinde, kanal rgb|ir.
export async function framePath(
  seq: string,
  frame: string | number,
  mod: string,
): Promise<string> {
  const m = parseModality(mod);
  const n = parseFrame(frame);
  const index = await loadIndex(seq);
  if (!index.frames.includes(n)) throw new HttpError(404, `kare yok: ${seq} ${n}`);
  const name = `${String(n).padStart(6, "0")}.${m === "rgb" ? "jpg" : "png"}`;
  return join(seqDir(seq), m, name);
}

export async function readFrame(seq: string, frame: string | number, mod: string): Promise<Buffer> {
  const path = await framePath(seq, frame, mod);
  try {
    return await readFile(path);
  } catch {
    throw new HttpError(404, `kare dosyası yok: ${seq} ${frame} ${mod}`);
  }
}

export function errorResponse(e: unknown): Response {
  if (e instanceof HttpError) return Response.json({ error: e.message }, { status: e.status });
  console.error(e);
  return Response.json({ error: "iç hata" }, { status: 500 });
}
