// C++ motorunun HTTP API'si (Faz 3 tasarımı §4.1). Motor ulaşılamazsa 502, zaman aşımında 504 ve
// anlaşılır ileti; motorun kendi yanıtı (200 / 400 / 413) olduğu gibi iletilir.
import { HttpError } from "@/lib/data";

export type DetectParams = {
  conf: number;
  track: boolean;
  gate: string;
  session: string;
};

export type Part = { bytes: Uint8Array; type: string };

export function engineUrl(): string {
  return process.env.ENGINE_URL ?? "http://engine:8080";
}

// Motor bir isteği bu süreden uzun tutarsa iptal edilir (S11); oynatıcı askıda kalmaz.
export const ENGINE_TIMEOUT_MS = 10_000;

async function call(path: string, init: RequestInit): Promise<Response> {
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), ENGINE_TIMEOUT_MS);
  let res: Response;
  let body: string;
  try {
    res = await fetch(`${engineUrl()}${path}`, { ...init, cache: "no-store", signal: abort.signal });
    body = await res.text();
  } catch {
    if (abort.signal.aborted) throw new HttpError(504, "motor zaman aşımı");
    throw new HttpError(502, "motor yanıt vermiyor");
  } finally {
    clearTimeout(timer);
  }
  return new Response(body, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
  });
}

export function detect(p: DetectParams, rgb: Part, ir: Part): Promise<Response> {
  const q = new URLSearchParams({
    conf: String(p.conf),
    track: String(p.track),
    gate: p.gate,
    session: p.session,
  });
  const form = new FormData();
  form.set("rgb", new Blob([new Uint8Array(rgb.bytes)], { type: rgb.type }), "rgb");
  form.set("ir", new Blob([new Uint8Array(ir.bytes)], { type: ir.type }), "ir");
  return call(`/v1/detect?${q}`, { method: "POST", body: form });
}

export function resetSession(session: string): Promise<Response> {
  const q = new URLSearchParams({ session });
  return call(`/v1/session/reset?${q}`, { method: "POST" });
}
