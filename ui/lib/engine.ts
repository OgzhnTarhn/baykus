// C++ motorunun HTTP API'si (Faz 3 tasarımı §4.1). Motor ulaşılamazsa 502 ve anlaşılır ileti;
// motorun kendi yanıtı (200 / 400 / 413) olduğu gibi iletilir.
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

async function call(path: string, init: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${engineUrl()}${path}`, { ...init, cache: "no-store" });
  } catch {
    throw new HttpError(502, "motor yanıt vermiyor");
  }
  const body = await res.text();
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
