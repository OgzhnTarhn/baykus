// C++ motorunun HTTP API'si (Faz 3 tasarımı §4.1). Motor ulaşılamazsa 502, zaman aşımında 504 ve
// anlaşılır ileti; motorun kendi yanıtı (200 / 400 / 413) olduğu gibi iletilir.
// Faz 4 §5.7: bölge / olay / alarm / iz / model uçları da aynı sözleşmeyle iletilir.
import { HttpError } from "@/lib/data";
import type { ResetScope } from "@/lib/client";

export type DetectParams = {
  conf: number;
  track: boolean;
  gate: string;
  session: string;
  frame?: number;
  focus?: number | null;
  tiles?: number;
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
  if (p.frame !== undefined) q.set("frame", String(p.frame));
  if (p.focus !== undefined && p.focus !== null) q.set("focus", String(p.focus));
  if (p.tiles !== undefined) q.set("tiles", String(p.tiles));
  const form = new FormData();
  form.set("rgb", new Blob([new Uint8Array(rgb.bytes)], { type: rgb.type }), "rgb");
  form.set("ir", new Blob([new Uint8Array(ir.bytes)], { type: ir.type }), "ir");
  return call(`/v1/detect?${q}`, { method: "POST", body: form });
}

export function resetSession(session: string, scope?: ResetScope): Promise<Response> {
  const q = new URLSearchParams({ session });
  if (scope !== undefined) q.set("scope", scope);
  return call(`/v1/session/reset?${q}`, { method: "POST" });
}

export function getZones(session: string): Promise<Response> {
  const q = new URLSearchParams({ session });
  return call(`/v1/session/zones?${q}`, { method: "GET" });
}

export function putZones(session: string, body: string): Promise<Response> {
  const q = new URLSearchParams({ session });
  return call(`/v1/session/zones?${q}`, {
    method: "PUT",
    body,
    headers: { "content-type": "application/json" },
  });
}

export function getEvents(session: string, since: number, format: "json" | "csv"): Promise<Response> {
  const q = new URLSearchParams({ session, since: String(since), format });
  return call(`/v1/session/events?${q}`, { method: "GET" });
}

export function ackAlarm(session: string, alarm: string): Promise<Response> {
  const q = new URLSearchParams({ session, alarm });
  return call(`/v1/session/alarms/ack?${q}`, { method: "POST" });
}

export function getTrack(session: string, id: number): Promise<Response> {
  const q = new URLSearchParams({ session, id: String(id) });
  return call(`/v1/session/track?${q}`, { method: "GET" });
}

export function getModel(): Promise<Response> {
  return call("/v1/model", { method: "GET" });
}
