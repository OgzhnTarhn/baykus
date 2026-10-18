// Tarayıcı tarafı: motor yanıt tipleri (C++ sunucusu /v1/detect, Faz 3 tasarımı §4.1) ve Next
// rotalarına istekler. Bütün istekler aynı köke (127.0.0.1:3000); dış adres yok (kanıt 3).
import type { SequenceIndex } from "@/lib/data";

export type Gate = "off" | "advise" | "auto";

export type Detection = {
  cls: string;
  cls_id: number;
  score: number; // kalibre
  score_raw: number;
  box: { x1: number; y1: number; x2: number; y2: number };
  track_id: number | null;
};

export type CameraHealth = {
  state: string; // ok | lost | noisy | dark | blurry
  flags: string[];
  std: number | null;
  noise: number | null;
  blur: number | null;
  brightness?: number | null;
};

export type Health = {
  rgb: CameraHealth;
  ir: CameraHealth;
  pair: {
    misalign_px: number | null;
    dx_px: number | null;
    dy_px: number | null;
    response: number | null;
    misaligned_frame: boolean;
    misaligned: boolean;
  };
  gate: "none" | "ir_off" | "rgb_off";
  reason_code: string | null;
  reason: string;
  applied: boolean;
};

export type DetectResponse = {
  detections: Detection[];
  health: Health;
  timing_ms: Record<string, number>;
  session_new: boolean;
};

export type DetectSettings = { conf: number; track: boolean; gate: Gate; session: string };

export type { SequenceIndex };

async function json<T>(res: Response): Promise<T> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // gövde JSON değil; aşağıda durum koduyla ileti kurulur
  }
  if (!res.ok) {
    const msg = (body as { error?: unknown } | null)?.error;
    throw new Error(typeof msg === "string" ? msg : `istek başarısız (${res.status})`);
  }
  return body as T;
}

async function send(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, { cache: "no-store", ...init });
  } catch {
    throw new Error("arayüz sunucusu yanıt vermiyor");
  }
}

export function frameUrl(seq: string, frame: number, mod: "rgb" | "ir"): string {
  return `/api/frame/${encodeURIComponent(seq)}/${frame}/${mod}`;
}

export async function fetchSequences(): Promise<SequenceIndex[]> {
  return json<SequenceIndex[]>(await send("/api/sequences"));
}

export async function postDetect(
  seq: string,
  frame: number,
  s: DetectSettings,
): Promise<DetectResponse> {
  const res = await send("/api/detect", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ seq, frame, ...s }),
  });
  return json<DetectResponse>(res);
}

// Arıza açık: multipart (meta + bozulan kanallar, lib/faults buildFaultForm).
export async function postDetectForm(form: FormData): Promise<DetectResponse> {
  return json<DetectResponse>(await send("/api/detect", { method: "POST", body: form }));
}

export async function postReset(session: string): Promise<void> {
  const res = await send("/api/session/reset", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ session }),
  });
  await json<unknown>(res);
}
