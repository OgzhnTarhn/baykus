// Tarayıcı tarafı: motor yanıt tipleri (C++ sunucusu /v1/detect, Faz 3 tasarımı §4.1) ve Next
// rotalarına istekler. Bütün istekler aynı köke (127.0.0.1:3000); dış adres yok (kanıt 3).
import type { SequenceIndex } from "@/lib/data";

export type Gate = "off" | "advise" | "auto";

// Faz 4 tasarımı §5.7: hareket durumu, takip katmanı, ego, bölgeler, olaylar, alarmlar, araç özeti.
export type Motion = "moving" | "stationary" | "unknown";
export type Affine = [number, number, number, number, number, number]; // [a, b, tx, c, d, ty]
export type Box = { x1: number; y1: number; x2: number; y2: number };

export type Detection = {
  cls: string;
  cls_id: number;
  score: number; // kalibre
  score_raw: number;
  box: Box;
  track_id: number | null;
  motion?: Motion; // Faz 4 (oturumsuz: "unknown")
  speed_rel?: number | null;
  map_xy?: [number, number] | null;
};

// Takip katmanı (ham ≥ 0,1, conf süzgeci yok): harita, bölge ve kart bunu kullanır.
export type Track = {
  track_id: number;
  cls_id: number;
  box: Box;
  score_raw: number;
  score: number;
  motion: Motion;
  speed_rel: number | null;
  map_xy: [number, number] | null;
};

export type Ego = {
  ok: boolean;
  source: "ir" | "rgb" | "none";
  map_gen: number;
  provisional: boolean;
  frame_to_map: Affine | null;
  inliers: number | null;
  inlier_ratio: number | null;
  keyframe: boolean;
  relinked: boolean;
  relocalized: boolean;
  merge: { from: number; to: number; transform: Affine } | null;
};

export type ZoneKind = "area" | "restricted" | "dwell" | "line";
export type LineCount = "both" | "a_to_b" | "b_to_a";

export type ZoneDef = {
  id: string;
  name: string;
  kind: ZoneKind;
  gen: number;
  points: [number, number][];
  dwell_frames?: number; // yalnız dwell
  count?: LineCount; // yalnız line
  status?: "active" | "suspended"; // yalnız GET yanıtında
};

// Günlük kaydı (§5.13); /v1/detect `events[]` ve /v1/session/events aynı biçim.
export type ZoneEvent = {
  seq: number;
  n: number;
  frame: number;
  kind: string;
  zone_id: string | null;
  zone_name: string | null;
  zone_kind: string | null;
  track_id: number | null;
  dir: "a_to_b" | "b_to_a" | null;
  gen: number;
  alarm_id: string | null;
};

export type Alarm = {
  id: string;
  kind: "intrusion" | "dwell_exceeded";
  zone_id: string;
  track_ids: number[];
  since_n: number;
  since_frame: number;
  acked: boolean;
};

export type ZoneCount = {
  status: "active" | "suspended";
  inside: number | null;
  moving: number | null;
  stationary: number | null;
  unknown: number | null;
  entered_total: number | null;
  exceeded_total: number | null;
  a_to_b: number | null;
  b_to_a: number | null;
};
export type ZoneCounts = Record<string, ZoneCount>;

export type TrackSummary = {
  track_id: number;
  first_frame: number;
  last_frame: number;
  cls_id: number;
  class: string;
  motion: Motion;
  motion_since_frame: number;
  segments: { state: Motion; from_frame: number; to_frame: number }[];
  stationary_frames: number;
  path_rel: number | null;
  zones: { zone_id: string; enter_frame: number; exit_frame: number | null }[];
  alive: boolean;
};

export type EventsPage = { events: ZoneEvent[]; last_seq: number; dropped: number };

export type ResetScope = "all" | "tracks";
export type Tiles = 1 | 4;

export type CameraHealth = {
  state: string; // ok | lost | noisy | dark | blurry
  flags: string[];
  std: number | null;
  noise: number | null;
  blur: number | null;
  brightness?: number | null;
  frozen?: boolean | null; // Faz 4 §5.6 (oturumsuz: null)
  diff_mad?: number | null;
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
  // Faz 4 (oturumsuz: tracks/events/alarms boş, ego null, zone_counts {})
  tracks?: Track[];
  ego?: Ego | null;
  events?: ZoneEvent[];
  alarms?: Alarm[];
  zone_counts?: ZoneCounts;
  focus_track?: TrackSummary | null;
};

export type DetectSettings = {
  conf: number;
  track: boolean;
  gate: Gate;
  session: string;
  tiles?: Tiles; // küçük hedef modu (Görev 15D); yoksa motor varsayılanı
};

// Kareyi dondur (§5.6): kanalın tekrar gönderilecek kare numarası.
export type Freeze = { rgb?: number; ir?: number };

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
