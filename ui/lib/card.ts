// Araç kartı (Faz 4 tasarımı §5.8): TrackSummary -> ekran alanları. Süreler kare cinsinden tutulur;
// saniye değeri index.json fps'inden türetilir ve "(… kare/sn varsayımı)" notuyla gösterilir.
// Türkçe metin texts.ts tablosundan gelir.
import type { Motion, TrackSummary } from "@/lib/client";
import { boxStyle } from "@/lib/motion";
import { MOTION_TEXT, text } from "@/lib/texts";

export type CardView = {
  title: string;
  state: Motion;
  stateText: string;
  tag: string | null; // kutudaki harf etiketi (D/H); bilinmiyorsa null
  durationText: string;
  fpsNote: string;
  stationaryText: string;
  pathText: string;
  ended: boolean;
  lastSeenText: string | null;
  segments: { state: Motion; ratio: number }[];
  zones: { zone_id: string; name: string; text: string }[];
};

// Ondalık virgüllü tek hane (Türkçe ekran).
const fmt1 = (v: number) => v.toFixed(1).replace(".", ",");

const VERB: Record<Motion, string> = {
  moving: "hareket ediyor",
  stationary: "duruyor",
  unknown: "",
};

export function cardView(
  summary: TrackSummary,
  fps: number,
  currentFrame: number,
  zoneNames: Record<string, string> = {},
): CardView {
  const { motion } = summary;
  const frames = currentFrame - summary.motion_since_frame + 1;
  const seconds = fps > 0 ? frames / fps : 0;

  const durationText =
    motion === "unknown"
      ? "durum henüz belli değil"
      : `${frames} kare (≈ ${fmt1(seconds)} sn) ${VERB[motion]}`;

  const total = summary.last_frame - summary.first_frame + 1;
  const segments = summary.segments.map((s) => ({
    state: s.state,
    ratio: total > 0 ? (s.to_frame - s.from_frame + 1) / total : 0,
  }));

  const zones = summary.zones.map((z) => ({
    zone_id: z.zone_id,
    name: zoneNames[z.zone_id] ?? z.zone_id,
    text:
      z.exit_frame === null
        ? `içeride (giriş kare ${z.enter_frame})`
        : `giriş kare ${z.enter_frame}, çıkış kare ${z.exit_frame}`,
  }));

  const ended = !summary.alive;

  return {
    title: `#${summary.track_id} ${summary.class}`,
    state: motion,
    stateText: text(MOTION_TEXT, motion),
    tag: boxStyle(motion).tag,
    durationText,
    fpsNote: `(${fps} kare/sn varsayımı)`,
    stationaryText: `${summary.stationary_frames} kare`,
    pathText: summary.path_rel === null ? "–" : `${fmt1(summary.path_rel)} kutu boyu`,
    ended,
    lastSeenText: ended
      ? `son görülme: kare ${summary.last_frame} (${currentFrame - summary.last_frame} kare önce)`
      : null,
    segments,
    zones,
  };
}
