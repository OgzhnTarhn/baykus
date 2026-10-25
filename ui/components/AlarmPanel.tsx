"use client";
// Bölge alarmları sekmesi (Faz 4 tasarımı §5.8): açık alarmlar (onay düğmesiyle) ve geçmiş
// (onaylandı / onaylanmadan kapandı). Türkçe metin texts.ts tablosundan; renk tek başına anlam
// taşımaz, her satır tür etiketini de yazar.
import type { Alarm, ZoneDef } from "@/lib/client";
import { ALARM_TEXT_ZONE, text } from "@/lib/texts";

// Ondalık virgüllü tek hane (Türkçe ekran).
const fmt1 = (v: number) => v.toFixed(1).replace(".", ",");

const HISTORY_MAX = 50;

const KIND_LABEL: Record<Alarm["kind"], string> = {
  intrusion: "Yasak",
  dwell_exceeded: "Bekleme",
};

const KIND_BORDER: Record<Alarm["kind"], string> = {
  intrusion: "border-l-4 border-red-500",
  dwell_exceeded: "border-l-4 border-orange-400",
};

function zoneName(zones: ZoneDef[], zoneId: string): string {
  return zones.find((z) => z.id === zoneId)?.name ?? zoneId;
}

function trackText(ids: number[]): string {
  return ids.length === 0 ? "" : `iz ${ids.map((id) => `#${id}`).join(", ")}`;
}

function durationText(frames: number, fps: number): string {
  const seconds = fps > 0 ? frames / fps : 0;
  return `${frames} kare (≈ ${fmt1(seconds)} sn)`;
}

export function AlarmPanel({
  open,
  history,
  zones,
  frame,
  fps,
  onAck,
  pending,
}: {
  open: Alarm[];
  history: { alarm: Alarm; cleared_frame: number; acked: boolean }[];
  zones: ZoneDef[];
  frame: number;
  fps: number;
  onAck(id: string): void;
  pending: Set<string>;
}) {
  const recent = history.slice(0, HISTORY_MAX);
  return (
    <section className="flex flex-col gap-3 rounded border border-line bg-panel p-3">
      <div className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">Açık alarmlar</h2>
        {open.length === 0 ? (
          <p className="text-sm text-muted">Açık bölge alarmı yok.</p>
        ) : (
          <ul className="flex flex-col gap-1" role="status">
            {open.map((a) => (
              <li
                key={a.id}
                className={`flex flex-wrap items-center gap-x-3 gap-y-1 rounded border border-line bg-panel-2 px-2 py-1 text-sm ${KIND_BORDER[a.kind]}`}
              >
                <span className="font-medium">{text(ALARM_TEXT_ZONE, a.kind)}</span>
                <span className="rounded border border-line px-1.5 text-xs">
                  {KIND_LABEL[a.kind]}
                </span>
                <span>{zoneName(zones, a.zone_id)}</span>
                <span className="num text-xs text-muted">{trackText(a.track_ids)}</span>
                <span className="num text-xs text-muted">başlangıç kare {a.since_frame}</span>
                <span className="num text-xs text-muted">
                  {durationText(frame - a.since_frame, fps)}
                </span>
                {a.acked ? (
                  <span className="ml-auto text-xs text-muted">onaylandı</span>
                ) : (
                  <button
                    type="button"
                    className="ml-auto rounded border border-line px-2 py-0.5 text-xs hover:border-text disabled:opacity-40"
                    disabled={pending.has(a.id)}
                    onClick={() => onAck(a.id)}
                  >
                    Onayla
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">Geçmiş</h2>
        {recent.length === 0 ? (
          <p className="text-sm text-muted">Kapanmış bölge alarmı yok.</p>
        ) : (
          <ol className="max-h-48 overflow-y-auto text-xs" aria-label="Bölge alarm geçmişi">
            {recent.map((h, i) => (
              <li key={`${h.alarm.id}-${h.cleared_frame}-${i}`} className="flex gap-2 py-0.5">
                <span className="num w-32 text-muted">
                  kare {h.alarm.since_frame}–{h.cleared_frame}
                </span>
                <span>{text(ALARM_TEXT_ZONE, h.alarm.kind)}</span>
                <span className="text-muted">{zoneName(zones, h.alarm.zone_id)}</span>
                <span className="ml-auto">{h.acked ? "onaylandı" : "onaylanmadan kapandı"}</span>
              </li>
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}
