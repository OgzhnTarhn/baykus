// Zaman çizelgesi (Faz 4 §5.8): kare günlüğü olaylarını oturum damgasıyla toplar, süzgeçler,
// işaretçilere ölçekler ve alarm bantlarını açar. Eski oturum / dönemden gelen olaylar atılır
// (Review Focus 5): damga tutmayan add() çağrısı hiçbir şey yazmaz.
import type { ZoneEvent } from "@/lib/client";

// İşaretçi biçimleri renkten bağımsızdır: her olay türü kendi şekliyle ayırt edilir, renk
// yalnızca destekleyici bir ipucudur (renk körlüğü / gri ekran). Tür → şekil sabittir.
export type MarkerShape = "up" | "down" | "cross" | "square" | "diamond" | "bar" | "circle" | "dot";

export const MARKER_SHAPE: Record<string, MarkerShape> = {
  enter: "up",
  exit: "down",
  exit_lost: "cross",
  stopped: "square",
  line_cross: "diamond",
  alarm_start: "bar",
  alarm_ack: "bar",
  alarm_clear: "bar",
};

// map_* (konum aranıyor, yeniden konumlandı, birleşti, yeni harita) daire; bilinmeyen tür nokta.
export function shapeOf(kind: string): MarkerShape {
  return MARKER_SHAPE[kind] ?? (kind.startsWith("map_") ? "circle" : "dot");
}

export type Marker = { x: number; frame: number; seq: number; kind: string; shape: MarkerShape };

export type Band = { alarm_id: string; from: number; to: number | null; kind: string };

export type TimelineFilter = { kinds?: string[]; zone?: string; track?: number };

export class Timeline {
  private session: string | null = null;
  private epoch = 0;
  private events: ZoneEvent[] = [];
  private seqs = new Set<number>();

  // Oturum damgasını taşır ve birikmiş olayları temizler.
  reset(session: string, epoch: number): void {
    this.session = session;
    this.epoch = epoch;
    this.events = [];
    this.seqs = new Set();
  }

  // Aynı oturumda olaylar korunarak damga ilerletilir (aynı dizide atlama, scope=tracks: günlük
  // motorda korunur). Eski damgalı add() çağrıları bundan sonra yok sayılır.
  restamp(session: string, epoch: number): void {
    this.session = session;
    this.epoch = epoch;
  }

  // Damga tutmayan (eski oturum / dönem) olaylar yok sayılır; aynı seq bir kez saklanır.
  add(events: ZoneEvent[], session: string, epoch: number): void {
    if (session !== this.session || epoch !== this.epoch) return;
    for (const e of events) {
      if (this.seqs.has(e.seq)) continue;
      this.seqs.add(e.seq);
      this.events.push(e);
    }
    this.events.sort((a, b) => a.frame - b.frame || a.seq - b.seq);
  }

  all(): ZoneEvent[] {
    return [...this.events];
  }

  filtered(f: TimelineFilter): ZoneEvent[] {
    const kinds = f.kinds && f.kinds.length ? new Set(f.kinds) : null;
    return this.events.filter(
      (e) =>
        (!kinds || kinds.has(e.kind)) &&
        (f.zone === undefined || e.zone_id === f.zone || e.zone_name === f.zone) &&
        (f.track === undefined || e.track_id === f.track),
    );
  }

  // Kare numarasını [0, width] aralığına ölçekler; frameCount 1 iken bölme yok. `events` verilirse
  // (ör. süzgeçten geçenler) yalnız onlar işaretlenir.
  markers(frameCount: number, width: number, firstFrame = 1, events = this.events): Marker[] {
    const span = Math.max(1, frameCount - 1);
    return events.map((e) => ({
      x: ((e.frame - firstFrame) / span) * width,
      frame: e.frame,
      seq: e.seq,
      kind: e.kind,
      shape: shapeOf(e.kind),
    }));
  }

  // alarm_start bandı açar, alarm_clear kapatır; kapanmayan bant to: null ile açık kalır.
  alarmBands(): Band[] {
    const bands: Band[] = [];
    const open = new Map<string, Band>();
    for (const e of this.events) {
      if (e.alarm_id === null) continue;
      if (e.kind === "alarm_start") {
        const band: Band = {
          alarm_id: e.alarm_id,
          from: e.frame,
          to: null,
          kind: e.zone_kind === "restricted" ? "intrusion" : "dwell_exceeded",
        };
        open.set(e.alarm_id, band);
        bands.push(band);
      } else if (e.kind === "alarm_clear") {
        const band = open.get(e.alarm_id);
        if (band) {
          band.to = e.frame;
          open.delete(e.alarm_id);
        }
      }
    }
    return bands;
  }
}
