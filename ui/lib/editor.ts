// Bölge editörü taslağı (Faz 4 §5.8): her değişiklik motora PUT ile gider; motor reddederse
// son kabul edilen kümeye dönülür ve hata ekranda tutulur.
import type { LineCount, ZoneDef, ZoneKind } from "@/lib/client";
import { ZONE_KIND_TEXT } from "@/lib/texts";

export const DEFAULT_DWELL_SECONDS = 30;

export type Tool = ZoneKind;

export type CommitResult = { ok: true } | { ok: false; error: string };

const MAX_POINTS = 32;
const MIN_POLYGON_POINTS = 3;

function copyPoint(p: [number, number]): [number, number] {
  return [p[0], p[1]];
}

// Durum yalnız GET yanıtında; çalışma kümesi PUT tanımını taşır.
function copyZone(z: ZoneDef): ZoneDef {
  const out: ZoneDef = { ...z, points: z.points.map(copyPoint) };
  delete out.status;
  return out;
}

export class ZoneDraft {
  private fps: number;
  private zones: ZoneDef[] = [];
  private accepted: ZoneDef[] = [];
  private suspended = new Set<string>();
  private tool: Tool | null = null;
  private draftPoints: [number, number][] = [];
  private draftGen = 1;
  private nextId = 1;
  private lastError: string | null = null;

  constructor(fps: number) {
    this.fps = fps;
  }

  // GET yanıtı: çalışma ve son onaylı küme aynı olur, kimlik sayacı geri gitmez.
  load(zones: ZoneDef[]): void {
    this.zones = zones.map(copyZone);
    this.accepted = zones.map(copyZone);
    this.suspended = new Set(
      zones.filter((z) => z.status === "suspended").map((z) => z.id),
    );
    for (const z of zones) {
      const m = /^z(\d+)$/.exec(z.id);
      if (m) this.nextId = Math.max(this.nextId, Number(m[1]) + 1);
    }
    this.tool = null;
    this.draftPoints = [];
    this.lastError = null;
  }

  defs(): ZoneDef[] {
    return this.zones.map(copyZone);
  }

  drawing(): { tool: Tool | null; points: [number, number][] } {
    return { tool: this.tool, points: this.draftPoints.map(copyPoint) };
  }

  error(): string | null {
    return this.lastError;
  }

  isSuspended(id: string): boolean {
    return this.suspended.has(id);
  }

  start(tool: Tool, gen: number): void {
    this.tool = tool;
    this.draftGen = gen;
    this.draftPoints = [];
  }

  // Nokta ekler; hat ikinci noktada kendiliğinden biter.
  addPoint(p: [number, number]): ZoneDef | null {
    if (!this.tool) return null;
    // Çift tıklamanın önündeki iki tek tıklama aynı noktayı iki kez getirir: art arda aynı nokta yok.
    const last = this.draftPoints[this.draftPoints.length - 1];
    if (last !== undefined && last[0] === p[0] && last[1] === p[1]) return null;
    this.draftPoints.push(copyPoint(p));
    if (this.tool === "line" && this.draftPoints.length >= 2) return this.finish();
    return null;
  }

  // Enter: hat 2, çokgen 3 nokta ister; yetersizse çizim açık kalır.
  finish(): ZoneDef | null {
    if (!this.tool) return null;
    const min = this.tool === "line" ? 2 : MIN_POLYGON_POINTS;
    if (this.draftPoints.length < min) return null;

    const n = this.nextId++;
    const kind = this.tool;
    const def: ZoneDef = {
      id: `z${n}`,
      name: `${ZONE_KIND_TEXT[kind]} ${n}`,
      kind,
      gen: this.draftGen,
      points: this.draftPoints.map(copyPoint),
    };
    if (kind === "line") def.count = "both";
    if (kind === "dwell") {
      def.dwell_frames = Math.max(1, Math.round(DEFAULT_DWELL_SECONDS * this.fps));
    }

    this.zones.push(def);
    this.tool = null;
    this.draftPoints = [];
    return copyZone(def);
  }

  // Escape: açık çizim düşer, kimlik harcanmaz.
  cancel(): void {
    this.tool = null;
    this.draftPoints = [];
  }

  // PUT sonucu: başarıda çalışma kümesi onaylanır, hatada son onaylı kümeye dönülür.
  commit(result: CommitResult): void {
    if (result.ok) {
      this.accepted = this.zones.map(copyZone);
      this.lastError = null;
      return;
    }
    this.zones = this.accepted.map(copyZone);
    this.lastError = result.error;
  }

  rename(id: string, name: string): void {
    const z = this.editable(id);
    if (z) z.name = name;
  }

  setCount(id: string, count: LineCount): void {
    const z = this.editable(id);
    if (z && z.kind === "line") z.count = count;
  }

  // Hat yönü A ↔ B: noktalar ters çevrilir.
  flipLine(id: string): void {
    const z = this.editable(id);
    if (z && z.kind === "line") z.points.reverse();
  }

  setDwellSeconds(id: string, seconds: number, fps: number): void {
    const z = this.editable(id);
    if (z && z.kind === "dwell") {
      z.dwell_frames = Math.max(1, Math.round(seconds * fps));
    }
  }

  moveVertex(id: string, index: number, p: [number, number]): void {
    const z = this.editable(id);
    if (!z || index < 0 || index >= z.points.length) return;
    z.points[index] = copyPoint(p);
  }

  // Köşe ekleme yalnız çokgende; en fazla 32 nokta.
  insertVertex(id: string, index: number, p: [number, number]): void {
    const z = this.editable(id);
    if (!z || z.kind === "line") return;
    if (z.points.length >= MAX_POINTS) return;
    if (index < 0 || index > z.points.length) return;
    z.points.splice(index, 0, copyPoint(p));
  }

  // Köşe silme yalnız çokgende; 3 noktanın altına düşmez.
  deleteVertex(id: string, index: number): void {
    const z = this.editable(id);
    if (!z || z.kind === "line") return;
    if (z.points.length <= MIN_POLYGON_POINTS) return;
    if (index < 0 || index >= z.points.length) return;
    z.points.splice(index, 1);
  }

  // Silme askıdaki bölgede de tek istisnadır.
  remove(id: string): void {
    this.zones = this.zones.filter((z) => z.id !== id);
    this.suspended.delete(id);
  }

  private editable(id: string): ZoneDef | undefined {
    if (this.suspended.has(id)) return undefined;
    return this.zones.find((z) => z.id === id);
  }
}
