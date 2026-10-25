// Zaman çizelgesi sözleşmesi (Faz 4 §5.8, §5.13): oturum damgası, sıra tekilliği, süzgeçler,
// işaretçi ölçekleme ve alarm bantları. Modül henüz yok; testler davranışı tanımlar.
import { describe, expect, it } from "vitest";

import type { ZoneEvent } from "@/lib/client";
import { Timeline } from "@/lib/timeline";

// Modül henüz yok; dönüş tiplerini testte yazıyoruz ki dosya kendi başına tip denetiminden geçsin.
type Marker = { x: number; frame: number; seq: number; kind: string; shape: string };
type Band = { alarm_id: string; from: number; to: number | null; kind: string };

// zone_id ve zone_name aynı tutuldu: `filtered({ zone })` hangi alana baktığını bu test belirlemez.
function ev(seq: number, frame: number, kind: string, extra: Partial<ZoneEvent> = {}): ZoneEvent {
  return {
    seq,
    n: seq,
    frame,
    kind,
    zone_id: "z1",
    zone_name: "z1",
    zone_kind: "restricted",
    track_id: 1,
    dir: null,
    gen: 0,
    alarm_id: null,
    ...extra,
  };
}

describe("timeline", () => {
  it("markers scale to width and keep kinds", () => {
    const t = new Timeline();
    t.reset("s1", 1);
    t.add(
      [
        ev(1, 1, "enter"),
        ev(2, 51, "exit"),
        ev(3, 101, "exit_lost"),
        ev(4, 101, "stopped"),
        ev(5, 101, "line_cross"),
        ev(6, 101, "alarm_start"),
        ev(7, 101, "alarm_ack"),
        ev(8, 101, "alarm_clear"),
        ev(9, 101, "map_enter"),
        ev(10, 101, "other"),
      ],
      "s1",
      1,
    );

    const m = t.markers(101, 1000, 1);
    expect(m.map((x: Marker) => x.shape)).toEqual([
      "up",
      "down",
      "cross",
      "square",
      "diamond",
      "bar",
      "bar",
      "bar",
      "circle",
      "dot",
    ]);
    expect(m[0]).toEqual({ x: 0, frame: 1, seq: 1, kind: "enter", shape: "up" });
    expect(m[1].x).toBe(500);
    expect(m[2].x).toBe(1000);
    // firstFrame varsayılanı 1
    expect(t.markers(101, 1000)[0].x).toBe(0);
    // frameCount 1: max(1, 0) -> bölme yok
    expect(t.markers(1, 100)[0].x).toBe(0);
    // bütün map_* türleri daire; verilen olay listesi (süzgeç) yalnız onları işaretler
    const only = t.filtered({ kinds: ["exit"] });
    expect(t.markers(101, 1000, 1, only).map((x: Marker) => x.kind)).toEqual(["exit"]);
    t.add([ev(11, 5, "map_relocalized")], "s1", 1);
    expect(t.markers(101, 1000).find((x: Marker) => x.seq === 11)?.shape).toBe("circle");
  });

  it("filters combine kind zone and track", () => {
    const t = new Timeline();
    t.reset("s1", 1);
    t.add(
      [
        ev(1, 1, "enter", { zone_id: "z1", zone_name: "z1", track_id: 1 }),
        ev(2, 2, "exit", { zone_id: "z1", zone_name: "z1", track_id: 1 }),
        ev(3, 3, "enter", { zone_id: "z2", zone_name: "z2", track_id: 1 }),
        ev(4, 4, "enter", { zone_id: "z1", zone_name: "z1", track_id: 2 }),
      ],
      "s1",
      1,
    );

    expect(t.filtered({}).map((e: ZoneEvent) => e.seq)).toEqual([1, 2, 3, 4]);
    expect(t.filtered({ kinds: [] }).map((e: ZoneEvent) => e.seq)).toEqual([1, 2, 3, 4]);
    expect(t.filtered({ kinds: ["enter"] }).map((e: ZoneEvent) => e.seq)).toEqual([1, 3, 4]);
    expect(t.filtered({ zone: "z1" }).map((e: ZoneEvent) => e.seq)).toEqual([1, 2, 4]);
    expect(t.filtered({ track: 1 }).map((e: ZoneEvent) => e.seq)).toEqual([1, 2, 3]);
    expect(t.filtered({ kinds: ["enter"], zone: "z1", track: 1 }).map((e: ZoneEvent) => e.seq)).toEqual([1]);
    expect(t.filtered({ kinds: ["exit"], zone: "z2" })).toEqual([]);
  });

  it("alarm bands open until clear", () => {
    const t = new Timeline();
    t.reset("s1", 1);
    t.add(
      [
        ev(1, 10, "alarm_start", { alarm_id: "a1", zone_kind: "restricted" }),
        ev(2, 20, "alarm_clear", { alarm_id: "a1", zone_kind: "restricted" }),
        ev(3, 30, "alarm_start", { alarm_id: "a2", zone_kind: "dwell" }),
      ],
      "s1",
      1,
    );

    const bands: Band[] = t.alarmBands();
    expect(bands).toEqual([
      { alarm_id: "a1", from: 10, to: 20, kind: "intrusion" },
      { alarm_id: "a2", from: 30, to: null, kind: "dwell_exceeded" },
    ]);
  });

  it("restamp keeps events and drops the old stamp", () => {
    const t = new Timeline();
    t.reset("s1", 1);
    t.add([ev(1, 10, "enter")], "s1", 1);
    t.restamp("s1", 2);
    expect(t.all().map((e) => e.seq)).toEqual([1]);
    t.add([ev(2, 11, "exit")], "s1", 1);
    expect(t.all().map((e) => e.seq)).toEqual([1]);
    t.add([ev(3, 12, "exit")], "s1", 2);
    expect(t.all().map((e) => e.seq)).toEqual([1, 3]);
  });

  it("timeline ignores stale events", () => {
    const t = new Timeline();
    t.reset("s1", 1);

    // başka oturum / başka epoch: yok sayılır
    t.add([ev(1, 1, "enter")], "s2", 1);
    t.add([ev(2, 2, "enter")], "s1", 2);
    expect(t.all()).toEqual([]);

    t.add([ev(3, 3, "enter")], "s1", 1);
    expect(t.all().map((e: ZoneEvent) => e.seq)).toEqual([3]);

    // reset olayları temizler ve damgayı taşır
    t.reset("s2", 5);
    expect(t.all()).toEqual([]);
    t.add([ev(4, 4, "enter")], "s1", 1);
    expect(t.all()).toEqual([]);
    t.add([ev(5, 5, "enter")], "s2", 5);
    expect(t.all().map((e: ZoneEvent) => e.seq)).toEqual([5]);
  });

  it("duplicate seq is stored once", () => {
    const t = new Timeline();
    t.reset("s1", 1);

    t.add([ev(2, 20, "enter"), ev(1, 10, "exit")], "s1", 1);
    t.add([ev(2, 20, "enter"), ev(3, 10, "stopped")], "s1", 1);

    // kare, sonra seq sırası
    expect(t.all().map((e: ZoneEvent) => e.seq)).toEqual([1, 3, 2]);
    expect(t.all().length).toBe(3);
  });
});
