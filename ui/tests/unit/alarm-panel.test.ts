// Bölge alarm defteri (operatör ekranı): motorun açık alarm listesi + kare günlüğü olaylarından
// açık alarmlar, geçmiş (en yeni önce) ve onaylanmamış sayısı. Temizlenme olayı olmadan listeden
// düşen alarm da geçmişe yazılır; cleared_frame bu güncellemedeki en büyük kare, yoksa since_frame.
import { describe, expect, it } from "vitest";

import { ZoneAlarmBook } from "@/lib/alarms";
import type { Alarm, ZoneEvent } from "@/lib/client";

const alarm = (over: Partial<Alarm> = {}): Alarm => ({
  id: "a1",
  kind: "intrusion",
  zone_id: "z1",
  track_ids: [7],
  since_n: 3,
  since_frame: 10,
  acked: false,
  ...over,
});

const event = (over: Partial<ZoneEvent> = {}): ZoneEvent => ({
  seq: 1,
  n: 1,
  frame: 20,
  kind: "alarm_clear",
  zone_id: "z1",
  zone_name: "Depo",
  zone_kind: "area",
  track_id: 7,
  dir: null,
  gen: 1,
  alarm_id: "a1",
  ...over,
});

describe("ZoneAlarmBook", () => {
  it("alarm history marks cleared without ack", () => {
    const book = new ZoneAlarmBook();
    const a = alarm();
    book.update([a], []);
    expect(book.open()).toEqual([a]);
    expect(book.history()).toEqual([]);

    // Temizlenme olayı yok, alarm listeden düştü: since_frame ile geçmişe.
    book.update([], []);
    expect(book.open()).toEqual([]);
    expect(book.history()).toEqual([{ alarm: a, cleared_frame: 10, acked: false }]);

    // Temizlenme olayı: cleared_frame olayın karesi.
    const b = alarm({ id: "b1", since_frame: 12 });
    book.update([b], []);
    book.update([], [event({ kind: "alarm_clear", alarm_id: "b1", frame: 42 })]);
    expect(book.open()).toEqual([]);
    expect(book.history()[0]).toEqual({ alarm: b, cleared_frame: 42, acked: false });
    // En yeni önce.
    expect(book.history().map((h) => h.alarm.id)).toEqual(["b1", "a1"]);

    // Olay yoksa ama bu güncellemede başka olaylar varsa: en büyük kare.
    const c = alarm({ id: "c1", since_frame: 15 });
    book.update([c], []);
    book.update([], [event({ kind: "zone_enter", alarm_id: null, frame: 50 })]);
    expect(book.history()[0]).toEqual({ alarm: c, cleared_frame: 50, acked: false });
    book.update([], [
      event({ kind: "zone_enter", alarm_id: null, frame: 51 }),
      event({ kind: "zone_exit", alarm_id: null, frame: 53 }),
    ]);
    expect(book.history()).toEqual([
      { alarm: c, cleared_frame: 50, acked: false },
      { alarm: b, cleared_frame: 42, acked: false },
      { alarm: a, cleared_frame: 10, acked: false },
    ]);
  });

  it("unacked count drops after ack", () => {
    const book = new ZoneAlarmBook();
    const a = alarm();
    const b = alarm({ id: "b1", zone_id: "z2", since_frame: 11 });
    book.update([a, b], []);
    expect(book.unacked()).toBe(2);

    book.update([a, b], [event({ kind: "alarm_ack", alarm_id: "a1", frame: 21 })]);
    expect(book.unacked()).toBe(1);
    expect(book.open().map((x) => [x.id, x.acked])).toEqual([
      ["a1", true],
      ["b1", false],
    ]);

    // Bilinmeyen kimliğe onay sayıyı değiştirmez.
    book.update([a, b], [event({ kind: "alarm_ack", alarm_id: "yok", frame: 22 })]);
    expect(book.unacked()).toBe(1);

    book.update([a, b], [event({ kind: "alarm_ack", alarm_id: "b1", frame: 23 })]);
    expect(book.unacked()).toBe(0);
  });

  it("alarm cleared after ack is marked acked", () => {
    const book = new ZoneAlarmBook();
    const a = alarm();
    book.update([a], []);
    book.update([a], [event({ kind: "alarm_ack", alarm_id: "a1", frame: 21 })]);
    expect(book.unacked()).toBe(0);

    book.update([], [event({ kind: "alarm_clear", alarm_id: "a1", frame: 30 })]);
    expect(book.history()).toHaveLength(1);
    expect(book.history()[0].alarm.id).toBe("a1");
    expect(book.history()[0].cleared_frame).toBe(30);
    expect(book.history()[0].acked).toBe(true);

    // Temizlenme olayı olmadan düşen onaylı alarm da acked kalır.
    const b = alarm({ id: "b1", since_frame: 12 });
    book.update([b], []);
    book.update([b], [event({ kind: "alarm_ack", alarm_id: "b1", frame: 40 })]);
    book.update([], []);
    expect(book.history()[0].alarm.id).toBe("b1");
    expect(book.history()[0].cleared_frame).toBe(12);
    expect(book.history()[0].acked).toBe(true);
  });

  it("reset empties the book", () => {
    const book = new ZoneAlarmBook();
    const a = alarm();
    const b = alarm({ id: "b1", since_frame: 11 });
    book.update([a, b], []);
    book.update([b], [event({ kind: "alarm_clear", alarm_id: "a1", frame: 30 })]);
    expect(book.open()).toEqual([b]);
    expect(book.history()).toHaveLength(1);
    expect(book.unacked()).toBe(1);

    book.reset();
    expect(book.open()).toEqual([]);
    expect(book.history()).toEqual([]);
    expect(book.unacked()).toBe(0);

    // Sıfırdan başlar: eski alarm geçmişe taşınmaz.
    book.update([], []);
    expect(book.history()).toEqual([]);
  });
});
