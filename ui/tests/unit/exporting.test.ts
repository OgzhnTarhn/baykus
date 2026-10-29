// Dışa aktarma (Faz 4 §5.13): olay günlüğü JSON'u meta + tüm olayları taşır, CSV motorun
// metnini BOM ile korur, dosya adları yerel saatte sıfır dolgulu ve kararlıdır.
import { describe, expect, it } from "vitest";

import type { ZoneDef, ZoneEvent } from "@/lib/client";
import { csvBlob, eventsJson, excelCsv, fileName, type ExportMeta } from "@/lib/exporting";

function zone(id: string, over: Partial<ZoneDef> = {}): ZoneDef {
  return {
    id,
    name: `Bölge ${id.slice(1)}`,
    kind: "area",
    gen: 1,
    points: [
      [10, 20],
      [30, 40],
      [50, 60],
    ],
    ...over,
  };
}

function event(seq: number, over: Partial<ZoneEvent> = {}): ZoneEvent {
  return {
    seq,
    n: seq + 100,
    frame: seq * 3,
    kind: "enter",
    zone_id: "z1",
    zone_name: "Bölge 1",
    zone_kind: "area",
    track_id: 7,
    dir: null,
    gen: 1,
    alarm_id: null,
    ...over,
  };
}

function meta(over: Partial<ExportMeta> = {}): ExportMeta {
  return {
    sequence: "kavsak-01",
    session: "s-2026-03-05",
    created_at: "2026-03-05T07:08:09Z",
    model_sha256: "a".repeat(64),
    motion_sha256: null,
    fps_assumed: 25,
    zones: [zone("z1"), zone("z2", { kind: "line", count: "both" })],
    ...over,
  };
}

describe("exporting", () => {
  it("events json carries meta and all events", async () => {
    const m = meta();
    const events = [
      event(1),
      event(2, { kind: "exit", dir: "a_to_b", track_id: null }),
      event(3, { kind: "dwell", zone_id: null, zone_name: null, zone_kind: null, alarm_id: "al-9" }),
    ];

    const blob = eventsJson(m, events);
    expect(blob.type).toBe("application/json");

    const text = await blob.text();
    // 2 boşlukla girintili (pretty-printed) gövde.
    expect(text).toContain('\n  "meta"');
    expect(text).toContain('\n  "events"');

    const body = JSON.parse(text) as { meta: ExportMeta; events: ZoneEvent[] };
    expect(body.meta).toEqual(m);
    expect(body.events).toEqual(events);
    expect(body.events).toHaveLength(3);
    expect(body.events[2].alarm_id).toBe("al-9");
    expect(body.meta.zones.map((z) => z.id)).toEqual(["z1", "z2"]);
  });

  it("file names are stable", () => {
    // Yerel saat bileşenleriyle kurulan tarih: saat diliminden bağımsız.
    const created = new Date(2026, 2, 5, 7, 8, 9);

    expect(fileName("olaylar", "kavsak-01", created, "json")).toBe(
      "baykus_kavsak-01_olaylar_20260305-070809.json",
    );
    expect(fileName("harita", "kavsak-01", created, "png")).toBe(
      "baykus_kavsak-01_harita_20260305-070809.png",
    );
    expect(fileName("olaylar", "kavsak-01", created, "csv")).toBe(
      "baykus_kavsak-01_olaylar_20260305-070809.csv",
    );

    // Aynı girdi aynı adı verir; sıfır dolgu tek haneli alanlarda da korunur.
    expect(fileName("olaylar", "kavsak-01", created, "json")).toBe(
      fileName("olaylar", "kavsak-01", new Date(2026, 2, 5, 7, 8, 9), "json"),
    );
    expect(fileName("harita", "s1", new Date(2026, 0, 2, 3, 4, 5), "png")).toBe(
      "baykus_s1_harita_20260102-030405.png",
    );
  });

  it("csv keeps engine text with a BOM", async () => {
    const csv = "seq,kind,zone_name\n1,enter,Çıkış Bölgesi\n2,exit,Şişli\n";

    const blob = csvBlob(csv);
    expect(blob.type).toBe("text/csv;charset=utf-8");

    const bytes = new Uint8Array(await blob.arrayBuffer());
    expect(Array.from(bytes.slice(0, 3))).toEqual([0xef, 0xbb, 0xbf]);

    // BOM'dan sonrası motorun CSV'siyle birebir aynı (Türkçe karakterler UTF-8).
    expect(new TextDecoder("utf-8").decode(bytes.slice(3))).toBe(csv);
    expect(await blob.text()).toBe(csv);
  });

  it("excel csv turns commas into semicolons for plain rows", () => {
    // Düz satırlar: virgül ayırıcısı noktalı virgüle döner, satır sonları korunur.
    expect(excelCsv("seq,kind,zone_name\n1,enter,Çıkış\n")).toBe(
      "seq;kind;zone_name\n1;enter;Çıkış\n",
    );
    // CRLF girişte CRLF kalır.
    expect(excelCsv("seq,kind,zone_name\r\n1,enter,Çıkış\r\n")).toBe(
      "seq;kind;zone_name\r\n1;enter;Çıkış\r\n",
    );
    // Sondaki satır sonu yoksa eklenmez.
    expect(excelCsv("seq,kind\n1,enter")).toBe("seq;kind\n1;enter");
  });

  it("excel csv keeps empty fields", () => {
    // Boş alan boş kalır (motor null -> boş alan).
    expect(excelCsv("1,,x")).toBe("1;;x");
  });

  it("excel csv unquotes fields that no longer need quoting", () => {
    // Virgül içeren tırnaklı alan, ';' / '"' / CR / LF yoksa tırnaksız kalır.
    expect(excelCsv('1,"A, B"')).toBe("1;A, B");
  });

  it("excel csv quotes fields containing a semicolon", () => {
    // Noktalı virgül içeren alan tırnaklanır.
    expect(excelCsv("1,a;b")).toBe('1;"a;b"');
  });

  it("excel csv doubles inner quotes", () => {
    // İç tırnaklar ikilenir, alan tırnaklı kalır.
    expect(excelCsv('1,"say ""hi"""')).toBe('1;"say ""hi"""');
  });

  it("excel csv keeps an embedded LF inside one quoted field", () => {
    // Gömülü LF alanı bölmez; alan tırnaklı tek parça olarak kalır.
    expect(excelCsv('1,"a\nb"\n2,c\n')).toBe('1;"a\nb"\n2;c\n');
  });

  it("excel csv maps an empty string to an empty string", () => {
    expect(excelCsv("")).toBe("");
  });

  it("excel csv file name uses the olaylar-excel kind", () => {
    // Yeni tür 'olaylar-excel' dosya adında görünür.
    expect(fileName("olaylar-excel", "s1", new Date(2026, 0, 2, 3, 4, 5), "csv")).toBe(
      "baykus_s1_olaylar-excel_20260102-030405.csv",
    );
  });

  it("excel csv blob starts with a BOM", async () => {
    // Excel CSV'si de UTF-8 BOM ile başlar.
    const blob = csvBlob(excelCsv("seq,kind\n1,enter\n"));
    const bytes = new Uint8Array(await blob.arrayBuffer());
    expect(Array.from(bytes.slice(0, 3))).toEqual([0xef, 0xbb, 0xbf]);
  });
});
