// Bölge editörü (Faz 4 §5.8): çizim, köşe düzenleme, onaya gönderim. Motor 400 ile geri
// alınan değişiklikte çalışma kümesi son onaylı kümeye döner ve hata ekranda tutulur.
import { describe, expect, it } from "vitest";

import type { ZoneDef } from "@/lib/client";
import { ZoneDraft } from "@/lib/editor";
import { ZONE_KIND_TEXT } from "@/lib/texts";

function zone(id: string, over: Partial<ZoneDef> = {}): ZoneDef {
  return {
    id,
    name: `${ZONE_KIND_TEXT.area} ${id.slice(1)}`,
    kind: "area",
    gen: 1,
    points: [
      [0, 0],
      [10, 0],
      [10, 10],
    ],
    ...over,
  };
}

function points(d: ZoneDraft, id: string): [number, number][] | undefined {
  return d.defs().find((z) => z.id === id)?.points;
}

describe("zone editor (Faz 4)", () => {
  it("line finishes at two points", () => {
    const d = new ZoneDraft(30);
    d.start("line", 2);
    expect(d.drawing()).toEqual({ tool: "line", points: [] });

    // birinci nokta çizimi açık bırakır
    expect(d.addPoint([0, 0])).toBeNull();
    expect(d.drawing()).toEqual({ tool: "line", points: [[0, 0]] });

    // ikinci nokta hattı otomatik bitirir
    const line = d.addPoint([40, 10]);
    expect(line).toMatchObject({ id: "z1", kind: "line", gen: 2, count: "both" });
    expect(line?.points).toEqual([
      [0, 0],
      [40, 10],
    ]);
    expect(line?.name).toBe(`${ZONE_KIND_TEXT.line} 1`);
    expect(line?.dwell_frames).toBeUndefined();
    expect(d.drawing()).toEqual({ tool: null, points: [] });
    expect(d.defs()).toEqual([line]);

    // Enter ile tekrar bitirmek yeni bölge eklemez
    expect(d.finish()).toBeNull();
    expect(d.defs()).toHaveLength(1);

    // sayım yönü ve hat yönü (A ↔ B) düzenlenebilir
    d.setCount("z1", "a_to_b");
    expect(d.defs()[0].count).toBe("a_to_b");
    d.flipLine("z1");
    expect(points(d, "z1")).toEqual([
      [40, 10],
      [0, 0],
    ]);
    d.flipLine("z1");
    expect(points(d, "z1")).toEqual([
      [0, 0],
      [40, 10],
    ]);
  });

  it("repeated point from a double click is added once", () => {
    const d = new ZoneDraft(30);
    d.start("area", 1);
    d.addPoint([0, 0]);
    d.addPoint([10, 0]);
    d.addPoint([10, 10]);
    d.addPoint([10, 10]);
    expect(d.drawing().points).toEqual([
      [0, 0],
      [10, 0],
      [10, 10],
    ]);
    d.start("line", 1);
    d.addPoint([5, 5]);
    expect(d.addPoint([5, 5])).toBeNull();
    expect(d.drawing().points).toEqual([[5, 5]]);
  });

  it("polygon needs three points", () => {
    const d = new ZoneDraft(30);
    d.start("area", 1);
    d.addPoint([0, 0]);
    d.addPoint([10, 0]);

    // iki nokta yetersiz: bölge oluşmaz, çizim açık kalır
    expect(d.finish()).toBeNull();
    expect(d.drawing()).toEqual({ tool: "area", points: [[0, 0], [10, 0]] });
    expect(d.defs()).toEqual([]);

    d.addPoint([10, 10]);
    const poly = d.finish();
    expect(poly).toMatchObject({ id: "z1", kind: "area", gen: 1 });
    expect(poly?.points).toEqual([
      [0, 0],
      [10, 0],
      [10, 10],
    ]);
    expect(poly?.name).toBe(`${ZONE_KIND_TEXT.area} 1`);
    expect(poly?.dwell_frames).toBeUndefined();
    expect(poly?.count).toBeUndefined();
    expect(d.drawing()).toEqual({ tool: null, points: [] });
    expect(d.defs()).toEqual([poly]);

    // yasak bölge de çokgen: iki noktayla bitmez
    d.start("restricted", 4);
    d.addPoint([20, 20]);
    d.addPoint([30, 20]);
    expect(d.finish()).toBeNull();
    expect(d.drawing().tool).toBe("restricted");
    expect(d.defs()).toHaveLength(1);
    d.cancel();
  });

  it("escape cancels draft", () => {
    const d = new ZoneDraft(30);
    d.load([zone("z1")]);

    d.start("restricted", 1);
    d.addPoint([0, 0]);
    d.addPoint([5, 0]);
    d.addPoint([5, 5]);
    const before = d.defs();

    d.cancel();
    // açık çizim düşer, çalışma kümesi değişmez
    expect(d.drawing()).toEqual({ tool: null, points: [] });
    expect(d.defs()).toEqual(before);
    expect(d.defs()).toHaveLength(1);

    // iptal edilen çizim kimlik harcamaz
    d.start("area", 1);
    d.addPoint([0, 0]);
    d.addPoint([1, 0]);
    d.addPoint([1, 1]);
    expect(d.finish()?.id).toBe("z2");
    expect(d.defs()).toHaveLength(2);
  });

  it("failed commit restores last accepted zones and keeps error", () => {
    const d = new ZoneDraft(30);
    d.load([zone("z1")]);

    // çalışma kümesinde yerel değişiklik
    d.rename("z1", "Giriş");
    d.start("area", 1);
    d.addPoint([0, 0]);
    d.addPoint([1, 0]);
    d.addPoint([1, 1]);
    d.finish();
    expect(d.defs().map((z) => z.id)).toEqual(["z1", "z2"]);
    expect(d.error()).toBeNull();

    // motor 400 iletti: yerel değişiklik geri alınır, hata tutulur
    d.commit({ ok: false, error: "kendini kesen çokgen" });
    expect(d.error()).toBe("kendini kesen çokgen");
    expect(d.defs()).toEqual([zone("z1")]);
    expect(d.defs()[0].name).toBe(`${ZONE_KIND_TEXT.area} 1`);

    // sonraki onay hatayı temizler ve çalışma kümesini onaylar
    d.rename("z1", "Giriş");
    d.commit({ ok: true });
    expect(d.error()).toBeNull();
    expect(d.defs().map((z) => z.name)).toEqual(["Giriş"]);

    // başarısız onay son onaylı kümeye döner (ad geri alınır)
    d.rename("z1", "Çıkış");
    d.commit({ ok: false, error: "geçersiz bölge" });
    expect(d.error()).toBe("geçersiz bölge");
    expect(d.defs().map((z) => z.name)).toEqual(["Giriş"]);
  });

  it("dwell seconds round to frames", () => {
    const d = new ZoneDraft(30);
    d.start("dwell", 1);
    d.addPoint([0, 0]);
    d.addPoint([10, 0]);
    d.addPoint([10, 10]);
    const dwell = d.finish();
    // yeni bekleme sınırı: 30 kare/sn * 30 sn varsayılanı
    expect(dwell?.dwell_frames).toBe(900);
    expect(dwell?.name).toBe(`${ZONE_KIND_TEXT.dwell} 1`);

    d.setDwellSeconds("z1", 2.5, 30);
    expect(d.defs()[0].dwell_frames).toBe(75);
    // yuvarlama: 0,4 * 30 = 12
    d.setDwellSeconds("z1", 0.4, 30);
    expect(d.defs()[0].dwell_frames).toBe(12);
    // yuvarlama: 1,234 * 30 = 37,02 → 37
    d.setDwellSeconds("z1", 1.234, 30);
    expect(d.defs()[0].dwell_frames).toBe(37);
    // fps ile kareye dönüşür: 1,5 * 25 = 37,5 → 38
    d.setDwellSeconds("z1", 1.5, 25);
    expect(d.defs()[0].dwell_frames).toBe(38);
    // alt sınır 1 kare
    d.setDwellSeconds("z1", 0, 30);
    expect(d.defs()[0].dwell_frames).toBe(1);
    d.setDwellSeconds("z1", 0.001, 30);
    expect(d.defs()[0].dwell_frames).toBe(1);

    // yalnız bekleme sınırı taşır
    d.start("area", 1);
    d.addPoint([0, 0]);
    d.addPoint([1, 0]);
    d.addPoint([1, 1]);
    const area = d.finish();
    expect(area?.dwell_frames).toBeUndefined();
  });

  it("ids are never reused", () => {
    const d = new ZoneDraft(30);
    // sayaç en büyük z<n> sonrasından devam eder
    d.load([zone("z2"), zone("z7"), zone("z3")]);

    d.start("area", 1);
    d.addPoint([0, 0]);
    d.addPoint([1, 0]);
    d.addPoint([1, 1]);
    expect(d.finish()?.id).toBe("z8");
    expect(d.defs().map((z) => z.id)).toEqual(["z2", "z7", "z3", "z8"]);

    // silinen kimlik geri gelmez
    d.remove("z8");
    d.remove("z2");
    expect(d.defs().map((z) => z.id)).toEqual(["z7", "z3"]);

    d.start("line", 1);
    d.addPoint([0, 0]);
    expect(d.addPoint([2, 2])?.id).toBe("z9");
    d.start("restricted", 1);
    d.addPoint([0, 0]);
    d.addPoint([1, 0]);
    d.addPoint([1, 1]);
    expect(d.finish()?.id).toBe("z10");

    // adlar kimlik numarasını taşır
    expect(d.defs().map((z) => z.name)).toEqual([
      `${ZONE_KIND_TEXT.area} 7`,
      `${ZONE_KIND_TEXT.area} 3`,
      `${ZONE_KIND_TEXT.line} 9`,
      `${ZONE_KIND_TEXT.restricted} 10`,
    ]);

    // onaydan sonra da kimlikler yeniden başlamaz
    d.commit({ ok: true });
    d.start("area", 1);
    d.addPoint([0, 0]);
    d.addPoint([1, 0]);
    d.addPoint([1, 1]);
    expect(d.finish()?.id).toBe("z11");
  });

  it("suspended zones can only be deleted", () => {
    const d = new ZoneDraft(30);
    d.load([
      zone("z1", { status: "suspended" }),
      zone("z2", { status: "active" }),
    ]);
    expect(d.isSuspended("z1")).toBe(true);
    expect(d.isSuspended("z2")).toBe(false);
    expect(d.isSuspended("z9")).toBe(false);

    // askıda bölge: düzenleme yok
    d.rename("z1", "Yeni ad");
    d.moveVertex("z1", 0, [5, 5]);
    d.insertVertex("z1", 1, [5, 5]);
    d.deleteVertex("z1", 0);
    d.setDwellSeconds("z1", 5, 30);
    d.setCount("z1", "a_to_b");
    d.flipLine("z1");
    // durum yalnız GET yanıtında; tanım kümesinde yok
    expect(d.defs()[0]).toEqual(zone("z1"));

    // aktif bölge düzenlenebilir
    d.rename("z2", "Giriş");
    expect(d.defs()[1].name).toBe("Giriş");

    // silme tek istisna
    d.remove("z1");
    expect(d.defs().map((z) => z.id)).toEqual(["z2"]);
    expect(d.isSuspended("z1")).toBe(false);

    // askıda durum PUT tanımına girmez
    d.load([zone("z1", { status: "suspended" })]);
    expect(d.defs()).toEqual([zone("z1")]);
  });

  it("vertex edits keep polygons valid", () => {
    const d = new ZoneDraft(30);
    d.load([
      zone("z1", {
        points: [
          [0, 0],
          [10, 0],
          [10, 10],
          [0, 10],
        ],
      }),
    ]);
    const square: [number, number][] = [
      [0, 0],
      [20, 0],
      [20, 20],
      [0, 20],
    ];
    d.moveVertex("z1", 1, [20, 0]);
    d.moveVertex("z1", 2, [20, 20]);
    d.moveVertex("z1", 3, [0, 20]);
    d.moveVertex("z1", 0, [0, 0]);
    expect(points(d, "z1")).toEqual(square);

    // köşe ekleme: i. indeksin önüne
    d.insertVertex("z1", 2, [15, 5]);
    expect(points(d, "z1")).toEqual([
      [0, 0],
      [20, 0],
      [15, 5],
      [20, 20],
      [0, 20],
    ]);
    d.insertVertex("z1", 0, [-5, -5]);
    expect(points(d, "z1")).toEqual([
      [-5, -5],
      [0, 0],
      [20, 0],
      [15, 5],
      [20, 20],
      [0, 20],
    ]);

    // köşe silme ama çokgen 3 noktanın altına düşmez
    d.deleteVertex("z1", 0);
    d.deleteVertex("z1", 2);
    expect(points(d, "z1")).toEqual(square);
    d.deleteVertex("z1", 0);
    expect(points(d, "z1")).toEqual([
      [20, 0],
      [20, 20],
      [0, 20],
    ]);
    // alt sınır 3 nokta: silme artık yok
    d.deleteVertex("z1", 0);
    d.deleteVertex("z1", 2);
    expect(points(d, "z1")).toEqual([
      [20, 0],
      [20, 20],
      [0, 20],
    ]);

    // çokgende en fazla 32 nokta
    const many: [number, number][] = [];
    for (let i = 0; i < 32; i++) many.push([i, 0]);
    d.load([zone("z2", { points: many })]);
    d.insertVertex("z2", 1, [99, 99]);
    expect(d.defs()[0].points).toHaveLength(32);
    expect(points(d, "z2")).toEqual(many);
    d.insertVertex("z2", 32, [50, 50]);
    expect(d.defs()[0].points).toHaveLength(32);

    // hatta köşe düzenlenir ama eklenip silinmez
    d.load([
      zone("z3", {
        kind: "line",
        name: `${ZONE_KIND_TEXT.line} 3`,
        count: "both",
        points: [
          [0, 0],
          [10, 10],
        ],
      }),
    ]);
    d.moveVertex("z3", 1, [20, 20]);
    expect(points(d, "z3")).toEqual([
      [0, 0],
      [20, 20],
    ]);
    d.insertVertex("z3", 1, [5, 5]);
    d.deleteVertex("z3", 0);
    expect(points(d, "z3")).toEqual([
      [0, 0],
      [20, 20],
    ]);

    // bilinmeyen kimlik hiçbir şey yapmaz
    d.moveVertex("z99", 0, [1, 1]);
    d.insertVertex("z99", 0, [1, 1]);
    d.deleteVertex("z99", 0);
    d.rename("z99", "yok");
    expect(d.defs()).toHaveLength(1);
    expect(points(d, "z3")).toEqual([
      [0, 0],
      [20, 20],
    ]);
  });
});
