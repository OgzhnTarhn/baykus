// Bölge senkronu (Faz 4 §5.8): her düzenleme motora PUT ile gider; aynı anda tek PUT uçar,
// uçuş sırasındaki düzenlemeler tek bir takip PUT'una birleşir. Motor reddederse taslak son
// kabul edilen kümeye döner ve hata ekranda tutulur.
import { describe, expect, it, vi } from "vitest";

import type { ZoneDef } from "@/lib/client";
import { ZoneDraft } from "@/lib/editor";
import { ZoneSync } from "@/lib/zone-sync";
import { ZONE_KIND_TEXT } from "@/lib/texts";

type PutResult = { ok: true; zones: ZoneDef[] } | { ok: false; error: string };

type Pending = {
  defs: ZoneDef[];
  resolve: (r: PutResult) => void;
  reject: (e: unknown) => void;
};

// Elle çözülen sahte PUT: her çağrı bir bekleyen kayıt bırakır.
function fakePut() {
  const calls: Pending[] = [];
  const put = (defs: ZoneDef[]): Promise<PutResult> =>
    new Promise<PutResult>((resolve, reject) => {
      calls.push({ defs, resolve, reject });
    });
  return { calls, put };
}

// Zamanlayıcı mikro/makro görev farkını yut: PUT planlaması ne zaman olursa olsun görülsün.
function tick(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

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

function setup(zones: ZoneDef[] = [zone("z1")]) {
  const draft = new ZoneDraft(30);
  draft.load(zones);
  const { calls, put } = fakePut();
  const onChange = vi.fn();
  const sync = new ZoneSync(draft, put, onChange);
  return { draft, calls, onChange, sync };
}

describe("zone sync (Faz 4)", () => {
  it("edit sends one put with current defs", async () => {
    const { draft, calls, onChange, sync } = setup();

    sync.edit((d: ZoneDraft) => d.rename("z1", "kuzey kapı"));
    expect(onChange).toHaveBeenCalledTimes(1);
    // düzenleme taslağa eşzamanlı uygulanır
    expect(draft.defs()[0].name).toBe("kuzey kapı");

    await tick();
    expect(calls).toHaveLength(1);
    expect(calls[0].defs).toEqual([zone("z1", { name: "kuzey kapı" })]);
    expect(sync.busy()).toBe(true);

    calls[0].resolve({ ok: true, zones: calls[0].defs });
    await tick();
    await sync.idle();
    expect(sync.busy()).toBe(false);
    expect(calls).toHaveLength(1);
  });

  it("edits during a put are coalesced into one follow-up", async () => {
    const { calls, onChange, sync } = setup();

    sync.edit((d: ZoneDraft) => d.rename("z1", "a"));
    await tick();
    expect(calls).toHaveLength(1);

    // uçuş sırasında iki düzenleme: tek takip PUT'u, en son tanımlarla
    sync.edit((d: ZoneDraft) => d.rename("z1", "b"));
    sync.edit((d: ZoneDraft) => d.rename("z1", "c"));
    await tick();
    expect(calls).toHaveLength(1);
    expect(sync.busy()).toBe(true);

    calls[0].resolve({ ok: true, zones: calls[0].defs });
    await tick();
    expect(calls).toHaveLength(2);
    expect(calls[1].defs).toEqual([zone("z1", { name: "c" })]);

    calls[1].resolve({ ok: true, zones: calls[1].defs });
    await tick();
    await sync.idle();
    expect(sync.busy()).toBe(false);
    expect(calls).toHaveLength(2);
    // her düzenleme ve her commit için bir bildirim
    expect(onChange).toHaveBeenCalledTimes(5);
  });

  it("pending edit survives the first reply", async () => {
    const { draft, calls, sync } = setup([zone("z1"), zone("z2")]);

    sync.edit((d: ZoneDraft) => d.rename("z1", "a"));
    await tick();
    expect(calls).toHaveLength(1);

    // ilk PUT uçarken gelen düzenleme takip PUT'una girer
    sync.edit((d: ZoneDraft) => d.rename("z2", "b"));

    calls[0].resolve({ ok: true, zones: calls[0].defs });
    await tick();
    // yanıt bekleyen düzenlemeyi ezmemeli
    expect(draft.defs()).toEqual([
      zone("z1", { name: "a" }),
      zone("z2", { name: "b" }),
    ]);
    expect(calls).toHaveLength(2);
    expect(calls[1].defs).toEqual([
      zone("z1", { name: "a" }),
      zone("z2", { name: "b" }),
    ]);

    // takip PUT'u uçarken yeni düzenleme: bir sonraki PUT'a taşınır
    sync.edit((d: ZoneDraft) => d.rename("z1", "c"));
    calls[1].resolve({ ok: true, zones: calls[1].defs });
    await tick();
    expect(calls).toHaveLength(3);
    expect(calls[2].defs).toEqual([
      zone("z1", { name: "c" }),
      zone("z2", { name: "b" }),
    ]);

    calls[2].resolve({ ok: true, zones: calls[2].defs });
    await sync.idle();
    expect(draft.defs()).toEqual([
      zone("z1", { name: "c" }),
      zone("z2", { name: "b" }),
    ]);
  });

  it("open drawing survives a reply", async () => {
    const { draft, calls, sync } = setup();

    sync.edit((d: ZoneDraft) => d.rename("z1", "a"));
    await tick();
    expect(calls).toHaveLength(1);

    // yanıt gelmeden açılan çizim yanıtla kapanmamalı
    draft.start("area", 1);
    draft.addPoint([1, 1]);

    calls[0].resolve({ ok: true, zones: calls[0].defs });
    await tick();

    expect(draft.drawing().points).toEqual([[1, 1]]);
    expect(draft.drawing().tool).toBe("area");
  });

  it("failed put drops the pending edit built on it", async () => {
    const { draft, calls, sync } = setup([zone("z1"), zone("z2")]);

    sync.edit((d: ZoneDraft) => d.rename("z1", "a"));
    await tick();
    expect(calls).toHaveLength(1);

    // reddedilen PUT'un üstüne binen düzenleme takip PUT'u doğurmamalı
    sync.edit((d: ZoneDraft) => d.rename("z2", "b"));

    calls[0].resolve({ ok: false, error: "x" });
    await tick();
    expect(calls).toHaveLength(1);
    await sync.idle();

    expect(draft.defs()).toEqual([zone("z1"), zone("z2")]);
    expect(draft.error()).toBe("x");
    expect(sync.busy()).toBe(false);
  });

  it("failed put restores accepted zones and exposes the error", async () => {
    const { draft, calls, onChange, sync } = setup();

    sync.edit((d: ZoneDraft) => d.rename("z1", "yeni ad"));
    await tick();
    expect(calls).toHaveLength(1);

    calls[0].resolve({ ok: false, error: "bölge çakışıyor" });
    await tick();
    await sync.idle();

    expect(draft.defs()).toEqual([zone("z1")]);
    expect(draft.error()).toBe("bölge çakışıyor");
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(sync.busy()).toBe(false);
  });

  it("network error counts as failure", async () => {
    const { draft, calls, onChange, sync } = setup();

    sync.edit((d: ZoneDraft) => d.rename("z1", "yeni ad"));
    await tick();
    expect(calls).toHaveLength(1);

    calls[0].reject(new Error("bağlantı koptu"));
    await tick();
    await sync.idle();

    expect(draft.defs()).toEqual([zone("z1")]);
    expect(draft.error()).toBe("bağlantı koptu");
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(sync.busy()).toBe(false);
  });

  it("ok result reloads zones from the engine", async () => {
    const { draft, calls, sync } = setup();

    sync.edit((d: ZoneDraft) => d.rename("z1", "yerel ad"));
    await tick();
    expect(calls).toHaveLength(1);

    // motor kimliği ve durumu kendi yanıtıyla tazeler
    calls[0].resolve({
      ok: true,
      zones: [zone("z1", { name: "motor adı", status: "active" })],
    });
    await tick();
    await sync.idle();

    expect(draft.defs()).toEqual([zone("z1", { name: "motor adı" })]);
    expect(draft.defs()[0].status).toBeUndefined();
    expect(draft.error()).toBeNull();
    expect(sync.busy()).toBe(false);
  });
});
