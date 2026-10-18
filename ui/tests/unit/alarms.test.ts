// Alarm sözleşmesi (Faz 3 tasarımı §4.4; plan Görev 30): Python `baykus.health.m3ot.Alarms` ile
// aynı girdide her karede aynı aktif küme. Beklenenler Görev 14 fikstürlerinden (alarm_sequences ve
// klip türleri); başlama / temizlenme olayları ve sıfırlama burada.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { ALARM_TEXT, Alarms, gateBadge, type AlarmInput } from "@/lib/alarms";

const FIXTURES = join(__dirname, "..", "..", "..", "fixtures");
const load = <T>(rel: string): T => JSON.parse(readFileSync(join(FIXTURES, rel), "utf-8")) as T;

type Step = { input: AlarmInput; expected: string[] } | { reset: true };
type AlarmSeq = { name: string; steps: Step[] };
type IndexEntry = { name: string; kind: string; path: string };
type Clip = { variant: string; frames: { frame: number; health: AlarmInput; alarms: string[] }[] };

const input = (rgb: string[], ir: string[], misaligned = false): AlarmInput => ({
  rgb: { flags: rgb },
  ir: { flags: ir },
  pair: { misaligned },
});

describe("alarms", () => {
  it("matches python alarm_states on alarm_sequences", () => {
    const seqs = load<AlarmSeq[]>("expected/alarm_sequences.json");
    expect(seqs.length).toBeGreaterThan(0);
    for (const s of seqs) {
      const a = new Alarms();
      s.steps.forEach((step, i) => {
        if ("reset" in step) {
          a.reset();
          return;
        }
        a.update(step.input, i);
        expect(a.active(), `${s.name} adım ${i}`).toEqual(step.expected);
      });
    }
  });

  it("matches python alarm_states on clip variants", () => {
    const clips = load<IndexEntry[]>("index.json").filter((e) => e.kind === "clip");
    expect(clips.length).toBeGreaterThanOrEqual(3);
    for (const entry of clips) {
      const clip = load<Clip>(entry.path);
      const a = new Alarms();
      for (const f of clip.frames) {
        a.update(f.health, f.frame);
        expect(a.active(), `${entry.name} kare ${f.frame}`).toEqual(f.alarms);
      }
    }
  });

  it("records start and clear events with frame numbers", () => {
    const a = new Alarms();
    expect(a.update(input([], ["lost"]), 10)).toEqual([]);
    expect(a.update(input([], ["lost"]), 11)).toEqual([]);
    expect(a.update(input([], ["lost"]), 12)).toEqual([{ id: "ir_lost", kind: "start", frame: 12 }]);
    a.update(input([], []), 13);
    a.update(input([], []), 14);
    expect(a.update(input([], []), 15)).toEqual([{ id: "ir_lost", kind: "clear", frame: 15 }]);
    expect(a.history()).toEqual([
      { id: "ir_lost", kind: "start", frame: 12 },
      { id: "ir_lost", kind: "clear", frame: 15 },
    ]);
  });

  it("reset clears history", () => {
    const a = new Alarms();
    for (let f = 1; f <= 3; f++) a.update(input([], ["noisy"], true), f);
    expect(a.active()).toEqual(["ir_noisy", "misaligned"]);
    a.reset();
    expect(a.active()).toEqual([]);
    expect(a.history()).toEqual([]);
    // pencereler de boş: iki kare daha gürültü yetmez
    a.update(input([], ["noisy"]), 4);
    a.update(input([], ["noisy"]), 5);
    expect(a.active()).toEqual([]);
  });

  it("multiple alarms at once", () => {
    const a = new Alarms();
    for (let f = 1; f <= 3; f++) a.update(input(["lost"], ["lost", "noisy"], true), f);
    expect(a.active()).toEqual(["rgb_lost", "ir_lost", "ir_noisy", "misaligned"]);
  });

  it("text from table not from reason", () => {
    expect(ALARM_TEXT).toEqual({
      rgb_lost: "RGB kamera kayıp",
      ir_lost: "IR kamera kayıp",
      ir_noisy: "IR gürültülü",
      misaligned: "IR hizasız",
    });
    const a = new Alarms();
    const h = { ...input([], [], true), reason: "başka metin", reason_code: "misaligned" };
    a.update(h, 1);
    expect(a.items().map((x) => x.text)).toEqual(["IR hizasız"]);
  });

  it("gate badge comes from each frame's gate and applied", () => {
    expect(gateBadge({ gate: "ir_off", applied: true })).toEqual({
      text: "IR modele verilmiyor",
      level: "applied",
    });
    expect(gateBadge({ gate: "rgb_off", applied: true })).toEqual({
      text: "RGB modele verilmiyor",
      level: "applied",
    });
    expect(gateBadge({ gate: "ir_off", applied: false })).toEqual({
      text: "yalnız uyarı",
      level: "advise",
    });
    expect(gateBadge({ gate: "none", applied: false })).toBeNull();
  });
});
