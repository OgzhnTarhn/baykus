// Oynatıcı durum makinesi ve oturum kuralı (Faz 3 tasarımı §4.4 akış kontrolü; plan Görev 29).
// Motor yerine elle çözülen sözler: aynı anda kaç istek olduğu ve hangi karelerin istendiği görülür.
import { describe, expect, it } from "vitest";

import { createPlayer, type PlayerDeps } from "@/lib/player";
import { newSessionId, shouldReset } from "@/lib/session";

type Call = { seq: string; frame: number; resolve: (v: string) => void; reject: (e: Error) => void };

function harness(opts: { autoResolve?: boolean } = {}) {
  const calls: Call[] = [];
  const log: string[] = [];
  let inFlight = 0;
  let maxInFlight = 0;
  let clock = 0;
  const deps: PlayerDeps<string> = {
    detect: (seq, frame) => {
      log.push(`detect ${seq} ${frame}`);
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      return new Promise<string>((resolve, reject) => {
        const call: Call = {
          seq,
          frame,
          resolve: (v) => {
            inFlight -= 1;
            resolve(v);
          },
          reject: (e) => {
            inFlight -= 1;
            reject(e);
          },
        };
        calls.push(call);
        if (opts.autoResolve) call.resolve(`r${frame}`);
      });
    },
    reset: async () => {
      log.push("reset");
    },
    sleep: async (ms) => {
      clock += ms;
    },
    now: () => clock,
    onChange: () => {},
  };
  return {
    deps,
    calls,
    log,
    stats: () => ({ inFlight, maxInFlight }),
  };
}

// Mikro görevleri boşalt (sözlerin zinciri ilerlesin).
const flush = async () => {
  for (let i = 0; i < 20; i++) await Promise.resolve();
};

describe("session", () => {
  it("new session ids are uuids and differ", () => {
    const a = newSessionId();
    expect(a).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    expect(newSessionId()).not.toBe(a);
  });

  it("resets on sequence change, jump, back-step, repeat and loop start", () => {
    expect(shouldReset(null, { seq: "a", index: 0 })).toBe(false);
    expect(shouldReset({ seq: "a", index: 3 }, { seq: "a", index: 4 })).toBe(false);
    expect(shouldReset({ seq: "a", index: 3 }, { seq: "b", index: 4 })).toBe(true);
    expect(shouldReset({ seq: "a", index: 3 }, { seq: "a", index: 5 })).toBe(true);
    expect(shouldReset({ seq: "a", index: 3 }, { seq: "a", index: 2 })).toBe(true);
    expect(shouldReset({ seq: "a", index: 3 }, { seq: "a", index: 3 })).toBe(true);
    expect(shouldReset({ seq: "a", index: 14 }, { seq: "a", index: 0 })).toBe(true);
  });
});

describe("player", () => {
  it("one request in flight", async () => {
    const h = harness();
    const p = createPlayer(h.deps);
    p.load("a", [1, 2, 3, 4]);
    p.play();
    p.step();
    p.play();
    await flush();
    expect(h.calls).toHaveLength(1);
    h.calls[0].resolve("r1");
    await flush();
    expect(h.calls).toHaveLength(2);
    expect(h.stats().maxInFlight).toBe(1);
    expect(p.snapshot().frame).toBe(1);
    expect(p.snapshot().result).toBe("r1");
  });

  it("no frame skipped at high speed", async () => {
    const h = harness();
    const p = createPlayer(h.deps);
    p.setSpeed(1000);
    p.load("a", [10, 11, 12, 13, 14]);
    p.play();
    for (let i = 0; i < 5; i++) {
      await flush();
      h.calls[i].resolve(`r${i}`);
    }
    await flush();
    expect(h.calls.slice(0, 5).map((c) => c.frame)).toEqual([10, 11, 12, 13, 14]);
    expect(h.stats().maxInFlight).toBe(1);
  });

  it("speed is an upper bound: waits the rest of the frame interval", async () => {
    const h = harness({ autoResolve: true });
    const sleeps: number[] = [];
    h.deps.sleep = async (ms) => {
      sleeps.push(ms);
    };
    const p = createPlayer(h.deps);
    p.setSpeed(10);
    p.load("a", [1, 2, 3]);
    p.play();
    await flush();
    p.pause();
    await flush();
    expect(sleeps[0]).toBe(100);
  });

  it("reset on seek, back-step and loop", async () => {
    const h = harness({ autoResolve: true });
    const p = createPlayer(h.deps);
    p.load("a", [1, 2, 3]);
    p.step();
    await flush();
    p.step();
    await flush();
    expect(h.log).toEqual(["detect a 1", "detect a 2"]);

    p.seek(0); // geri adım
    await flush();
    expect(h.log.slice(2)).toEqual(["reset", "detect a 1"]);

    p.seek(2); // ileri atlama > 1
    await flush();
    expect(h.log.slice(4)).toEqual(["reset", "detect a 3"]);

    p.step(); // dizi sonu → başa döner
    await flush();
    expect(h.log.slice(6)).toEqual(["reset", "detect a 1"]);
    expect(p.snapshot().frame).toBe(1);
  });

  it("loops while playing and resets at the loop start", async () => {
    const h = harness({ autoResolve: true });
    const p = createPlayer(h.deps);
    p.load("a", [1, 2]);
    p.play();
    await flush();
    p.pause();
    await flush();
    expect(h.log.slice(0, 4)).toEqual(["detect a 1", "detect a 2", "reset", "detect a 1"]);
  });

  it("reset on sequence change", async () => {
    const h = harness({ autoResolve: true });
    const p = createPlayer(h.deps);
    p.load("a", [1, 2]);
    p.step();
    await flush();
    p.load("b", [1, 2]);
    p.step();
    await flush();
    expect(h.log).toEqual(["detect a 1", "reset", "detect b 1"]);
  });

  it("notifies reset listeners", async () => {
    const h = harness({ autoResolve: true });
    let resets = 0;
    const p = createPlayer({ ...h.deps, onReset: () => (resets += 1) });
    p.load("a", [1, 2, 3]);
    p.step();
    await flush();
    p.seek(2);
    await flush();
    expect(resets).toBe(1);
  });

  it("manual reset runs before the next request, never beside one", async () => {
    const h = harness();
    let resets = 0;
    const p = createPlayer({ ...h.deps, onReset: () => (resets += 1) });
    p.load("a", [1, 2, 3]);
    p.step();
    await flush();
    p.requestReset(); // istek yolda: sıfırlama bekler
    await flush();
    expect(h.log).toEqual(["detect a 1"]);
    h.calls[0].resolve("r1");
    await flush();
    p.step();
    await flush();
    expect(h.log).toEqual(["detect a 1", "reset", "detect a 2"]);
    expect(resets).toBe(1);
    h.calls[1].resolve("r2");
    await flush();

    p.requestReset(); // boşta: hemen; sonraki kare yeniden sıfırlamaz
    await flush();
    expect(h.log.slice(3)).toEqual(["reset"]);
    expect(resets).toBe(2);
    p.step();
    await flush();
    h.calls[2].resolve("r3");
    await flush();
    expect(h.log.slice(3)).toEqual(["reset", "detect a 3"]);
  });

  it("seek while busy waits for the request in flight", async () => {
    const h = harness();
    const p = createPlayer(h.deps);
    p.load("a", [1, 2, 3, 4]);
    p.step();
    await flush();
    p.seek(3);
    await flush();
    expect(h.calls).toHaveLength(1);
    h.calls[0].resolve("r1");
    await flush();
    expect(h.calls.map((c) => c.frame)).toEqual([1, 4]);
    expect(h.stats().maxInFlight).toBe(1);
  });

  it("engine error pauses", async () => {
    const h = harness();
    const p = createPlayer(h.deps);
    p.load("a", [1, 2, 3]);
    p.play();
    await flush();
    h.calls[0].reject(new Error("motor yanıt vermiyor"));
    await flush();
    const s = p.snapshot();
    expect(s.state).toBe("paused");
    expect(s.error).toBe("motor yanıt vermiyor");
    expect(s.busy).toBe(false);
    await flush();
    expect(h.calls).toHaveLength(1); // sonsuz döngü yok
  });

  it("error is cleared by the next successful frame", async () => {
    const h = harness();
    const p = createPlayer(h.deps);
    p.load("a", [1, 2, 3]);
    p.step();
    await flush();
    h.calls[0].reject(new Error("motor yanıt vermiyor"));
    await flush();
    p.step();
    await flush();
    h.calls[1].resolve("r1");
    await flush();
    expect(p.snapshot().error).toBeNull();
    expect(p.snapshot().frame).toBe(1);
  });
});

// İnceleme bulguları (Ö1, Ö2, Ö5, S2): kareler arası bekleme elle çözülen sözle.
function manualSleep(h: ReturnType<typeof harness>) {
  const wakes: (() => void)[] = [];
  h.deps.sleep = () => new Promise<void>((r) => wakes.push(r));
  return wakes;
}

describe("player review fixes", () => {
  it("seek during the frame wait does not start a second loop", async () => {
    const h = harness({ autoResolve: true });
    const wakes = manualSleep(h);
    const p = createPlayer(h.deps);
    p.load("a", [1, 2, 3, 4, 5, 6, 7, 8]);
    p.play();
    await flush();
    expect(h.log).toEqual(["detect a 1"]);
    p.seek(4); // döngü beklemede
    await flush();
    expect(h.log).toEqual(["detect a 1"]);
    wakes[0]();
    await flush();
    expect(h.log).toEqual(["detect a 1", "reset", "detect a 5"]);
    expect(wakes).toHaveLength(2); // tek döngü: bir bekleme daha, ikinci döngü yok
    wakes[1]();
    await flush();
    expect(h.log.slice(3)).toEqual(["detect a 6"]);
    p.pause();
  });

  it("a stale response after a sequence change keeps playing the new sequence", async () => {
    const h = harness();
    const p = createPlayer(h.deps);
    p.load("a", [1, 2]);
    p.step();
    await flush();
    p.load("b", [7, 8]);
    p.play(); // istek yolda
    h.calls[0].resolve("old");
    await flush();
    expect(h.log).toEqual(["detect a 1", "reset", "detect b 7"]);
    expect(p.snapshot().state).toBe("playing");
    expect(p.snapshot().result).toBeNull(); // eski dizinin yanıtı gösterilmez
    p.pause();
  });

  it("after an error the next request resets the session", async () => {
    const h = harness();
    const p = createPlayer(h.deps);
    p.load("a", [1, 2, 3]);
    p.step();
    await flush();
    h.calls[0].resolve("r1");
    await flush();
    p.step();
    await flush();
    h.calls[1].reject(new Error("kare görüntüsü yüklenemedi")); // motor kareyi işlemiş olabilir
    await flush();
    p.step();
    await flush();
    expect(h.log).toEqual(["detect a 1", "detect a 2", "reset", "detect a 2"]);
  });

  it("reset pressed after pausing during the frame wait is sent", async () => {
    const h = harness({ autoResolve: true });
    const wakes = manualSleep(h);
    const p = createPlayer(h.deps);
    p.load("a", [1, 2, 3]);
    p.play();
    await flush();
    p.pause();
    p.requestReset();
    wakes[0]();
    await flush();
    expect(h.log).toEqual(["detect a 1", "reset"]);
  });
});

// İkinci inceleme (docs/reviews/2026-10-19-faz3-kod-incelemesi.md Ö1): tek başına sıfırlama sürerken
// gelen oynat / atlama sıfırlama bitince uygulanır.
function manualReset(h: ReturnType<typeof harness>) {
  const done: (() => void)[] = [];
  h.deps.reset = () =>
    new Promise<void>((r) => {
      h.log.push("reset");
      done.push(r);
    });
  return done;
}

describe("player second review fixes", () => {
  it("play pressed during a standalone reset starts playback", async () => {
    const h = harness();
    const done = manualReset(h);
    const p = createPlayer(h.deps);
    p.load("a", [1, 2, 3]);
    p.requestReset();
    p.play();
    done[0]();
    await flush();
    expect(h.log).toEqual(["reset", "detect a 1"]);
    expect(p.snapshot().state).toBe("playing");
    p.pause();
  });

  it("seek during a standalone reset is applied once", async () => {
    const h = harness({ autoResolve: true });
    const done = manualReset(h);
    const p = createPlayer(h.deps);
    p.load("a", [1, 2, 3, 4]);
    p.requestReset();
    p.seek(2);
    done[0]();
    await flush();
    expect(h.log).toEqual(["reset", "detect a 3"]);
    p.step();
    await flush();
    expect(h.log.slice(2)).toEqual(["detect a 4"]); // bekleyen atlama ikinci kez çalışmaz
  });

  it("a failed standalone reset drops the pending seek", async () => {
    const h = harness({ autoResolve: true });
    const fails: ((e: Error) => void)[] = [];
    h.deps.reset = () => new Promise<void>((_, reject) => fails.push(reject));
    const p = createPlayer(h.deps);
    p.load("a", [1, 2, 3]);
    p.requestReset();
    p.seek(1);
    fails[0](new Error("motor yanıt vermiyor"));
    await flush();
    expect(h.log).toEqual([]);
    expect(p.snapshot().error).toBe("motor yanıt vermiyor");
    expect(p.snapshot().state).toBe("paused");
  });
});
