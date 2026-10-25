// Araç kartı (Faz 4 tasarımı §5.7): TrackSummary -> ekran alanları. Süre kare sayısından ve
// varsayılan fps'ten türetilir; Türkçe metin texts.ts tablosundan gelir.
import { describe, expect, it } from "vitest";

import { cardView } from "@/lib/card";
import type { TrackSummary } from "@/lib/client";

function summary(over: Partial<TrackSummary> = {}): TrackSummary {
  return {
    track_id: 7,
    first_frame: 100,
    last_frame: 220,
    cls_id: 0,
    class: "car",
    motion: "stationary",
    motion_since_frame: 101,
    segments: [{ state: "stationary", from_frame: 101, to_frame: 220 }],
    stationary_frames: 120,
    path_rel: null,
    zones: [],
    alive: true,
    ...over,
  };
}

describe("card", () => {
  it("card shows stationary duration in frames and assumed seconds", () => {
    const v = cardView(summary(), 30, 220);

    // başlık ve durum
    expect(v.title).toBe("#7 car");
    expect(v.state).toBe("stationary");
    expect(v.stateText).toBe("duran");
    expect(v.tag).toBe("D");

    // N = 220 - 101 + 1 = 120 kare; 120 / 30 = 4,0 sn (ondalık virgül)
    expect(v.durationText).toBe("120 kare (≈ 4,0 sn) duruyor");
    expect(v.fpsNote).toBe("(30 kare/sn varsayımı)");
    expect(v.stationaryText).toBe("120 kare");

    // hareketli ve bilinmeyen durumda aynı kare sayısı, farklı fiil
    expect(cardView(summary({ motion: "moving" }), 30, 220).durationText).toBe(
      "120 kare (≈ 4,0 sn) hareket ediyor",
    );
    expect(cardView(summary({ motion: "moving" }), 30, 220).tag).toBe("H");
    const unknown = cardView(summary({ motion: "unknown" }), 30, 220);
    expect(unknown.durationText).toBe("durum henüz belli değil");
    expect(unknown.stateText).toBe("bilinmiyor");
    expect(unknown.tag).toBeNull();

    // yol uzunluğu kutu boyu cinsinden, ondalık virgülle
    expect(v.pathText).toBe("–");
    expect(cardView(summary({ path_rel: 12.34 }), 30, 220).pathText).toBe("12,3 kutu boyu");
  });

  it("card marks ended track as last seen", () => {
    // canlı iz: son görülme satırı yok
    const live = cardView(summary(), 30, 220);
    expect(live.ended).toBe(false);
    expect(live.lastSeenText).toBeNull();

    // bitmiş iz: kare ve kaç kare önce görüldüğü
    const ended = cardView(summary({ alive: false }), 30, 250);
    expect(ended.ended).toBe(true);
    expect(ended.lastSeenText).toBe("son görülme: kare 220 (30 kare önce)");
  });

  it("card segment ratios sum to one", () => {
    // ilk 100, son 220 -> toplam 121 kare; 51 + 70 = 121
    const v = cardView(
      summary({
        segments: [
          { state: "moving", from_frame: 100, to_frame: 150 },
          { state: "stationary", from_frame: 151, to_frame: 220 },
        ],
      }),
      30,
      220,
    );

    expect(v.segments.map((s) => s.state)).toEqual(["moving", "stationary"]);
    expect(v.segments[0].ratio).toBeCloseTo(51 / 121, 10);
    expect(v.segments[1].ratio).toBeCloseTo(70 / 121, 10);
    const sum = v.segments.reduce((acc, s) => acc + s.ratio, 0);
    expect(sum).toBeCloseTo(1, 10);
  });

  it("card zone rows use names", () => {
    const v = cardView(
      summary({
        zones: [
          { zone_id: "z1", enter_frame: 10, exit_frame: null },
          { zone_id: "z2", enter_frame: 5, exit_frame: 9 },
        ],
      }),
      30,
      220,
      { z1: "Kapı" },
    );

    expect(v.zones).toEqual([
      { zone_id: "z1", name: "Kapı", text: "içeride (giriş kare 10)" },
      { zone_id: "z2", name: "z2", text: "giriş kare 5, çıkış kare 9" },
    ]);
  });
});
