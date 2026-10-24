import { describe, expect, it } from "vitest";

import type { Detection, Motion } from "@/lib/client";
import { boxStyle, visible } from "@/lib/motion";

function det(track_id: number, motion?: Motion): Detection {
  return {
    cls: "car",
    cls_id: 0,
    score: 0.9,
    score_raw: 0.9,
    box: { x1: 10, y1: 20, x2: 110, y2: 80 },
    track_id,
    ...(motion === undefined ? {} : { motion }),
  };
}

describe("motion look (Faz 4)", () => {
  it("stationary box style is dashed with D tag", () => {
    // duran araç: kesikli ince kutu, "D" etiketi
    expect(boxStyle("stationary")).toEqual({ dash: "6 4", width: 1, tag: "D" });
    // hareketli araç: düz kalın kutu, "H" etiketi
    expect(boxStyle("moving")).toEqual({ dash: null, width: 3, tag: "H" });
    // bilinmeyen ve yok: Faz 3 görünümü (düz, orta kalınlık, etiketsiz)
    expect(boxStyle("unknown")).toEqual({ dash: null, width: 2, tag: null });
    expect(boxStyle(undefined)).toEqual({ dash: null, width: 2, tag: null });
  });

  it("only-moving filter hides stationary and unknown in detections only", () => {
    const input: Detection[] = [
      det(1, "moving"),
      det(2, "stationary"),
      det(3, "unknown"),
      det(4),
    ];

    // süzgeç kapalı: hepsi, sıra korunur
    expect(visible(input, false)).toEqual(input);
    expect(visible(input, false).map((d) => d.track_id)).toEqual([1, 2, 3, 4]);

    // süzgeç açık: yalnız hareketli
    expect(visible(input, true).map((d) => d.track_id)).toEqual([1]);

    // girdi dizisi değişmez
    expect(input).toHaveLength(4);
    expect(input.map((d) => d.track_id)).toEqual([1, 2, 3, 4]);
    expect(input.map((d) => d.motion)).toEqual(["moving", "stationary", "unknown", undefined]);
  });
});
