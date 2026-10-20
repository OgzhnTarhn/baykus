import { afterEach, describe, expect, it, vi } from "vitest";

import { HttpError } from "@/lib/data";
import { ENGINE_TIMEOUT_MS, detect } from "@/lib/engine";

const part = { bytes: new Uint8Array([1, 2, 3]), type: "image/png" };
const params = { conf: 0.25, track: true, gate: "advise", session: "s" };

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("engine client", () => {
  it("detect aborts after timeout", async () => {
    vi.useFakeTimers();
    // Motor hiç cevap vermiyor; yalnız iptal sinyaliyle düşen bir fetch
    vi.stubGlobal(
      "fetch",
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(new DOMException("x", "AbortError")));
        }),
    );
    const pending = detect(params, part, part);
    const check = expect(pending).rejects.toMatchObject({ status: 504, message: "motor zaman aşımı" });
    await vi.advanceTimersByTimeAsync(ENGINE_TIMEOUT_MS);
    await check;
    await expect(pending).rejects.toBeInstanceOf(HttpError);
  });

  it("unreachable engine is still 502", async () => {
    vi.stubGlobal("fetch", () => Promise.reject(new TypeError("fetch failed")));
    await expect(detect(params, part, part)).rejects.toMatchObject({
      status: 502,
      message: "motor yanıt vermiyor",
    });
  });
});
