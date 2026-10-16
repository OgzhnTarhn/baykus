"""Bench çekirdeği (spec §14): aşama aşama zamanlama, ısınma turları atılır, p50/p95/ortalama.

Ön/son işleme doğruluk ölçümüyle aynı (`baykus.edge.frames`). Geç füzyonda WBF doğruluk
ölçümündeki ayarla (`skip_box_thr`: üst sınır maliyet), ardından `conf` süzgeci (operatör
ekranı varsayılanı, §15.3).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from baykus.bench.sysinfo import peak_rss_mb, sysinfo
from baykus.edge.frames import postprocess, preprocess
from baykus.fusion.wbf import fuse_image
from baykus.runinfo import git_info, iso, utc_now


def summarize(samples_ms: Sequence[float]) -> dict:
    a = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "mean": float(a.mean()),
        "n": int(a.size),
    }


def time_pipeline(
    step: Callable[[int], dict[str, float]], warmup: int, runs: int
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for i in range(warmup + runs):
        t = step(i)
        if i < warmup:
            continue
        for k, v in t.items():
            out.setdefault(k, []).append(v)
        out.setdefault("total", []).append(sum(t.values()))
    return out


def _ms(t0: int, t1: int) -> float:
    return (t1 - t0) / 1e6


def _input(backend, img: np.ndarray):
    x, lb = preprocess(img, backend.imgsz)
    return x.astype(backend.input_dtype, copy=False), lb


class SinglePipeline:
    """Tek model: pre → infer → post."""

    def __init__(self, backend, frames: list[np.ndarray], conf: float) -> None:
        self.backend, self.frames, self.conf = backend, frames, conf
        self.last: list[dict] = []

    def step(self, i: int) -> dict[str, float]:
        img = self.frames[i % len(self.frames)]
        t0 = time.perf_counter_ns()
        x, lb = _input(self.backend, img)
        t1 = time.perf_counter_ns()
        out = self.backend.infer(x)
        t2 = time.perf_counter_ns()
        self.last = postprocess(out, lb, self.conf, 0)
        t3 = time.perf_counter_ns()
        return {"pre": _ms(t0, t1), "infer": _ms(t1, t2), "post": _ms(t2, t3)}


class LateFusionPipeline:
    """İki model + WBF: pre (ikisi) → infer_rgb → infer_ir → post (ikisi) → fuse."""

    def __init__(
        self, rgb, ir, frames: list[tuple[np.ndarray, np.ndarray]], selection: dict, conf: float
    ) -> None:
        s = selection["selected"]
        if isinstance(s["ir_weight"], dict):
            raise ValueError("aydınlatmaya göre ağırlık (E4b) bench'te desteklenmiyor")
        self.rgb, self.ir, self.frames, self.conf = rgb, ir, frames, conf
        self.weights = (float(s["w_rgb"]), float(s["ir_weight"]))
        self.iou_thr, self.skip = float(s["iou_thr"]), float(selection["skip_box_thr"])
        self.last: list[dict] = []

    def step(self, i: int) -> dict[str, float]:
        bgr, ir = self.frames[i % len(self.frames)]
        h, w = bgr.shape[:2]
        t0 = time.perf_counter_ns()
        xr, lr = _input(self.rgb, bgr)
        xi, li = _input(self.ir, ir)
        t1 = time.perf_counter_ns()
        out_r = self.rgb.infer(xr)
        t2 = time.perf_counter_ns()
        out_i = self.ir.infer(xi)
        t3 = time.perf_counter_ns()
        dr = postprocess(out_r, lr, self.skip, 0)
        di = postprocess(out_i, li, self.skip, 0)
        t4 = time.perf_counter_ns()
        fused = fuse_image([dr, di], self.weights, width=w, height=h, iou_thr=self.iou_thr,
                           skip_box_thr=self.skip)  # fmt: skip
        self.last = [d for d in fused if d["score"] >= self.conf]
        t5 = time.perf_counter_ns()
        return {
            "pre": _ms(t0, t1),
            "infer_rgb": _ms(t1, t2),
            "infer_ir": _ms(t2, t3),
            "post": _ms(t3, t4),
            "fuse": _ms(t4, t5),
        }


def bench_record(
    *,
    pipeline: str,
    variant: str,
    hardware: str,
    models: list[dict],
    times: dict[str, list[float]],
    protocol: dict,
    threads: int | None,
) -> dict:
    stages = {k: summarize(v) for k, v in times.items()}
    return {
        "pipeline": pipeline,
        "variant": variant,
        "hardware": hardware,
        "models": models,
        "protocol": {**protocol, "batch": 1},
        "threads": threads,
        "stages": stages,
        "fps": 1000.0 / stages["total"]["mean"],
        "peak_rss_mb": peak_rss_mb(),
        "system": sysinfo(),
        "git_commit": git_info()["git_commit"],
        "created_at": iso(utc_now()),
    }
