"""Harita kayması, duran araç kararlılığı ve yeniden konumlama raporu (Faz 4 tasarımı §6.3).

- **Harita kayması:** aynı nesilde `t` ve `t + Δ` için ürünün `P = M_{t+Δ}⁻¹ · M_t`'si ile
  referansın doğrudan SIFT + homografi kaydının (GT kutuları maskeli) dört köşe + merkezdeki en
  büyük farkı (px); Δ başına ortanca ve p95. Merkezdeki fark ayrıca: köşeler benzerlik modelinin
  perspektif hatasını da taşır (görüntü dışına doğru büyür), merkez çoğunlukla kaymayı.
- **Kararlılık:** GT'de kesintisiz ≥ 30 kare duran bölümler (iz başına); bölümde `map_xy`'nin
  standart sapması (merkeze RMS uzaklık) / harita kutu boyu ortancası; bölüm farklı nesillere
  yayılıyorsa en çok noktalı nesil.
- **Yeniden konumlama:** kesintisiz koşuda her `every` karede bir benzetilmiş atlama (geri / ileri
  `jump` kare, `scope=tracks` = `break_continuity`), en çok `window_frames` kare; başarı, deneme
  sayısı, yeniden konumlanan `frame_to_map`'in kesintisiz koşudaki aynı kareye köşe farkı ve yanlış
  bağlama (fark > 10 px).
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Callable
from typing import NamedTuple

import cv2
import numpy as np

from baykus.motion.config import MotionCfg
from baykus.motion.ego import Ego
from baykus.motion.geometry import apply, compose, corner_error, invert, scale
from baykus.motion.images import Box, half_gray
from baykus.motion.labels import Frames, LabelRow, register_direct
from baykus.motion.measure import FrameOut
from baykus.motion.reloc import Relocalizer

FALSE_LINK_PX = 10.0
MIN_SEGMENT = 30


def _m(ego: dict) -> np.ndarray:
    return np.asarray(ego["frame_to_map"], np.float64).reshape(2, 3)


def pct(xs: list[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def _direct(a, b, mask_a, mask_b, ta, tb):
    return register_direct(a, b, mask_a, mask_b)


def map_drift(
    frames: Frames,
    run: list[FrameOut],
    gt: dict[int, dict[int, Box]],
    deltas: tuple[int, ...] = (15, 60, 150),
    step: int = 10,
    register: Callable = _direct,
) -> dict:
    by_frame = {fr.frame: fr for fr in run}
    at: dict[int, list[Box]] = defaultdict(list)
    for per in gt.values():
        for f, b in per.items():
            at[f].append(b)
    out = {}
    for d in deltas:
        diffs: list[float] = []
        centers: list[float] = []
        ref_failed = skipped_gen = skipped_ok = 0
        for t in frames.frames[::step]:
            a, b = by_frame.get(t), by_frame.get(t + d)
            if a is None or b is None:
                continue
            if not (a.ego["ok"] and b.ego["ok"]):
                skipped_ok += 1
                continue
            if a.ego["map_gen"] != b.ego["map_gen"]:
                skipped_gen += 1
                continue
            ia, ib = frames.ir(t), frames.ir(t + d)
            h = register(ia, ib, at.get(t, []), at.get(t + d, []), t, t + d)
            if h is None:
                ref_failed += 1
                continue
            hh, ww = ia.shape[:2]
            pts = np.array([[0, 0], [ww, 0], [ww, hh], [0, hh], [ww / 2, hh / 2]], np.float64)
            ref = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.asarray(h, np.float64))
            prod = apply(compose(invert(_m(b.ego)), _m(a.ego)), pts)
            err = np.linalg.norm(prod - ref.reshape(-1, 2), axis=1)
            diffs.append(float(np.max(err)))
            centers.append(float(err[4]))
        out[str(d)] = {
            "pairs": len(diffs),
            "ref_failed": ref_failed,
            "skipped_gen": skipped_gen,
            "skipped_not_ok": skipped_ok,
            "median_px": pct(diffs, 50),
            "p95_px": pct(diffs, 95),
            "center_median_px": pct(centers, 50),
            "center_p95_px": pct(centers, 95),
        }
    return out


def _segments(labels: list[LabelRow]) -> list[tuple[int, list[int]]]:
    by_track: dict[int, list[int]] = defaultdict(list)
    for r in labels:
        if r.label == "stationary":
            by_track[r.track_id].append(r.frame)
    segs = []
    for tid, fs in sorted(by_track.items()):
        fs.sort()
        cur = [fs[0]]
        for f in fs[1:]:
            if f == cur[-1] + 1:
                cur.append(f)
            else:
                segs.append((tid, cur))
                cur = [f]
        segs.append((tid, cur))
    return [(t, s) for t, s in segs if len(s) >= MIN_SEGMENT]


def stationary_stability(labels: list[LabelRow], run: list[FrameOut]) -> dict:
    """Tek dizi; `run` iz kimlikleri GT kimlikleri (oracle)."""
    by_frame = {fr.frame: fr for fr in run}
    ratios: list[float] = []
    skipped = 0
    for tid, seg in _segments(labels):
        pts: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
        for f in seg:
            fr = by_frame.get(f)
            tr = None if fr is None else next((t for t in fr.tracks if t["track_id"] == tid), None)
            if tr is None or tr.get("map_xy") is None or not fr.ego["ok"]:
                continue
            x1, y1, x2, y2 = tr["box"]
            size = math.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)) * scale(_m(fr.ego))
            pts[fr.ego["map_gen"]].append((np.asarray(tr["map_xy"], np.float64), size))
        best = max(pts.values(), key=len, default=[])
        if len(best) < 2:
            skipped += 1
            continue
        xy = np.array([p for p, _ in best])
        rms = float(np.sqrt(np.mean(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1))))
        ratios.append(rms / float(np.median([s for _, s in best])))
    return {
        "segments": len(ratios),
        "skipped": skipped,
        "min_frames": MIN_SEGMENT,
        "median": pct(ratios, 50),
        "p95": pct(ratios, 95),
        "ratios": ratios,
    }


class Trial(NamedTuple):
    seq: str
    at: int  # atlamanın yapıldığı kare (kesintisiz koşuda)
    direction: str  # back | forward
    success: bool
    attempts: int
    diff_px: float | None  # kesintisiz koşuya göre köşe farkı (aynı nesil değilse None)


class _CountingReloc(Relocalizer):
    def __init__(self, cfg: MotionCfg):
        super().__init__(cfg)
        self.attempts = 0

    def try_merge(self, *a, **kw):
        self.attempts += 1
        return super().try_merge(*a, **kw)


def reloc_trials(
    frames: Frames,
    gt: dict[int, dict[int, Box]],
    cfg: MotionCfg,
    every: int = 100,
    jump: int = 50,
) -> list[Trial]:
    at: dict[int, list[Box]] = defaultdict(list)
    for per in gt.values():
        for f, b in per.items():
            at[f].append(b)
    present = set(frames.frames)
    gray = {f: half_gray(frames.ir(f)) for f in frames.frames}
    reloc = _CountingReloc(cfg)
    ego = Ego(cfg, reloc)
    base: dict[int, tuple[bool, int, np.ndarray | None]] = {}
    snaps: dict[int, Ego] = {}
    for i, f in enumerate(frames.frames, start=1):
        r = ego.update(gray[f], "ir", at.get(f, []), f)
        base[f] = (r.ok, r.map_gen, r.frame_to_map)
        if i % every == 0:
            snaps[f] = copy.deepcopy(ego)
    h, w = frames.ir(frames.frames[0]).shape[:2]  # tam çözünürlük
    trials = []
    for j, snap in snaps.items():
        for direction, start in (("back", j - jump), ("forward", j + jump)):
            if start not in present:
                continue
            e = copy.deepcopy(snap)
            e.reloc.attempts = 0
            e.break_continuity()
            success, diff = False, None
            for f in range(start, start + cfg.reloc.window_frames):
                if f not in present:
                    break
                r = e.update(gray[f], "ir", at.get(f, []), f)
                if r.relocalized:
                    success = True
                    ok, gen, m = base[f]
                    if ok and gen == r.map_gen:
                        diff = corner_error(r.frame_to_map, m, w, h)
                    break
            trials.append(Trial(frames.name, j, direction, success, e.reloc.attempts, diff))
    return trials


def reloc_summary(trials: list[Trial]) -> dict:
    def summ(ts: list[Trial]) -> dict:
        diffs = [t.diff_px for t in ts if t.success and t.diff_px is not None]
        return {
            "trials": len(ts),
            "success": sum(t.success for t in ts),
            "success_rate": sum(t.success for t in ts) / len(ts) if ts else None,
            "attempts": {
                "median": pct([t.attempts for t in ts], 50),
                "max": max((t.attempts for t in ts), default=None),
            },
            "compared": len(diffs),
            "diff_px": {"median": pct(diffs, 50), "p95": pct(diffs, 95)},
            "false_links": sum(d > FALSE_LINK_PX for d in diffs),
        }

    by_seq: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_seq[t.seq].append(t)
    return {
        **summ(trials),
        "false_link_px": FALSE_LINK_PX,
        "per_seq": {s: summ(ts) for s, ts in sorted(by_seq.items())},
    }


def reloc_report(
    frames: Frames,
    gt: dict[int, dict[int, Box]],
    cfg: MotionCfg,
    every: int = 100,
    jump: int = 50,
) -> dict:
    return reloc_summary(reloc_trials(frames, gt, cfg, every, jump))
