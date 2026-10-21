import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.io import imread_any
from baykus.motion.config import MotionCfg
from baykus.motion.ego import Ego
from baykus.motion.geometry import corner_error, invert
from baykus.motion.images import half_gray
from baykus.motion.reloc import Relocalizer

ROOT = Path(__file__).resolve().parents[2]
F4 = ROOT / "fixtures" / "f4"
CFG_PATH = ROOT / "configs" / "engine" / "motion.json"
W, H = 640, 512

pytestmark = pytest.mark.skipif(not F4.exists(), reason="fixtures/f4 yok")


@pytest.fixture(scope="module")
def cfg() -> MotionCfg:
    return MotionCfg.load(CFG_PATH)


def _ir(sub: str, n: int) -> np.ndarray:
    return imread_any(F4 / sub / "ir" / f"{n:06d}.png", cv2.IMREAD_GRAYSCALE)


def _warps(kind: str) -> list[dict]:
    data = json.loads((F4 / "warps.json").read_text(encoding="utf-8"))
    return [w for w in data["warps"] if w["kind"] == kind]


def _matrix(w: dict) -> np.ndarray:
    return np.asarray(w["matrix"], np.float64).reshape(2, 3)


# --- kabul 3: küçük çarpıklıklar --------------------------------------------------------------


def test_kabul3_small_warps(cfg):
    warps = _warps("small")
    assert len(warps) == 50
    errs = []
    for w in warps:
        m = _matrix(w)
        base = _ir("clip", w.get("frame", 1))
        ego = Ego(cfg)
        ego.update(half_gray(base), "ir", [], 1)
        r = ego.update(half_gray(cv2.warpAffine(base, m, (W, H))), "ir", [], 2)
        errs.append(corner_error(r.frame_to_map, invert(m), W, H) if r.ok else float("inf"))
    assert sum(e <= 0.5 for e in errs) >= 48, sorted(errs)[-5:]


# --- kabul 10a: büyük kaymalar (yeniden konumlandırma) ----------------------------------------


def test_kabul10a_large_shifts(cfg):
    warps = _warps("large")
    assert len(warps) == 30
    errs = []
    for w in warps:
        m = _matrix(w)
        base = _ir("clip", 1)
        ego = Ego(cfg, Relocalizer(cfg))
        ego.update(half_gray(base), "ir", [], 1)
        ego.break_continuity()
        r = ego.update(half_gray(cv2.warpAffine(base, m, (W, H))), "ir", [], 2)
        merged = r.ok and r.relocalized and r.merge is not None and r.merge.to_gen == 1
        errs.append(corner_error(r.frame_to_map, invert(m), W, H) if merged else float("inf"))
    assert sum(e <= 1.0 for e in errs) >= 29, sorted(errs)[-5:]


# --- kabul 10c: aynalanmış kareler asla birleşmez ---------------------------------------------


def test_kabul10c_mirrored_never_merges(cfg):
    ego = Ego(cfg, Relocalizer(cfg))
    res = [ego.update(half_gray(_ir("clip", f)), "ir", [], f) for f in range(1, 21)]
    ego.break_continuity()
    res += [ego.update(half_gray(_ir("mirror", f)), "ir", [], f) for f in range(21, 61)]
    assert all(r.merge is None for r in res)
