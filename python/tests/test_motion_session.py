from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest

from baykus.engine_ref.pipeline import Pipeline
from baykus.io import imread_any
from baykus.motion.config import MotionCfg
from baykus.motion.session import SessionRef
from baykus.track.runner import TRACK_FLOOR, load_cfg, run_tracks
from tests.test_engine_ref import _models

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "fixtures" / "m3ot" / "clip" / "sequences" / "clip"
W, H = 640, 512
PAD_Y = 64  # 640×512 → 640×640 letterbox: kazanç 1, üst dolgu 64


@pytest.fixture(scope="module")
def mcfg() -> MotionCfg:
    return MotionCfg.load(ROOT / "configs" / "engine" / "motion.json")


@pytest.fixture(scope="module")
def tcfg() -> dict:
    return load_cfg(ROOT / "configs" / "engine" / "tracker.json")


class ScriptBackend:
    """Çağrı sırasıyla kare başına ORT çıktısı (`[1, 300, 6]`, letterbox koordinatı)."""

    imgsz, channels = 640, 4

    def __init__(self, outs: list[np.ndarray]):
        self.outs, self.calls = outs, 0

    def infer(self, x: np.ndarray) -> np.ndarray:
        out = self.outs[self.calls % len(self.outs)]
        self.calls += 1
        return out


def _cam(f: int) -> tuple[float, float]:
    return 1.0 * (f - 1), 0.5 * (f - 1)  # kamera kayması (görüntü −bu kadar kayar)


def _rows(f: int) -> np.ndarray:
    """Üç araç: 1 ve 2 haritada duran, 3 haritada 4 px/kare giden. 2 ilk 5 karede yüksek skorla
    iz açar, sonra ham 0,2 (kalibre 0,2: takipte, 0,25 gösterimde yok)."""
    ox, oy = _cam(f)
    s2 = 0.9 if f <= 5 else 0.2
    cars = [(120, 150, 0.9, 0), (300, 260, s2, 1), (200 + 4 * (f - 1), 380, 0.8, 0)]
    out = np.zeros((1, 300, 6), np.float32)
    k = 0
    for i, (x, y, s, c) in enumerate(cars):
        if i == 0 and f % 7 == 0:
            continue  # ara sıra kaçan tespit
        x1, y1 = x - ox, y - oy
        out[0, k] = (x1, y1 + PAD_Y, x1 + 30, y1 + 20 + PAD_Y, s, c)
        k += 1
    return out


def _png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def _jpg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


_BASE_IR = imread_any(CLIP / "ir" / "000001.png", cv2.IMREAD_GRAYSCALE)
_BASE_RGB = imread_any(CLIP / "rgb" / "000001.jpg", cv2.IMREAD_COLOR)


def _pair(f: int) -> tuple[bytes, bytes]:
    ox, oy = _cam(f)
    m = np.float64([[1, 0, -ox], [0, 1, -oy]])
    return (
        _jpg(cv2.warpAffine(_BASE_RGB, m, (W, H))),
        _png(cv2.warpAffine(_BASE_IR, m, (W, H))),
    )


ZERO_IR = _png(np.zeros((H, W), np.uint8))


def _session(mcfg, tcfg, n: int = 80) -> SessionRef:
    pipe = Pipeline(_models(), backend=ScriptBackend([_rows(f) for f in range(1, n + 1)]))
    return SessionRef(pipe, tcfg, mcfg)


def test_tracks_independent_of_conf(mcfg, tcfg):
    a, b = _session(mcfg, tcfg), _session(mcfg, tcfg)
    for f in range(1, 31):
        rgb, ir = _pair(f)
        ra = a.step(rgb, ir, conf=0.25, frame=f)
        rb = b.step(rgb, ir, conf=1.0, frame=f)
        assert ra["tracks"] == rb["tracks"] and ra["ego"] == rb["ego"]
        assert rb["detections"] == []
    assert len(ra["tracks"]) == 3 and len(ra["detections"]) == 2  # 2. araç gösterimde yok


def test_step_matches_track_runner(mcfg, tcfg):
    s = _session(mcfg, tcfg)
    got, frames = [], []
    for f in range(1, 61):
        r = s.step(*_pair(f), frame=f)
        frames.append({"frame": f, "dets": [asdict(d) for d in s.last_dets_all]})
        got.append(sorted(t["track_id"] for t in r["tracks"]))
    ref, _ = run_tracks(frames, tcfg)
    want = [sorted(t["track_id"] for t in fr["tracks"] if t["track_id"] >= 0) for fr in ref]
    assert got == want
    assert any(got)


def test_motion_fields_and_ego(mcfg, tcfg):
    s = _session(mcfg, tcfg)
    for f in range(1, 41):
        r = s.step(*_pair(f), frame=f)
    assert r["ego"]["ok"] and r["ego"]["map_gen"] == 1 and r["ego"]["source"] == "ir"
    by_cls = {t["cls_id"]: t for t in r["tracks"] if t["score_raw"] >= 0.5}
    moving = [t for t in r["tracks"] if t["motion"] == "moving"]
    assert len(moving) == 1 and moving[0]["box"]["x1"] > 200  # yalnız 3. araç
    assert by_cls[0]["map_xy"] is not None
    d = r["detections"][0]
    assert set(d) >= {"cls", "cls_id", "score", "score_raw", "box", "track_id", "motion",
                      "speed_rel", "map_xy"}  # fmt: skip
    assert r["health"]["ir"]["frozen"] is False and r["health"]["ir"]["diff_mad"] > 0.5


def test_ego_mask_skips_low_score_boxes(mcfg, tcfg, monkeypatch):
    # Düşük skorlu kutu takip katmanında kalır ama ego maskesine girmez (Görev 15C, Kabul 10 b).
    s = _session(mcfg, tcfg)
    seen: list[list] = []
    orig = s.ego.update

    def spy(gray, source, boxes, frame):
        seen.append(list(boxes))
        return orig(gray, source, boxes, frame)

    monkeypatch.setattr(s.ego, "update", spy)
    for f in range(1, 11):
        r = s.step(*_pair(f), frame=f)
        if f < 6:  # 2. araç 6. kareden sonra ham 0,2; diğerleri 0,8-0,9
            continue
        dets = s.last_dets_all
        low = [d.box for d in dets if d.score_raw < mcfg.ego.mask_min_score]
        high = [d.box for d in dets if d.score_raw >= mcfg.ego.mask_min_score]
        assert low and high
        assert all(d.score_raw >= TRACK_FLOOR for d in dets)  # ikisi de takip katmanında
        assert seen[-1] == high
    assert len(r["tracks"]) == 3  # takip katmanı bütün katmanı kullanır


def _run_with_ir_drop(mcfg, tcfg, drop: range, n: int):
    s = _session(mcfg, tcfg)
    out = {}
    for f in range(1, n + 1):
        rgb, ir = _pair(f)
        out[f] = s.step(rgb, ZERO_IR if f in drop else ir, frame=f)
    return out


def test_ir_drop_merges_back_to_gen1(mcfg, tcfg):
    # Ego yalnız IR: IR kaybı ego kaybıdır; > relink_max_gap kare → geçici nesil → birleşme nesil 1.
    out = _run_with_ir_drop(mcfg, tcfg, range(25, 37), 45)
    assert all(not out[f]["ego"]["ok"] and out[f]["ego"]["source"] == "none" for f in range(25, 37))
    r = out[37]["ego"]
    assert r["relocalized"] and r["merge"]["from"] == 2 and r["merge"]["to"] == 1
    assert all(out[f]["ego"]["map_gen"] == 1 for f in range(37, 46))


def test_short_ir_drop_relinks(mcfg, tcfg):
    out = _run_with_ir_drop(mcfg, tcfg, range(25, 35), 40)  # 10 kare ≤ relink_max_gap
    r = out[35]["ego"]
    assert r["ok"] and r["relinked"] and r["map_gen"] == 1 and r["merge"] is None


def test_frozen_ir_frame_invalidates_ego(mcfg, tcfg):
    s = _session(mcfg, tcfg)
    s.step(*_pair(1), frame=1)
    rgb2, ir2 = _pair(2)
    assert s.step(rgb2, ir2, frame=2)["ego"]["ok"]
    rgb3, _ = _pair(3)
    r = s.step(rgb3, ir2, frame=3)  # IR baytları tekrar: donmuş kare
    assert r["health"]["ir"]["frozen"] and r["health"]["ir"]["diff_mad"] == 0.0
    assert not r["health"]["rgb"]["frozen"]
    assert not r["ego"]["ok"] and r["ego"]["frame_to_map"] is None


def test_scope_tracks_keeps_map_and_raises_id_base(mcfg, tcfg):
    s = _session(mcfg, tcfg)
    for f in range(1, 41):
        r = s.step(*_pair(f), frame=f)
    top = max(t["track_id"] for t in r["tracks"])
    s.reset("tracks")
    seen = []
    for f in range(10, 25):  # geri atlama
        r = s.step(*_pair(f), frame=f)
        seen += [t["track_id"] for t in r["tracks"]]
        if f == 10:
            assert r["ego"]["relocalized"] and r["ego"]["merge"]["to"] == 1
        assert r["ego"]["map_gen"] == 1
    assert seen and min(seen) > top


def test_scope_all_clears_everything(mcfg, tcfg):
    s = _session(mcfg, tcfg)
    for f in range(1, 21):
        s.step(*_pair(f), frame=f)
    s.reset("all")
    fresh = _session(mcfg, tcfg)
    fresh.pipeline.backend.calls = s.pipeline.backend.calls  # aynı ORT çıktıları
    for f in range(1, 21):
        assert s.step(*_pair(f), frame=f) == fresh.step(*_pair(f), frame=f)


def test_rejects_bad_scope_and_conf(mcfg, tcfg):
    s = _session(mcfg, tcfg)
    with pytest.raises(ValueError):
        s.reset("maps")
    with pytest.raises(ValueError):
        s.step(*_pair(1), conf=1.5)
    assert s.n == 0  # hatalı istek durum oluşturmaz
