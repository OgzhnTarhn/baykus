"""Motorun işlem sırasının referansı (Faz 3 tasarımı §4.1):

çöz → sağlık (her modda) → donmuş kare (oturumlu, ham girdi) → kapı (`auto` ve karar ≠ none ise
kanal sıfırlanır) → 4 kanal → letterbox → ORT → çözümleme + taban süzgeci (ham skor ≥ floor) →
kalibre skor → sıralama. Donmuş kare (Faz 4 §5.6) `health.<kanal>.frozen` / `diff_mad` olarak
eklenir; yarı çözünürlük gri kareler (kapıdan önce) sonuçta `rgb_half`, `ir_half`.
Ham skor float32 (model çıktısı), kalibre skor double. Sıralama: kalibre azalan, ham azalan,
ORT satır indeksi artan.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from baykus.data.multichannel import stack_rgbt
from baykus.edge.frames import Letterbox, preprocess
from baykus.engine_ref.compose import decode_pair
from baykus.engine_ref.models import Models
from baykus.health.calibrate import apply_isotonic
from baykus.health.features import frame_features
from baykus.health.frozen import FrozenDetector
from baykus.health.gate import apply_gate, assess
from baykus.health.temporal import MisalignWindow
from baykus.motion.images import half_gray

GATE_MODES = ("off", "advise", "auto")
TILES_MODES = (1, 4)


@dataclass(frozen=True)
class Det:
    index: int  # ORT çıktısındaki satır
    cls_id: int
    score_raw: float  # float32 değeri
    score: float  # kalibre, double
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (görüntü pikseli)


def decode_rows(
    out: np.ndarray,
    lb: Letterbox,
    floor: float,
    calib: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[Det]:
    """`[1, N, 6]` → Det (satır sırasıyla). Geri ölçekleme, kırpma ve `w, h ≤ 0` atma
    `baykus.edge.frames.postprocess` ile aynı. `calib` yoksa `score` = `score_raw`."""
    dets: list[Det] = []
    for i, row in enumerate(out[0]):
        raw = float(row[4])
        if raw < floor:
            continue
        x1 = (float(row[0]) - lb.pad_x) / lb.gain
        y1 = (float(row[1]) - lb.pad_y) / lb.gain
        x2 = (float(row[2]) - lb.pad_x) / lb.gain
        y2 = (float(row[3]) - lb.pad_y) / lb.gain
        x1, x2 = min(max(x1, 0.0), float(lb.width)), min(max(x2, 0.0), float(lb.width))
        y1, y2 = min(max(y1, 0.0), float(lb.height)), min(max(y2, 0.0), float(lb.height))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        score = raw if calib is None else float(apply_isotonic(np.float64(raw), *calib))
        dets.append(Det(i, int(row[5]), raw, score, (x1, y1, x2, y2)))
    return dets


def order(dets: list[Det]) -> list[Det]:
    return sorted(dets, key=lambda d: (-d.score, -d.score_raw, d.index))


def display(dets_all: list[Det], conf: float) -> list[Det]:
    """Gösterim katmanı: kalibre skor ≥ conf (sıra korunur)."""
    return [d for d in dets_all if d.score >= conf]


class Pipeline:
    def __init__(self, models: Models, backend=None) -> None:
        if backend is None:
            from baykus.edge.backends import OrtBackend

            backend = OrtBackend(models.onnx)
        self.models, self.backend = models, backend

    def run(
        self,
        rgb_bytes: bytes,
        ir_bytes: bytes,
        *,
        gate: str = "advise",
        floor: float = 0.1,
        window: MisalignWindow | None = None,
        frozen: FrozenDetector | None = None,
        tiles: int = 1,
    ) -> dict:
        if gate not in GATE_MODES:
            raise ValueError(f"bilinmeyen gate modu: {gate!r}; seçenekler: {GATE_MODES}")
        m = self.models
        bgr, ir = decode_pair(rgb_bytes, ir_bytes)
        health = assess(frame_features(bgr, ir, vector=True), m.thresholds, m.gate.r_min)
        if window is not None:
            health = window.update(health)
        halves: dict = {"rgb_half": None, "ir_half": None, "frozen": None}
        if frozen is not None:
            rgb_half, ir_half = half_gray(bgr), half_gray(ir)
            fz = frozen.update(
                rgb_half, ir_half, "lost" in health["rgb"]["flags"], "lost" in health["ir"]["flags"]
            )
            for ch in ("rgb", "ir"):
                health[ch]["frozen"] = fz[ch]["frozen"]
                health[ch]["diff_mad"] = fz[ch]["diff_mad"]
            halves = {"rgb_half": rgb_half, "ir_half": ir_half, "frozen": fz}
        if gate == "off":  # sağlık yine ölçülür; yalnız kapı kararı "none" sayılır
            health["gate"] = "none"
        health["applied"] = gate == "auto" and health["gate"] != "none"
        if health["applied"]:
            bgr, ir = apply_gate(bgr, ir, health["gate"])
        return {"health": health, **halves, **self.detect(bgr, ir, floor, tiles)}

    def detect(self, bgr: np.ndarray, ir: np.ndarray, floor: float = 0.1, tiles: int = 1) -> dict:
        """Tespit: `tiles=1` tam kare (bugünkü davranış), `tiles=4` 2 × 2 pencere
        (`baykus.engine_ref.tiles`)."""
        if tiles not in TILES_MODES:
            raise ValueError(f"bilinmeyen tiles modu: {tiles!r}; seçenekler: {TILES_MODES}")
        if tiles == 1:
            return self._detect_one(bgr, ir, floor)[0]
        from baykus.engine_ref.tiles import merge, windows  # döngüsel içe aktarma

        wins = windows(bgr.shape[1], bgr.shape[0])
        per_tile: list[list[Det]] = []
        tensors: list[bytes] = []
        for x0, y0, pw, ph in wins:
            tile_bgr = np.ascontiguousarray(bgr[y0 : y0 + ph, x0 : x0 + pw])
            tile_ir = np.ascontiguousarray(ir[y0 : y0 + ph, x0 : x0 + pw])
            res, x = self._detect_one(tile_bgr, tile_ir, floor)
            per_tile.append(res["dets_all"])
            tensors.append(x.tobytes())
        return {
            "dets_all": merge(per_tile, wins),
            "tensor_sha256": hashlib.sha256(b"".join(tensors)).hexdigest(),
            "raw": None,
            "lb": None,
            "windows": wins,
        }

    def _detect_one(
        self, bgr: np.ndarray, ir: np.ndarray, floor: float = 0.1
    ) -> tuple[dict, np.ndarray]:
        """Çözülmüş (ve gerekirse kapılanmış) çiftten tespit: 4 kanal → letterbox → ORT.
        Dönüş: (sonuç sözlüğü, letterbox tensörü)."""
        m = self.models
        x, lb = preprocess(stack_rgbt(bgr, ir), imgsz=getattr(self.backend, "imgsz", 640))
        raw = self.backend.infer(x)
        dets = decode_rows(raw, lb, floor, (m.calib_x, m.calib_y))
        return {
            "dets_all": order(dets),
            "tensor_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "raw": raw,
            "lb": lb,
        }, x
