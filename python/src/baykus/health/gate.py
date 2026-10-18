"""Sensör durumları ve arızaya duyarlı kanal kapısı (Faz 2 tasarımı §4.1-4.2).

Kapı kuralı (plan F2 Spec yorumları 4-5): öncelik kayıp > gürültülü > hizasız; RGB'de yalnız
kayıp kapıya girer; kapı hiçbir zaman iki kanalı birden kapatmaz. Karanlık ve bulanık yalnız
bilgi. Durum kodları ASCII (Faz 3 motoru ve arayüzü okur), operatöre giden neden Türkçe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baykus.health.features import frame_features

STATE_NAMES = ("rgb_lost", "ir_lost", "ir_noisy", "misaligned", "rgb_blurry", "ir_blurry")
FLAG_ORDER = ("lost", "noisy", "dark", "blurry")
GATES = ("none", "ir_off", "rgb_off")
REASONS = {
    "rgb_lost": "RGB kayıp",
    "ir_lost": "IR kayıp",
    "ir_noisy": "IR gürültülü",
    "misaligned": "IR hizasız",
    "both_lost": "iki kamera da kayıp",
}


@dataclass(frozen=True)
class Thresholds:
    rgb_lost: float
    ir_lost: float
    ir_noisy: float
    misaligned: float
    rgb_blurry: float
    ir_blurry: float
    dark: float
    sha256: str


def load_thresholds(path: Path) -> Thresholds:
    """baykus-health fit çıktısı → Thresholds; sha256 dosya baytlarından."""
    data = Path(path).read_bytes()
    raw = json.loads(data.decode("utf-8"))
    states = raw.get("states") or {}
    missing = [n for n in STATE_NAMES if "threshold" not in (states.get(n) or {})]
    if missing or "dark_brightness" not in raw:
        raise ValueError(f"{path}: eşik dosyasında eksik: {missing or ['dark_brightness']}")
    return Thresholds(
        **{n: float(states[n]["threshold"]) for n in STATE_NAMES},
        dark=float(raw["dark_brightness"]),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _below(v: float | None, t: float) -> bool:
    return v is not None and v <= t


def _above(v: float | None, t: float) -> bool:
    return v is not None and v >= t


def _camera(flags: dict[str, bool], **values) -> dict:
    on = [f for f in FLAG_ORDER if flags.get(f)]
    return {"state": on[0] if on else "ok", "flags": on, **values}


def decide_gate(
    rgb_lost: bool, ir_lost: bool, ir_noisy: bool, misaligned: bool
) -> tuple[str, str | None]:
    """(kapı, neden kodu); öncelik kayıp > gürültülü > hizasız, iki kanal birden kapanmaz."""
    if rgb_lost and ir_lost:
        return "none", "both_lost"
    if rgb_lost:
        return "rgb_off", "rgb_lost"
    if ir_lost:
        return "ir_off", "ir_lost"
    if ir_noisy:
        return "ir_off", "ir_noisy"
    if misaligned:
        return "ir_off", "misaligned"
    return "none", None


def assess(f: dict[str, float | None], th: Thresholds, r_min: float | None = None) -> dict:
    """Özellikler → durumlar → kapı. `r_min` (Faz 3 kare kuralı) yalnız hizasızlık alarmını
    kaldırabilir: `response < r_min` (ya da yanıt yok) ise kare hizasız sayılmaz. None = Faz 2."""
    rgb_lost = _below(f["rgb_std"], th.rgb_lost)
    ir_lost = _below(f["ir_std"], th.ir_lost)
    ir_noisy = not ir_lost and _above(f["ir_noise"], th.ir_noisy)
    response = f["misalign_response"]
    misaligned = (
        not (rgb_lost or ir_lost)
        and _above(f["misalign_px"], th.misaligned)
        and (r_min is None or _above(response, r_min))
    )
    rgb = _camera(
        {
            "lost": rgb_lost,
            "dark": f["rgb_brightness"] is not None and f["rgb_brightness"] < th.dark,
            "blurry": _below(f["rgb_blur"], th.rgb_blurry),
        },
        std=f["rgb_std"],
        noise=f["rgb_noise"],
        blur=f["rgb_blur"],
        brightness=f["rgb_brightness"],
    )
    ir = _camera(
        {"lost": ir_lost, "noisy": ir_noisy, "blurry": _below(f["ir_blur"], th.ir_blurry)},
        std=f["ir_std"],
        noise=f["ir_noise"],
        blur=f["ir_blur"],
    )
    gate, why = decide_gate(rgb_lost, ir_lost, ir_noisy, misaligned)
    return {
        "rgb": rgb,
        "ir": ir,
        "pair": {
            "misalign_px": f["misalign_px"],
            "dx_px": f.get("misalign_dx"),
            "dy_px": f.get("misalign_dy"),
            "response": response,
            # zaman kuralı (baykus.health.temporal) `misaligned`'ı yeniden yazar; tek karede eşit
            "misaligned_frame": misaligned,
            "misaligned": misaligned,
        },
        "gate": gate,
        "reason_code": why,
        "reason": REASONS[why] if why else "",
    }


def apply_gate(bgr: np.ndarray, ir: np.ndarray, gate: str) -> tuple[np.ndarray, np.ndarray]:
    if gate == "none":
        return bgr, ir
    if gate == "ir_off":
        return bgr, np.zeros_like(ir)
    if gate == "rgb_off":
        return np.zeros_like(bgr), ir
    raise ValueError(f"bilinmeyen kapı kararı: {gate!r}; seçenekler: {GATES}")


def check_frame(
    bgr: np.ndarray, ir: np.ndarray, th: Thresholds, r_min: float | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Özellikler → durumlar → kapı. `changed`: kapatılan kanalda sıfır olmayan piksel vardı."""
    h = assess(frame_features(bgr, ir, vector=True), th, r_min)
    closed = {"ir_off": ir, "rgb_off": bgr}.get(h["gate"])
    h["changed"] = bool(closed is not None and closed.any())
    out_bgr, out_ir = apply_gate(bgr, ir, h["gate"])
    return out_bgr, out_ir, h
