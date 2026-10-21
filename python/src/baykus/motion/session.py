"""Oturumlu referans koşucusu (Faz 4 tasarımı §5.1-5.7; motor `runtime.cpp` oturum yolu).

İşlem sırası: çöz → sağlık + zaman kuralı → donmuş kare (ham) → kapı → tespit → kanal seçimi →
ego (seçili kanal; donmuş karede görüntü yok; maske takip katmanı kutuları; `frame`) → takip →
hareket → yanıt. Kanal seçimi yalnız IR (tasarım §5.1 bulgusu): IR `lost` / `noisy` ya da IR donmuş
alarmı → `source: none` (ego kaybı; dönüşte kısa kayıpsa bağlama, uzunsa yeniden konumlama).

Dönen sözlük motorun `/v1/detect` gövdesinin ölçülen kısmı: `detections[]` (gösterim, `conf`
süzgeçli), `tracks[]` (takip katmanında kimliği olan bütün tespitler, `conf`'tan bağımsız), `ego`,
`health` (`<kanal>.frozen`, `<kanal>.diff_mad` dahil). API kimliği = takipçi kimliği + oturum tabanı
(§5.5 "Kimlikler"); `scope="tracks"` tabanı o ana kadar verilen en büyük kimlik + 1 yapar.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from baykus.classes import CLASS_NAMES
from baykus.engine_ref.pipeline import Det, Pipeline, display
from baykus.health.frozen import FrozenDetector
from baykus.health.temporal import MisalignWindow
from baykus.motion.config import MotionCfg
from baykus.motion.ego import Ego
from baykus.motion.reloc import Relocalizer
from baykus.motion.state import MotionState, TrackMotion, TrackObs
from baykus.track.runner import TRACK_FLOOR, TrackStepper

SCOPES = ("all", "tracks")
_UNKNOWN = TrackMotion("unknown", None, None)


def _box(d: Det) -> dict:
    x1, y1, x2, y2 = d.box
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _motion_fields(m: TrackMotion) -> dict:
    xy = None if m.map_xy is None else [m.map_xy[0], m.map_xy[1]]
    return {"motion": m.motion, "speed_rel": m.speed_rel, "map_xy": xy}


class SessionRef:
    def __init__(self, pipeline: Pipeline, tracker_cfg: dict, motion_cfg: MotionCfg):
        self.pipeline, self.cfg = pipeline, motion_cfg
        g = pipeline.models.gate
        self.window = MisalignWindow(g.window, g.min_frames, g.tol_px)
        self.stepper = TrackStepper(tracker_cfg)
        self.ego = Ego(motion_cfg, Relocalizer(motion_cfg))
        self.motion = MotionState(motion_cfg)
        self.frozen = FrozenDetector(motion_cfg)
        self.id_base, self.max_api_id, self.n = 0, -1, 0
        self.last_dets_all: list[Det] = []

    def reset(self, scope: Literal["all", "tracks"] = "all") -> None:
        if scope not in SCOPES:
            raise ValueError(f"bilinmeyen scope: {scope!r}; seçenekler: {SCOPES}")
        self.stepper.reset()
        self.motion.reset()
        self.window.reset()
        self.frozen.reset()
        if scope == "tracks":
            self.id_base = self.max_api_id + 1
            self.ego.break_continuity()  # kütüphane korunur
        else:
            self.ego.reset()
            self.id_base, self.max_api_id, self.n = 0, -1, 0

    def step(
        self,
        rgb: bytes,
        ir: bytes,
        gate: str = "advise",
        conf: float = 0.25,
        frame: int | None = None,
    ) -> dict:
        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf [0, 1] aralığında olmalı")
        r = self.pipeline.run(rgb, ir, gate=gate, window=self.window, frozen=self.frozen)
        self.n += 1
        health, fz = r["health"], r["frozen"]
        dets_all: list[Det] = r["dets_all"]
        self.last_dets_all = dets_all
        layer = [d for d in dets_all if d.score_raw >= TRACK_FLOOR]

        ir_flags = health["ir"]["flags"]
        ir_bad = "lost" in ir_flags or "noisy" in ir_flags or fz["ir"]["alarm"]
        source = "none" if ir_bad else "ir"
        gray = None if source == "none" or fz["ir"]["frozen"] else r["ir_half"]
        ego = self.ego.update(gray, source, [d.box for d in layer], frame)

        ids: dict[int, int] = {}
        for t in self.stepper.update([asdict(d) for d in layer]):
            if t["track_id"] >= 0:
                ids[t["index"]] = t["track_id"] + self.id_base
        if ids:
            self.max_api_id = max(self.max_api_id, *ids.values())
        by_index = {d.index: d for d in layer}
        motions = self.motion.update(
            ego, [TrackObs(api, by_index[idx].box) for idx, api in ids.items()]
        )

        tracks = []
        for d in layer:
            if d.index not in ids:
                continue
            api = ids[d.index]
            tracks.append(
                {
                    "track_id": api,
                    "cls_id": d.cls_id,
                    "box": _box(d),
                    "score_raw": d.score_raw,
                    "score": d.score,
                    **_motion_fields(motions[api]),
                }
            )
        detections = []
        for d in display(dets_all, conf):
            api = ids.get(d.index)
            detections.append(
                {
                    "cls": CLASS_NAMES[d.cls_id],
                    "cls_id": d.cls_id,
                    "score": d.score,
                    "score_raw": d.score_raw,
                    "box": _box(d),
                    "track_id": api,
                    **_motion_fields(_UNKNOWN if api is None else motions[api]),
                }
            )
        return {"detections": detections, "tracks": tracks, "ego": ego.to_json(), "health": health}
