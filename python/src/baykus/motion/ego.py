"""Ego-hareket ve harita nesli, Python referansı (Faz 4 tasarımı §5.1, §5.11).

Motor karşılığı `engine/src/ego.cpp`.

Anahtar kareli seyrek optik akış (BoT-SORT GMC "sparseOptFlow" tarifi): anahtar karenin maskeli
köşeleri LK ile doğrudan geçerli kareye, ileri-geri denetimi, `estimateAffinePartial2D` RANSAC
(yarı çözünürlük). `ok` = iç nokta ≥ `ok_min_inliers` ve oranı ≥ `ok_min_ratio`; oran = iç nokta /
RANSAC'a giren nokta. Harita dönüşümü `Mₜ = M_k · A_full⁻¹` (kare → harita).

Ego yalnız IR (tasarım §5.1 bulgusu): kaynak `ir` ya da `none`; kanal değişimi kuralı (kaynak
değişirse kopma) genel olarak durur.

Nesil kuralları:
- kayıp karesi (`gray_half=None`, `source="none"`, kayıt başarısız): `ok=False`, anahtar korunur;
- kısa kayıptan dönüş (≤ `relink_max_gap`): son anahtar kareye `relink_levels` ile bağlanma, yaş
  kuralı uygulanmaz; başarısızsa sonraki geçerli karede kopma;
- kopma (kaynak değişti, uzun kayıp, bağlama başarısız, `break_continuity()`): açık geçici nesil
  kalıcı olarak kapanır, yeni geçici nesil (`M = I`) açılır; `window_frames` kare sonra kalıcı.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from baykus.motion.config import MotionCfg
from baykus.motion.geometry import compose, invert, to_full
from baykus.motion.images import Box

_IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass(frozen=True)
class Merge:
    from_gen: int
    to_gen: int
    transform: np.ndarray  # geçici nesil koordinatı → hedef nesil (2×3, tam çözünürlük)

    def to_json(self) -> dict:
        return {
            "from": self.from_gen,
            "to": self.to_gen,
            "transform": [float(v) for v in self.transform.reshape(-1)],
        }


@dataclass(frozen=True)
class EgoResult:
    ok: bool
    source: str
    map_gen: int
    provisional: bool
    frame_to_map: np.ndarray | None
    inliers: int | None  # kayıt yapılmadıysa (neslin ilk karesi, kayıp karesi) None
    inlier_ratio: float | None
    keyframe: bool
    relinked: bool
    relocalized: bool
    merge: Merge | None

    def to_json(self) -> dict:
        """Tasarım §5.7 `ego` nesnesi; `frame_to_map` `[a, b, tx, c, d, ty]` ya da null."""
        m = self.frame_to_map
        return {
            "ok": self.ok,
            "source": self.source,
            "map_gen": self.map_gen,
            "provisional": self.provisional,
            "frame_to_map": None if m is None else [float(v) for v in m.reshape(-1)],
            "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio,
            "keyframe": self.keyframe,
            "relinked": self.relinked,
            "relocalized": self.relocalized,
            "merge": None if self.merge is None else self.merge.to_json(),
        }


class Relocalizer(Protocol):
    """Görev 5 (`baykus.motion.reloc`)."""

    def try_merge(self, gray_half, boxes, source, frame, m_prov) -> Merge | None: ...


def ego_mask(shape: tuple[int, int], boxes: list[Box], dilate_px: float, border_px: int):
    """Yarı çözünürlük maske (255 = kullanılabilir): kutular `dilate_px` (tam çöz.) genişletilmiş,
    kenar bandı `border_px // 2`. Tam `x` ↔ yarı `(x − 0,5) / 2`; kutu dışa yuvarlanır."""
    h, w = shape
    mask = np.full((h, w), 255, np.uint8)
    b = border_px // 2
    if b > 0:
        mask[:b], mask[-b:], mask[:, :b], mask[:, -b:] = 0, 0, 0, 0
    for x1, y1, x2, y2 in boxes:
        hx1 = max(0, math.floor((x1 - dilate_px - 0.5) / 2))
        hy1 = max(0, math.floor((y1 - dilate_px - 0.5) / 2))
        hx2 = min(w - 1, math.ceil((x2 + dilate_px - 0.5) / 2))
        hy2 = min(h - 1, math.ceil((y2 + dilate_px - 0.5) / 2))
        if hx1 <= hx2 and hy1 <= hy2:
            mask[hy1 : hy2 + 1, hx1 : hx2 + 1] = 0
    return mask


def _in_mask(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    xi = np.floor(pts[:, 0] + 0.5).astype(np.int64)
    yi = np.floor(pts[:, 1] + 0.5).astype(np.int64)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    keep = np.zeros(len(pts), bool)
    keep[inside] = mask[yi[inside], xi[inside]] > 0
    return keep


class Ego:
    def __init__(self, cfg: MotionCfg, reloc: Relocalizer | None = None):
        self.cfg = cfg
        self.reloc = reloc
        self.reset()

    # --- durum ------------------------------------------------------------------------------

    def reset(self) -> None:
        """`session/reset?scope=all`: nesil 1'e döner, her şey silinir."""
        self.n = 0
        self.gen = 1
        self.gens: dict[int, dict] = {}
        self._key: np.ndarray | None = None  # anahtar kare (yarı çöz. gri)
        self._key_pts = np.empty((0, 2), np.float32)
        self._m_key = _IDENTITY.copy()
        self._gap = 0
        self._age = 0
        self._source: str | None = None
        self._pending_break = False

    def break_continuity(self) -> None:
        """`session/reset?scope=tracks` (atlama): sonraki geçerli kare geçici nesil açar."""
        if self._key is not None:
            self._pending_break = True

    @property
    def generations(self) -> dict[int, dict]:
        return self.gens

    @property
    def key_points(self) -> np.ndarray:
        """Anahtar karenin köşeleri (yarı çözünürlük, N×2)."""
        return self._key_pts.reshape(-1, 2).astype(np.float64)

    # --- adım -------------------------------------------------------------------------------

    def update(
        self, gray_half: np.ndarray | None, source: str, boxes: list[Box], frame: int | None
    ) -> EgoResult:
        self.n += 1
        if gray_half is None or source == "none":
            res = self._lost("none")
        elif self._key is None:
            res = self._first(gray_half, source, boxes)
        elif (
            self._pending_break or source != self._source or self._gap > self.cfg.ego.relink_max_gap
        ):
            res = self._break(gray_half, source, boxes)
        else:
            res = self._track(gray_half, source, boxes)
        return self._close_window(res)

    def _result(self, ok: bool, source: str, m: np.ndarray | None, **kw) -> EgoResult:
        base = dict(
            inliers=None, inlier_ratio=None, keyframe=False, relinked=False, relocalized=False,
            merge=None,
        )  # fmt: skip
        base.update(kw)
        return EgoResult(
            ok=ok,
            source=source,
            map_gen=self.gen,
            provisional=bool(self.gens.get(self.gen, {}).get("provisional", False)),
            frame_to_map=None if m is None else m.copy(),
            **base,
        )

    def _lost(self, source: str) -> EgoResult:
        if self._key is not None:
            self._gap += 1
            self._age += 1
        return self._result(False, source, None)

    def _corners(self, gray: np.ndarray, boxes: list[Box]) -> np.ndarray:
        e = self.cfg.ego
        mask = ego_mask(gray.shape[:2], boxes, e.mask_dilate_px, e.border_px)
        pts = cv2.goodFeaturesToTrack(
            gray, e.max_corners, e.quality, e.min_distance, mask=mask, blockSize=e.block_size
        )
        return np.empty((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)

    def _set_key(self, gray: np.ndarray, pts: np.ndarray, m: np.ndarray) -> None:
        self._key, self._key_pts, self._m_key, self._age = gray.copy(), pts, m.copy(), 0

    def _open(self, gray: np.ndarray, source: str, boxes: list[Box], provisional: bool):
        """Neslin ilk karesi: `M = I`, anahtar kare. Köşe azsa açılmaz (None)."""
        pts = self._corners(gray, boxes)
        if len(pts) < self.cfg.ego.ok_min_inliers:
            return None
        if provisional:
            for g in self.gens.values():
                g["provisional"] = False  # açık geçici nesil birleşmeden kapanır (§5.11)
            self.gen = max(self.gens) + 1
        self.gens[self.gen] = {"provisional": provisional, "opened_n": self.n}
        self._set_key(gray, pts, _IDENTITY)
        self._source, self._gap, self._pending_break = source, 0, False
        return self._result(True, source, _IDENTITY, keyframe=True)

    def _first(self, gray: np.ndarray, source: str, boxes: list[Box]) -> EgoResult:
        return self._open(gray, source, boxes, provisional=False) or self._result(
            False, source, None
        )

    def _break(self, gray: np.ndarray, source: str, boxes: list[Box]) -> EgoResult:
        res = self._open(gray, source, boxes, provisional=True)
        if res is None:  # köşe az: kopma bekler
            self._pending_break = True
            return self._lost(source)
        return res

    def _register(self, gray: np.ndarray, boxes: list[Box], levels: int):
        """Anahtar kare → geçerli kare; (A_half | None, iç nokta, oran)."""
        e = self.cfg.ego
        p0 = self._key_pts.reshape(-1, 1, 2).astype(np.float32)
        if len(p0) == 0:
            return None, 0, 0.0
        win = (e.lk_win, e.lk_win)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self._key, gray, p0, None, winSize=win, maxLevel=levels
        )
        if p1 is None:
            return None, 0, 0.0
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._key, p1, None, winSize=win, maxLevel=levels
        )
        if pb is None:
            return None, 0, 0.0
        a, b = p0.reshape(-1, 2), p1.reshape(-1, 2)
        fb = np.linalg.norm(pb.reshape(-1, 2) - a, axis=1)
        good = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & np.isfinite(fb)
        good &= fb <= e.fb_max_px
        good &= _in_mask(ego_mask(gray.shape[:2], boxes, e.mask_dilate_px, e.border_px), b)
        n = int(good.sum())
        if n < 3:  # estimateAffinePartial2D en az 2 nokta ister; 3'ün altı anlamsız
            return None, 0, 0.0
        a_half, inl = cv2.estimateAffinePartial2D(
            a[good],
            b[good],
            method=cv2.RANSAC,
            ransacReprojThreshold=e.ransac_px,
            maxIters=e.ransac_iters,
            confidence=e.confidence,
            refineIters=e.refine_iters,
        )
        if a_half is None or inl is None:
            return None, 0, 0.0
        k = int(inl.sum())
        return a_half, k, k / n

    def _track(self, gray: np.ndarray, source: str, boxes: list[Box]) -> EgoResult:
        e = self.cfg.ego
        relink = self._gap > 0
        a_half, inl, ratio = self._register(gray, boxes, e.relink_levels if relink else e.lk_levels)
        ok = a_half is not None and inl >= e.ok_min_inliers and ratio >= e.ok_min_ratio
        if not ok:
            if relink:
                self._pending_break = True
            self._lost(source)
            return self._result(False, source, None, inliers=inl, inlier_ratio=ratio)
        self._gap = 0
        self._age += 1
        m = compose(self._m_key, invert(to_full(a_half)))
        shift = math.hypot(float(a_half[0, 2]), float(a_half[1, 2]))
        key = (
            inl < e.key_min_inliers
            or ratio < e.key_min_ratio
            or shift > e.key_max_shift_px
            or (not relink and self._age >= e.key_max_age)
        )
        if key:
            self._set_key(gray, self._corners(gray, boxes), m)
        return self._result(
            True, source, m, inliers=inl, inlier_ratio=ratio, keyframe=key, relinked=relink
        )

    def _close_window(self, res: EgoResult) -> EgoResult:
        """Geçici nesil `window_frames` kare sonra kalıcı olur (yeniden konumlama Görev 5)."""
        g = self.gens.get(self.gen)
        if g is None or not g["provisional"]:
            return res
        if self.n - g["opened_n"] < self.cfg.reloc.window_frames:
            return res
        g["provisional"] = False
        return EgoResult(**{**res.__dict__, "provisional": False})
