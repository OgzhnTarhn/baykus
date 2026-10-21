"""Harita belleği ve yeniden konumlama, Python referansı (Faz 4 tasarımı §5.11).

Kütüphane oturum başına; her anahtar karede bir kayıt (nesil, `M_k`, dizi kare no, istek sırası `n`,
kanal, ORB noktaları + tanımlayıcıları; yarı çözünürlük, ego maskesiyle). Geçici nesil açılınca
aday kayıtlarla (aynı kanal, geçici nesil dışı) ORB eşleştirmesi: BF Hamming çapraz denetim,
`estimateAffinePartial2D` RANSAC `reloc.ransac_px`; yineleme, güven ve iyileştirme ego'nunkiler.
Aday `ok` = iç nokta ≥ `min_inliers` ve oranı (iç nokta / eşleşme) ≥ `min_ratio`.

Birleşme: seçilen kayıt `rec`, `A` kayıt → geçerli kare; `M' = M_rec · A_full⁻¹`,
`T = M' · M_prov⁻¹` (geçici nesil → hedef nesil).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import cv2
import numpy as np

from baykus.motion.config import MotionCfg
from baykus.motion.ego import Merge, ego_mask
from baykus.motion.geometry import compose, invert, to_full
from baykus.motion.images import Box


@dataclass(frozen=True)
class Record:
    gen: int
    m_key: np.ndarray
    frame: int | None
    n: int
    source: str
    kps: np.ndarray  # N×2 float32, yarı çözünürlük
    desc: np.ndarray  # N×32 uint8


class Library:
    def __init__(self, cfg: MotionCfg):
        self.cfg = cfg
        self.records: list[Record] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(self, rec: Record) -> None:
        self.records.append(rec)
        if len(self.records) > self.cfg.reloc.library_max:
            del self.records[: len(self.records) - self.cfg.reloc.library_max]

    def candidates(self, source: str, frame: int | None, exclude_gen: int) -> list[Record]:
        """`frame` varsa `|Δframe| ≤ max_dframe` olan en yakın kayıtlar (eşitlikte yeni `n`);
        yoksa en yeni kayıtlar. Yakın kayıt yoksa aday yok (haritanın görmediği yer)."""
        rc = self.cfg.reloc
        pool = [r for r in self.records if r.source == source and r.gen != exclude_gen]
        if frame is None:
            return sorted(pool, key=lambda r: -r.n)[: rc.candidates]
        near = [r for r in pool if r.frame is not None and abs(r.frame - frame) <= rc.max_dframe]
        return sorted(near, key=lambda r: (abs(r.frame - frame), -r.n))[: rc.candidates]

    def remap_gen(self, from_gen: int, to_gen: int, t: np.ndarray) -> None:
        self.records = [
            dataclasses.replace(r, gen=to_gen, m_key=compose(t, r.m_key))
            if r.gen == from_gen
            else r
            for r in self.records
        ]

    def reset(self) -> None:
        self.records = []


def orb_features(
    gray_half: np.ndarray, boxes: list[Box], cfg: MotionCfg
) -> tuple[np.ndarray, np.ndarray]:
    e = cfg.ego
    mask = ego_mask(gray_half.shape[:2], boxes, e.mask_dilate_px, e.border_px)
    orb = cv2.ORB_create(nfeatures=cfg.reloc.orb_features)
    kps, desc = orb.detectAndCompute(gray_half, mask)
    if desc is None or not kps:
        return np.empty((0, 2), np.float32), np.empty((0, 32), np.uint8)
    # OpenCV ORB çıktı sırası derlemeye göre değişiyor (Windows tekerleği ile Linux kaynak
    # derlemesinde aynı nokta kümesi farklı sırada); eşleştirme ve RANSAC sıraya duyarlı.
    # Noktaları ve tanımlayıcı satırlarını birlikte kanonik sıraya diz (kararlı).
    order = sorted(
        range(len(kps)),
        key=lambda i: (
            kps[i].pt[1],
            kps[i].pt[0],
            kps[i].octave,
            kps[i].response,
            kps[i].angle,
        ),
    )
    pts = np.float32([kps[i].pt for i in order]).reshape(-1, 2)
    return pts, desc[order]


def match(
    cur_kps: np.ndarray, cur_desc: np.ndarray, rec: Record, cfg: MotionCfg
) -> tuple[np.ndarray | None, int, float]:
    """Kayıt → geçerli kare `A_half`; `ok` değilse `A` None."""
    rc, e = cfg.reloc, cfg.ego
    if len(cur_desc) == 0 or len(rec.desc) == 0:
        return None, 0, 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(rec.desc, cur_desc)
    if len(pairs) < 3:
        return None, 0, 0.0
    src = np.float32([rec.kps[p.queryIdx] for p in pairs])
    dst = np.float32([cur_kps[p.trainIdx] for p in pairs])
    a, inl = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=rc.ransac_px,
        maxIters=e.ransac_iters,
        confidence=e.confidence,
        refineIters=e.refine_iters,
    )
    if a is None or inl is None:
        return None, 0, 0.0
    k = int(inl.sum())
    ratio = k / len(pairs)
    ok = k >= rc.min_inliers and ratio >= rc.min_ratio
    return (a if ok else None), k, ratio


class Relocalizer:
    def __init__(self, cfg: MotionCfg):
        self.cfg = cfg
        self.library = Library(cfg)

    def reset(self) -> None:
        self.library.reset()

    def add_keyframe(
        self,
        gray_half: np.ndarray,
        boxes: list[Box],
        gen: int,
        m_key: np.ndarray,
        frame: int | None,
        n: int,
        source: str,
    ) -> None:
        kps, desc = orb_features(gray_half, boxes, self.cfg)
        self.library.add(Record(gen, np.array(m_key, np.float64), frame, n, source, kps, desc))

    def try_merge(
        self,
        gray_half: np.ndarray,
        boxes: list[Box],
        source: str,
        frame: int | None,
        m_prov: np.ndarray,
        prov_gen: int,
    ) -> Merge | None:
        cands = self.library.candidates(source, frame, exclude_gen=prov_gen)
        if not cands:
            return None
        kps, desc = orb_features(gray_half, boxes, self.cfg)
        best: tuple[tuple, Record, np.ndarray] | None = None
        for rec in cands:
            a, inl, _ = match(kps, desc, rec, self.cfg)
            if a is None:
                continue
            dframe = abs(rec.frame - frame) if rec.frame is not None and frame is not None else 0
            key = (-inl, dframe, -rec.n)  # en çok iç nokta, sonra yakın frame, sonra yeni n
            if best is None or key < best[0]:
                best = (key, rec, a)
        if best is None:
            return None
        _, rec, a = best
        m_target = compose(rec.m_key, invert(to_full(a)))
        return Merge(prov_gen, rec.gen, compose(m_target, invert(m_prov)))
