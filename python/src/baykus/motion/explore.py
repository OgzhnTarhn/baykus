"""Val keşfinin komut olarak yeniden üretimi (Faz 4 tasarımı §4; ölçüm değil keşif).

Tasarım §4'teki geçici betiğin yöntemi aynen: (1) maskesiz ORB 2000 + çapraz denetimli eşleşme +
ardışık `estimateAffinePartial2D` (RANSAC 2 px) → öteleme, dönme, ölçek ve zincirlenmiş ölçek, IR ve
RGB'de; (2) GT kutuları maskeli (4 px) ORB + ardışık homografi (RANSAC 2 px) zinciri → 15 karelik
pencerede telafi edilmiş merkez yer değiştirmesi / 15 / kutu boyu (`t` karesinde) histogramı
(`≤ 0,03`, bant, `≥ 0,10`); (3) GT kutu boyunun 100 karelik bloklardaki büyümesi.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from baykus.motion.images import Box
from baykus.motion.labels import Frames

ORB_FEATURES = 2000
MIN_MATCHES = 10
RANSAC_PX = 2.0
MASK_PX = 4
K = 15
LO, HI = 0.03, 0.10
BLOCK = 100


def _orb():
    return cv2.ORB_create(ORB_FEATURES), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def _pts(kp_a, kp_b, matches):
    return (
        np.float32([kp_a[m.queryIdx].pt for m in matches]),
        np.float32([kp_b[m.trainIdx].pt for m in matches]),
    )


def camera_steps(read) -> dict:
    """read(i) → gri kare (sırayla). Ardışık benzerlik dönüşümlerinin istatistikleri."""
    orb, bf = _orb()
    prev = None
    steps, rots, scales = [], [], []
    cum = np.eye(3)
    for img in read:
        kp, des = orb.detectAndCompute(img, None)
        if prev is not None and des is not None and prev[1] is not None:
            m = bf.match(prev[1], des)
            if len(m) >= MIN_MATCHES:
                p0, p1 = _pts(prev[0], kp, m)
                a, _ = cv2.estimateAffinePartial2D(
                    p0, p1, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX
                )
                if a is not None:
                    steps.append(math.hypot(a[0, 2], a[1, 2]))
                    rots.append(abs(math.degrees(math.atan2(a[1, 0], a[0, 0]))))
                    scales.append(math.hypot(a[0, 0], a[1, 0]))
                    cum = np.vstack([a, [0, 0, 1]]) @ cum
        prev = (kp, des)
    if not steps:
        return {
            "step_px": {"median": None, "p90": None, "max": None},
            "rot_deg_median": None,
            "scale_median": None,
            "chained_scale": None,
            "pairs": 0,
        }
    s = np.array(steps)
    return {
        "step_px": {
            "median": float(np.median(s)),
            "p90": float(np.percentile(s, 90)),
            "max": float(s.max()),
        },
        "rot_deg_median": float(np.median(rots)),
        "scale_median": float(np.median(scales)),
        "chained_scale": float(math.hypot(cum[0, 0], cum[1, 0])),
        "pairs": len(steps),
    }


def _masked(shape, boxes: list[Box]) -> np.ndarray:
    m = np.full(shape, 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        m[
            max(0, int(y1) - MASK_PX) : int(y2) + MASK_PX,
            max(0, int(x1) - MASK_PX) : int(x2) + MASK_PX,
        ] = 0
    return m


def speed_hist(frames: Frames, gt: dict[int, dict[int, Box]]) -> dict:
    boxes_at: dict[int, list[Box]] = {}
    for per in gt.values():
        for f, b in per.items():
            boxes_at.setdefault(f, []).append(b)
    orb, bf = _orb()
    chain: dict[int, tuple[int, np.ndarray]] = {}  # kare → (zincir no, birikimli homografi)
    prev, link = None, 0
    for f in frames.frames:
        img = frames.ir(f)
        kp, des = orb.detectAndCompute(img, _masked(img.shape, boxes_at.get(f, [])))
        h = None
        if prev is not None and des is not None and prev[1] is not None:
            m = bf.match(prev[1], des)
            if len(m) >= 4:
                h, _ = cv2.findHomography(*_pts(prev[0], kp, m), cv2.RANSAC, RANSAC_PX)
        if h is None or prev is None or prev[2] not in chain:
            link += 1
            chain[f] = (link, np.eye(3))
        else:
            c, acc = chain[prev[2]]
            chain[f] = (c, h @ acc)
        prev = (kp, des, f)
    lo = band = hi = 0
    for per in gt.values():
        for t, b in per.items():
            a = t - K
            if a not in per or t not in chain or a not in chain or chain[a][0] != chain[t][0]:
                continue
            hk = chain[t][1] @ np.linalg.inv(chain[a][1])
            pa = per[a]
            c0 = np.float32([[[(pa[0] + pa[2]) / 2, (pa[1] + pa[3]) / 2]]])
            p = cv2.perspectiveTransform(c0, hk)[0, 0]
            c1 = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            v = float(np.linalg.norm(c1 - p)) / K / math.sqrt((b[2] - b[0]) * (b[3] - b[1]))
            if v <= LO:
                lo += 1
            elif v >= HI:
                hi += 1
            else:
                band += 1
    return {"lo": lo, "band": band, "hi": hi, "total": lo + band + hi}


def box_growth(gt: dict[int, dict[int, Box]]) -> dict:
    """√(kutu alanı ortancası) 100 karelik bloklarda: ilk blok ve ≥ 100 kutulu son blok."""
    blocks: dict[int, list[float]] = {}
    for per in gt.values():
        for f, (x1, y1, x2, y2) in per.items():
            blocks.setdefault(f // BLOCK, []).append((x2 - x1) * (y2 - y1))
    if not blocks:
        return {"first_block": None, "last_block": None, "ratio": None}
    first = min(blocks)
    full = [b for b in blocks if len(blocks[b]) >= BLOCK] or [max(blocks)]
    last = max(full)
    a, b = math.sqrt(np.median(blocks[first])), math.sqrt(np.median(blocks[last]))
    return {"first_block": a, "last_block": b, "ratio": b / a, "blocks": [first, last]}


def explore_sequence(frames: Frames, rgb_frames, gt: dict[int, dict[int, Box]]) -> dict:
    lens = [len(per) for per in gt.values()]
    return {
        "ir": camera_steps(frames.ir(f) for f in frames.frames),
        "rgb": camera_steps(cv2.cvtColor(rgb_frames(f), cv2.COLOR_BGR2GRAY) for f in frames.frames),
        "v_hist": speed_hist(frames, gt),
        "box_growth": box_growth(gt),
        "tracks": len(gt),
        "track_len_median": float(np.median(lens)) if lens else None,
        "frames": len(frames.frames),
    }


METHOD = (
    "tasarım §4 keşif yöntemi: maskesiz ORB 2000 + ardışık benzerlik (RANSAC 2 px); "
    "GT maskeli ORB + ardışık homografi zinciri, K=15, v = yer değiştirme / K / kutu boyu (t), "
    "≤0,03 / bant / ≥0,10; kutu boyu 100 karelik bloklarda"
)
