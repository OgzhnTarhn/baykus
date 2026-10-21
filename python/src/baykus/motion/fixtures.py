"""Hareket eşlik fikstürleri (Faz 4 tasarımı §5.9, plan F4 Görev 9).

Yalnız M3OT val (`1_08`; test asla). Release paketi `fixtures-f4.tar` (repoya değil):
`clip/` (60 ardışık kare, özgün JPG / PNG baytları), `mirror/` (21-60. kareler yatay aynalanmış),
`zero_rgb.png`, `zero_ir.png`, `scenarios.json` (8 senaryo; her istek `frame` taşır),
`warps/` + `warps.json` (Kabul 3 küçük benzerlik, Kabul 10 büyük öteleme; yalnız IR).
Beklenen senaryo çıktıları pakette yok: ONNX Runtime işlemciye göre yuvarlar, CI aynı makinede
`baykus-motion clip-ref` ile üretir.

Sınır kuralı (üretici denetler, Faz 3 §4.6 gibi): anahtar kare / bağlama / yeniden konumlama /
nesil kararına giren ölçüler (iç nokta, oran, öteleme) eşiğinden en az 2 nokta / 0,02 / 1 px uzak
olmalı — yalnız ölçünün eşiğin iki yanında kararı değiştirdiği yerde. Ayrıca sağlık özellikleri
(Faz 3 `check_margins`, `clip_violations`) ve donmuş kare `diff_mad` (göreli 1e-3). Sağlık payları
Faz 3'ün 1/100'ü: ölçülen C++ / Python farkı ≤ 4e-12 px, ≤ 2e-16 göreli ve yanıtta ≤ 6e-17
(`results/engine/parity_health.json`); Faz 3'ün 10 × tolerans payıyla 60 karelik klip hiç geçmez
(Faz 3'te 15 karede bile 9 / 1172). Tutmayan ya da
beklenen nesil olaylarını üretmeyen başlangıç karesi bir sonrakine kaydırılır (yalnız eşlik için).

Repoda: `fixtures/motion/maps_val.json.gz` (iki dizi × 600 kare ego, Python; maske takip katmanı
tespitleri, kaynak her karede IR) ve `fixtures/expected/motion_val.json` (hareket birim eşliği,
Kabul 2): ego sabitken `MotionState`'in iz × kare çıktısı `[track_id, motion, speed_rel, x, y]`.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import shutil
import tarfile
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np

from baykus.engine_ref.compose import decode_pair
from baykus.engine_ref.fixtures import check_margins, clip_violations
from baykus.engine_ref.pipeline import Pipeline
from baykus.health.m3ot import alarm_states
from baykus.io import imread_any
from baykus.motion.config import MotionCfg
from baykus.motion.ego import Ego, EgoResult, Merge
from baykus.motion.geometry import SIMILARITIES, similarity
from baykus.motion.images import half_gray, load_sequence
from baykus.motion.reloc import Relocalizer, match, orb_features
from baykus.motion.session import SessionRef
from baykus.motion.state import MotionState, TrackObs
from baykus.track.runner import TRACK_FLOOR

SEQ, CLIP_LEN = "1_08", 60
INL_MARGIN, RATIO_MARGIN, PX_MARGIN, MAD_REL = 2, 0.02, 1.0, 1e-3
HEALTH_MARGINS = {"rel_margin": 1e-5, "px_margin": 0.05, "response_margin": 0.002}
HEALTH_DIST_MARGIN = 0.02  # px
MIN_TRACK_ROWS = 150  # clean'de iz × kare (seçilen ilk klipte 2'ydi: hareket eşliği boş kalıyordu)
IR_DROP, IR_FREEZE, SHORT_GAP, LONG_GAP = (25, 36), (20, 29), (40, 44), (30, 45)
SEEK_UNTIL, SEEK_BACK, MIRROR_FROM, RESET_ALL_AT = 40, 10, 21, 40
WARP_EVERY = 6  # 1, 7, ..., 55 → 10 kaynak kare
LARGE_SHIFTS = (60.0, 120.0, 200.0)  # Kabul 10: x yönünde öteleme (tam çözünürlük px)
JPG_QUALITY, PNG_LEVEL = 95, 9
SCENARIOS = (
    "clean",
    "ir_drop",
    "ir_freeze",
    "short_gap",
    "long_gap",
    "seek",
    "scene_change",
    "reset_all",
)
ZERO_RGB, ZERO_IR = "zero_rgb.png", "zero_ir.png"


# --- sınır kuralı -----------------------------------------------------------------------------


def _fragile(decide: Callable[[dict], object], vals: dict, rules) -> list[str]:
    """Eşiğe `margin`'den yakın ölçü, eşiğin iki yanında (t ± margin) kararı değiştirirse ihlal."""
    out = []
    for name, thr, margin in rules:
        v = vals[name]
        if abs(v - thr) < margin and decide({**vals, name: thr - margin}) != decide(
            {**vals, name: thr + margin}
        ):
            out.append(f"{name} {v:.4g} eşiğe ({thr:g}) {margin:g}'den yakın")
    return out


def ego_margins(
    cfg: MotionCfg, inliers: int, ratio: float, shift: float, relink: bool, age: int
) -> list[str]:
    """Bir kaydın `ok` ve anahtar kare kararı (`Ego._track`); `age` karar anındaki yaş."""
    e = cfg.ego

    def decide(v: dict) -> tuple[bool, bool]:
        ok = v["inliers"] >= e.ok_min_inliers and v["ratio"] >= e.ok_min_ratio
        key = ok and (
            v["inliers"] < e.key_min_inliers
            or v["ratio"] < e.key_min_ratio
            or v["shift"] > e.key_max_shift_px
            or (not relink and age >= e.key_max_age)
        )
        return ok, key

    rules = (
        ("inliers", e.ok_min_inliers, INL_MARGIN),
        ("inliers", e.key_min_inliers, INL_MARGIN),
        ("ratio", e.ok_min_ratio, RATIO_MARGIN),
        ("ratio", e.key_min_ratio, RATIO_MARGIN),
        ("shift", e.key_max_shift_px, PX_MARGIN),
    )
    return _fragile(decide, {"inliers": inliers, "ratio": ratio, "shift": shift}, rules)


def reloc_margins(cfg: MotionCfg, matches: list[tuple[int, float]]) -> list[str]:
    """Aday başına `ok` (iç nokta, oran) ve seçim (en çok iç nokta; ikinciden ≥ 2 nokta önde).
    Daha güçlü bir aday zaten seçilecekse zayıf adayın sınırı karara girmez."""
    r = cfg.reloc

    def ok(v: dict) -> bool:
        return v["inliers"] >= r.min_inliers and v["ratio"] >= r.min_ratio

    rules = (("inliers", r.min_inliers, INL_MARGIN), ("ratio", r.min_ratio, RATIO_MARGIN))
    vals = [{"inliers": k, "ratio": q} for k, q in matches]
    robust = [v["inliers"] for v in vals if ok(v) and not _fragile(ok, v, rules)]
    out = []
    for i, v in enumerate(vals):
        msgs = _fragile(ok, v, rules)
        if msgs and not any(k >= v["inliers"] + INL_MARGIN for k in robust):
            out += [f"aday {i}: {m}" for m in msgs]
    good = sorted((v["inliers"] for v in vals if ok(v)), reverse=True)
    if len(good) > 1 and good[0] - good[1] < INL_MARGIN:
        out.append(f"en iyi iki aday iç nokta {good[0]} / {good[1]}: {INL_MARGIN}'den yakın")
    return out


class _Probe:
    def __init__(self) -> None:
        self.where, self.issues = "", []

    def add(self, msgs: Iterable[str]) -> None:
        self.issues += [f"{self.where}: {m}" for m in msgs]


class _ProbeReloc(Relocalizer):
    def __init__(self, cfg: MotionCfg, probe: _Probe):
        super().__init__(cfg)
        self.probe = probe

    def try_merge(self, gray_half, boxes, source, frame, m_prov, prov_gen):
        cands = self.library.candidates(source, frame, exclude_gen=prov_gen)
        if cands:
            kps, desc = orb_features(gray_half, boxes, self.cfg)
            ms = [match(kps, desc, rec, self.cfg)[1:] for rec in cands]
            self.probe.add(f"yeniden konumlama {m}" for m in reloc_margins(self.cfg, ms))
        return super().try_merge(gray_half, boxes, source, frame, m_prov, prov_gen)


class _ProbeEgo(Ego):
    """Ego'nun karar ölçülerini sınır kuralıyla denetler (sonucu değiştirmez)."""

    def __init__(self, cfg: MotionCfg, probe: _Probe):
        self.probe, self._n_corners = probe, 0
        super().__init__(cfg, _ProbeReloc(cfg, probe))

    def _corners(self, gray, boxes):
        pts = super()._corners(gray, boxes)
        self._n_corners = len(pts)
        return pts

    def _open(self, gray, source, boxes, provisional):
        res = super()._open(gray, source, boxes, provisional)
        thr = self.cfg.ego.ok_min_inliers
        if abs(self._n_corners - thr) < INL_MARGIN:
            self.probe.add([f"nesil açılışı köşe {self._n_corners} eşiğe ({thr}) yakın"])
        return res

    def _register(self, gray, boxes, levels):
        a, inl, ratio = super()._register(gray, boxes, levels)
        if a is not None:
            shift = math.hypot(float(a[0, 2]), float(a[1, 2]))
            msgs = ego_margins(self.cfg, inl, ratio, shift, self._gap > 0, self._age + 1)
            self.probe.add(f"ego {m}" for m in msgs)
        return a, inl, ratio


def pick_start(starts: Iterable[int], check: Callable[[int], list[str]]) -> tuple[int, list[str]]:
    """İlk sorunsuz başlangıç; atlananlar notlara (başlangıç başına ilk sorun)."""
    notes = []
    for s in starts:
        issues = check(s)
        if not issues:
            return s, notes
        more = f" (+{len(issues) - 1})" if len(issues) > 1 else ""
        notes.append(f"başlangıç {s}: {issues[0]}{more}")
    raise ValueError("sınır kuralını geçen başlangıç yok: " + "; ".join(notes[-3:]))


def pick_best(
    starts: Iterable[int], evaluate: Callable[[int], tuple[list[str], list[str]]]
) -> tuple[int, list[str], list[str]]:
    """Sınır kuralını hiçbir başlangıç tam geçmediğinde (kullanıcı kararı 2026-10-21): senaryo
    kapsamı tam olanlar arasında en az sorunlu (eşitlikte önceki). `evaluate` → (sorunlar, kapsam
    eksikleri). Eşiğe yakın kalan kayıtlar pakette listelenir; C++ ölçümü (Görev 11) karar verir."""
    best, notes = None, []
    for s in starts:
        issues, missing = evaluate(s)
        if missing:
            notes.append(f"başlangıç {s}: {len(issues)} sorun, kapsam eksik ({missing[0]})")
            continue
        notes.append(f"başlangıç {s}: {len(issues)} sorun")
        if best is None or len(issues) < len(best[1]):
            best = (s, issues)
    if best is None:
        raise ValueError("senaryo kapsamı tam olan başlangıç yok: " + "; ".join(notes[-3:]))
    return best[0], best[1], notes


def margin_records(issues: list[str]) -> list[dict]:
    """`"<senaryo> kare <n>: <ileti>"` → `{scenario, frame, text}`; yer bilgisi olmayan satır
    (zaman kuralı) `frame: None`."""
    out = []
    for line in issues:
        m = re.match(r"^(\w+) kare (\d+): (.*)$", line)
        if m:
            out.append({"scenario": m[1], "frame": int(m[2]), "text": m[3]})
        else:
            name, _, text = line.partition(": ")
            out.append({"scenario": name, "frame": None, "text": text})
    return out


# --- paket ------------------------------------------------------------------------------------


def _clip(f: int) -> dict:
    return {"frame": f, "rgb": f"clip/rgb/{f:06d}.jpg", "ir": f"clip/ir/{f:06d}.png"}


def scenarios(n: int = CLIP_LEN) -> dict[str, list[dict]]:
    """Tasarım §5.9 senaryoları; `ir_drop` 25-36 (12 kare > `relink_max_gap`: ego kaybı →
    geçici nesil → nesil 1'e birleşme; plan değişikliği 2026-10-20)."""

    def step(f: int, reset: str | None = None, **over) -> dict:
        return {**_clip(f), **over, "reset_before": reset}

    def within(f: int, span: tuple[int, int]) -> bool:
        return span[0] <= f <= span[1]

    zero = {"rgb": ZERO_RGB, "ir": ZERO_IR}
    frozen_ir = _clip(IR_FREEZE[0] - 1)["ir"]
    frames = range(1, n + 1)
    return {
        "clean": [step(f) for f in frames],
        "ir_drop": [step(f, **({"ir": ZERO_IR} if within(f, IR_DROP) else {})) for f in frames],
        "ir_freeze": [
            step(f, **({"ir": frozen_ir} if within(f, IR_FREEZE) else {})) for f in frames
        ],
        "short_gap": [step(f, **(zero if within(f, SHORT_GAP) else {})) for f in frames],
        "long_gap": [step(f, **(zero if within(f, LONG_GAP) else {})) for f in frames],
        "seek": [step(f) for f in range(1, SEEK_UNTIL + 1)]
        + [step(f, "tracks" if f == SEEK_BACK else None) for f in range(SEEK_BACK, n + 1)],
        "scene_change": [
            step(f, rgb=f"mirror/rgb/{f:06d}.jpg", ir=f"mirror/ir/{f:06d}.png")
            if f >= MIRROR_FROM
            else step(f)
            for f in frames
        ],
        "reset_all": [step(f, "all" if f == RESET_ALL_AT else None) for f in frames],
    }


def _encode(ext: str, img: np.ndarray) -> bytes:
    params = (
        [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY] if ext == ".jpg" else [cv2.IMWRITE_PNG_COMPRESSION,
                                                                     PNG_LEVEL]
    )  # fmt: skip
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise OSError(f"{ext} kodlanamadı")
    return buf.tobytes()


def _put(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _put_json(path: Path, rec) -> None:
    text = json.dumps(rec, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_clip(seq_dir: Path, start: int, out: Path, n: int = CLIP_LEN) -> dict:
    """`clip/`, `mirror/`, sıfır görüntüler; klip karesi k = dizi karesi `start + k − 1`."""
    seq_dir, out = Path(seq_dir), Path(out)
    index = json.loads((seq_dir / "index.json").read_text(encoding="utf-8"))
    if index["split"] == "test":
        raise ValueError("test bölümü fikstür olmaz (tasarım §5.9)")
    wanted = list(range(start, start + n))
    if not set(wanted) <= set(index["frames"]):
        raise ValueError(f"{seq_dir.name}: {start}-{start + n - 1} ardışık kareler yok")
    for sub in ("clip", "mirror"):
        shutil.rmtree(out / sub, ignore_errors=True)
    size = None
    for k, f in enumerate(wanted, start=1):
        rgb = (seq_dir / "rgb" / f"{f:06d}.jpg").read_bytes()
        ir = (seq_dir / "ir" / f"{f:06d}.png").read_bytes()
        _put(out / "clip" / "rgb" / f"{k:06d}.jpg", rgb)
        _put(out / "clip" / "ir" / f"{k:06d}.png", ir)
        if k == 1 or k >= MIRROR_FROM:
            bgr, gray = decode_pair(rgb, ir)
        if k == 1:
            size = [int(gray.shape[1]), int(gray.shape[0])]
            _put(out / ZERO_RGB, _encode(".png", np.zeros_like(bgr)))
            _put(out / ZERO_IR, _encode(".png", np.zeros_like(gray)))
        if k >= MIRROR_FROM:
            _put(out / "mirror" / "rgb" / f"{k:06d}.jpg", _encode(".jpg", cv2.flip(bgr, 1)))
            _put(out / "mirror" / "ir" / f"{k:06d}.png", _encode(".png", cv2.flip(gray, 1)))
    rec = {
        "name": "clip",
        "split": index["split"],
        "frames": list(range(1, n + 1)),
        "fps": 30,
        "size": size,
        "source": f"M3OT {index['split']} {seq_dir.name} kare {start}-{start + n - 1}, CC BY 4.0",
        "start": start,
    }
    _put_json(out / "clip" / "index.json", rec)
    return rec


def write_warps(out: Path, n: int = CLIP_LEN) -> list[dict]:
    """Klibin IR karelerinden bilinen dönüşümler (`cv2.warpAffine`, doğrusal, kenar 0): her kaynak
    karede `geometry.SIMILARITIES` (görüntü merkezi etrafında; Kabul 3) ve `LARGE_SHIFTS`
    (Kabul 10). `matrix` kaynak → çarpık görüntü (tam çözünürlük)."""
    out = Path(out)
    shutil.rmtree(out / "warps", ignore_errors=True)
    items = []
    for f in range(1, n + 1, WARP_EVERY):
        src = imread_any(out / "clip" / "ir" / f"{f:06d}.png", cv2.IMREAD_GRAYSCALE)
        h, w = src.shape
        specs = [("small", p) for p in SIMILARITIES] + [
            ("large", (t, 0.0, 0.0, 1.0)) for t in LARGE_SHIFTS
        ]
        for kind, (tx, ty, deg, s) in specs:
            m = similarity(tx, ty, deg, s, center=(w / 2, h / 2))
            path = f"warps/{len(items):03d}.png"
            _put(out / path, _encode(".png", cv2.warpAffine(src, m, (w, h))))
            items.append(
                {
                    "path": path,
                    "src_frame": f,
                    "kind": kind,
                    "params": {"tx": tx, "ty": ty, "deg": deg, "scale": s},
                    "matrix": [float(v) for v in m.reshape(-1)],
                }
            )
    rec = {"interp": "INTER_LINEAR", "border": "BORDER_CONSTANT 0", "warps": items}
    _put_json(out / "warps.json", rec)
    return items


def write_tar(stage: Path, path: Path) -> tuple[str, int]:
    """Belirlenimci tar: sıralı adlar, mtime 0, uid/gid 0, mod 0644, sıkıştırmasız."""
    stage, path = Path(stage), Path(path)
    files = sorted((p.relative_to(stage).as_posix(), p) for p in stage.rglob("*") if p.is_file())
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, p in files:
            data = p.read_bytes()
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            tar.addfile(ti, io.BytesIO(data))
    raw = buf.getvalue()
    _put(path, raw)
    return hashlib.sha256(raw).hexdigest(), len(raw)


# --- senaryo koşusu ---------------------------------------------------------------------------


def run_scenarios(
    pipe: Pipeline,
    tracker_cfg: dict,
    cfg: MotionCfg,
    root: Path,
    scen: dict[str, list[dict]],
    *,
    probe: bool = False,
) -> dict[str, dict]:
    """Her senaryo yeni bir `SessionRef` oturumunda (`gate=advise`, takip açık, `frame` ile).
    Kare kaydı: `frame`, `reset_before`, `SessionRef.step` çıktısı ve Faz 3 sağlık alarmları
    (sıfırlamada alarmlar da sıfırlanır). `probe=True`: sınır kuralı sorunları `issues`'a."""
    root = Path(root)
    m = pipe.models
    out = {}
    for name, steps in scen.items():
        sess = SessionRef(pipe, tracker_cfg, cfg)
        pr = _Probe()
        if probe:
            sess.ego = _ProbeEgo(cfg, pr)
        frames, segments = [], [[]]
        for s in steps:
            if s["reset_before"]:
                sess.reset(s["reset_before"])
                segments.append([])
            pr.where = f"{name} kare {s['frame']}"
            r = sess.step(
                (root / s["rgb"]).read_bytes(),
                (root / s["ir"]).read_bytes(),
                gate="advise",
                frame=s["frame"],
            )
            frames.append({"frame": s["frame"], "reset_before": s["reset_before"], **r})
            segments[-1].append(r["health"])
            if probe:
                pr.add(check_margins(r["health"], m.gate, m.thresholds, **HEALTH_MARGINS))
                for ch in ("rgb", "ir"):
                    mad, thr = r["health"][ch]["diff_mad"], cfg.frozen.mad_max
                    lost = "lost" in r["health"][ch]["flags"]
                    if mad is not None and not lost and abs(mad - thr) < MAD_REL * thr:
                        pr.add([f"{ch} diff_mad {mad:.5f} eşiğe ({thr:g}) yakın"])
        alarms = [a for seg in segments for a in alarm_states(seg)]
        for fr, a in zip(frames, alarms, strict=True):
            fr["alarms"] = a
        if probe:
            for seg in segments:
                pr.issues += [
                    f"{name}: {v}"
                    for v in clip_violations(seg, m.gate, dist_margin=HEALTH_DIST_MARGIN)
                ]
        out[name] = {"frames": frames, "issues": pr.issues}
    return out


class _TraceEgo(Ego):
    """`Ego.update` çağrılarını kaydeder (yeniden oynatma için); sonucu değiştirmez."""

    def __init__(self, cfg: MotionCfg, calls: list[dict]):
        self.calls = calls
        super().__init__(cfg, Relocalizer(cfg))

    def update(self, gray_half, source, boxes, frame):
        res = super().update(gray_half, source, boxes, frame)
        self.calls.append(
            {
                "frame": frame,
                "gray": gray_half is not None,
                "source": source,
                "boxes": [[float(v) for v in b] for b in boxes],
                "ego": res.to_json(),
            }
        )
        return res


def ego_trace(
    pipe: Pipeline,
    tracker_cfg: dict,
    cfg: MotionCfg,
    root: Path,
    scen: dict[str, list[dict]],
) -> dict:
    """Her senaryo yeni bir `SessionRef` oturumunda; `Ego.update` girdileri (gri var mı, kaynak,
    kutular, kare) ve çıktısı (`EgoResult.to_json`) kaydedilir. Yeniden oynatma: aynı girdilerle
    `Ego` + `Relocalizer` aynı çıktıyı verir (`reset_before`: all → reset, tracks →
    break_continuity)."""
    root = Path(root)
    out = {}
    for name, steps in scen.items():
        sess = SessionRef(pipe, tracker_cfg, cfg)
        calls: list[dict] = []
        sess.ego = _TraceEgo(cfg, calls)
        rows = []
        for s in steps:
            if s["reset_before"]:
                sess.reset(s["reset_before"])
            sess.step(
                (root / s["rgb"]).read_bytes(),
                (root / s["ir"]).read_bytes(),
                gate="advise",
                frame=s["frame"],
            )
            c = calls[-1]
            rows.append(
                {
                    "frame": c["frame"],
                    "reset_before": s["reset_before"],
                    "ir": s["ir"] if c["gray"] else None,
                    "source": c["source"],
                    "boxes": c["boxes"],
                    "ego": c["ego"],
                }
            )
        out[name] = rows
    return {"motion_sha256": cfg.sha256, "scenarios": out}


def coverage(runs: dict[str, dict]) -> list[str]:
    """Senaryoların sınaması gereken nesil olayları (plan Görev 9 Step 1)."""
    ego = {k: [f["ego"] for f in v["frames"]] for k, v in runs.items()}
    out = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            out.append(msg)

    keys = sum(e["keyframe"] for e in ego["clean"][1:])
    need(keys >= 3, f"clean: anahtar kare değişimi {keys} < 3")
    need({e["map_gen"] for e in ego["clean"]} == {1}, "clean: nesil 1 dışına çıktı")
    for name in ("ir_drop", "long_gap", "seek"):
        rel = [e for e in ego[name] if e["relocalized"]]
        need(
            len(rel) == 1 and rel[0]["merge"]["to"] == 1,
            f"{name}: nesil 1'e tek birleşme yok ({len(rel)} yeniden konumlama)",
        )
        last = ego[name][-1]
        need(last["map_gen"] == 1 and not last["provisional"], f"{name}: son kare nesil 1 değil")
    i = next((k for k, f in enumerate(runs["short_gap"]["frames"]) if f["frame"] > SHORT_GAP[1]))
    need(ego["short_gap"][i]["relinked"], "short_gap: kayıptan dönüşte bağlama yok")
    need({e["map_gen"] for e in ego["short_gap"]} == {1}, "short_gap: nesil değişti")
    sc = ego["scene_change"]
    need(not any(e["merge"] for e in sc), "scene_change: birleşme var (yanlış bağlama)")
    need(sc[-1]["map_gen"] > 1 and not sc[-1]["provisional"], "scene_change: kalıcı yeni nesil yok")
    frozen = [f["frame"] for f in runs["ir_freeze"]["frames"] if f["health"]["ir"]["frozen"]]
    want = list(range(IR_FREEZE[0], IR_FREEZE[1] + 1))
    need(frozen == want, f"ir_freeze: donmuş kareler {frozen}")
    after = next(f for f in runs["reset_all"]["frames"] if f["reset_before"] == "all")["ego"]
    need(after["keyframe"] and after["map_gen"] == 1, "reset_all: sıfırlamadan sonra nesil 1 yok")
    return out


def track_coverage(runs: dict[str, dict], min_rows: int = MIN_TRACK_ROWS) -> list[str]:
    """`clean`'de en az `min_rows` iz × kare ve hem `moving` hem `stationary` kararı: Kabul 1'in
    iz / hareket eşliği ve bölge senaryosu boş kalmasın."""
    rows = [t["motion"] for f in runs["clean"]["frames"] for t in f["tracks"]]
    out = [] if len(rows) >= min_rows else [f"clean: iz × kare {len(rows)} < {min_rows}"]
    out += [f"clean: {m!r} kararı yok" for m in ("moving", "stationary") if m not in rows]
    return out


class _Cached:
    """Aynı girdi tensörüne aynı ORT çıktısı (başlangıç denemeleri arasında tekrar yok)."""

    def __init__(self, backend) -> None:
        self.backend, self._memo = backend, {}

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def infer(self, x: np.ndarray) -> np.ndarray:
        key = hashlib.sha1(x.tobytes()).digest()
        if key not in self._memo:
            self._memo[key] = self.backend.infer(x)
        return self._memo[key]


def build(
    models,
    dev_root: Path,
    dev_cfg: dict,
    cfg: MotionCfg,
    tracker_cfg: dict,
    stage: Path,
    *,
    backend=None,
    n: int = CLIP_LEN,
    log: Callable[[str], None] = print,
    start_from: int = 1,
    candidates: list[int] | None = None,
) -> dict:
    """Başlangıcı sınır kuralı ve nesil olaylarıyla seçer, paket içeriğini `stage`'e yazar.
    `candidates` verilirse `pick_best` (en az sorunlu, kapsamı tam); sorunlar `margins.json`'a."""
    if dev_cfg["split"] == "test" or SEQ not in dev_cfg["sequences"]:
        raise ValueError(f"fikstür yalnız val {SEQ}'den (config: {dev_cfg['split']})")
    seq = load_sequence(Path(dev_root) / "sequences" / SEQ, allow_test=False)
    stage = Path(stage)
    if stage.exists():
        shutil.rmtree(stage)
    if backend is None:
        from baykus.edge.backends import OrtBackend

        backend = OrtBackend(models.onnx)
    pipe = Pipeline(models, backend=_Cached(backend))
    scen = scenarios(n)
    frames = set(seq.frames)
    starts = [
        s for s in seq.frames if s >= start_from and all(f in frames for f in range(s, s + n))
    ]

    def check(start: int) -> list[str]:
        write_clip(seq.root, start, stage, n)
        # önce yalnız `clean` (ucuz ön eleme; diğer senaryolar onun önekleriyle başlar)
        first = run_scenarios(pipe, tracker_cfg, cfg, stage, {"clean": scen["clean"]}, probe=True)
        issues = first["clean"]["issues"]
        if not issues:
            runs = run_scenarios(pipe, tracker_cfg, cfg, stage, scen, probe=True)
            issues = [i for v in runs.values() for i in v["issues"]]
            issues += coverage(runs) + track_coverage(runs)
        log(f"başlangıç {start}: {len(issues)} sorun" + (f" — {issues[0]}" if issues else ""))
        return issues

    def evaluate(start: int) -> tuple[list[str], list[str]]:
        write_clip(seq.root, start, stage, n)
        runs = run_scenarios(pipe, tracker_cfg, cfg, stage, scen, probe=True)
        issues = [i for v in runs.values() for i in v["issues"]]
        missing = coverage(runs) + track_coverage(runs)
        log(f"başlangıç {start}: {len(issues)} sorun, kapsam eksiği {len(missing)}")
        return issues, missing

    if candidates:
        bad = sorted(set(candidates) - set(starts))
        if bad:
            raise ValueError(f"60 ardışık karesi olmayan başlangıçlar: {bad}")
        start, left, notes = pick_best(candidates, evaluate)
    else:
        (start, notes), left = pick_start(starts, check), []
    clip = write_clip(seq.root, start, stage, n)
    warps = write_warps(stage, n)
    _put_json(stage / "scenarios.json", scen)
    rule = {
        "inliers": INL_MARGIN,
        "ratio": RATIO_MARGIN,
        "shift_px": PX_MARGIN,
        "diff_mad_rel": MAD_REL,
        **HEALTH_MARGINS,
        "health_dist_px": HEALTH_DIST_MARGIN,
    }
    margins = {"start": start, "candidates": candidates, "rule": rule, "near": margin_records(left)}
    _put_json(stage / "margins.json", margins)
    meta = {
        "seq": SEQ,
        "start": start,
        "source": clip["source"],
        "warps": len(warps),
        "near": len(left),
        "best": bool(candidates),
    }
    (stage / "README.md").write_text(_readme(meta, notes), encoding="utf-8", newline="\n")
    return {**meta, "notes": notes}


def _readme(meta: dict, notes: list[str]) -> str:
    lines = [
        "# fixtures-f4 (üretici: `baykus-motion fixtures`)",
        "",
        f"Klip: {meta['source']} (60 kare, özgün baytlar). Seçim yalnız eşlik testi için "
        "(tasarım §5.9 sınır kuralı).",
        "",
        "Senaryolar (`scenarios.json`): " + ", ".join(f"`{s}`" for s in SCENARIOS) + ".",
        f"Dönüşümler (`warps.json`): {meta['warps']} IR görüntü.",
        *(
            [
                "",
                f"Eşiğe yakın kalan kayıt: {meta['near']} (`margins.json`). 1-541 arasında sınır "
                "kuralını tam geçen başlangıç yok (ego oranı anahtar kare eşiği 0,5'e sık sık "
                "±0,02 yakın); kullanıcı kararıyla (2026-10-21) kapsamı tam, en az sorunlu aday "
                "seçildi. C++ ego (plan Görev 11) bu kayıtlarda ölçülür; karar farkı çıkarsa "
                "fikstür yeniden seçilir.",
            ]
            if meta["best"]
            else []
        ),
        "",
        "Görüntüler M3OT veri setinden (Scientific Data 2025, figshare "
        "https://figshare.com/s/01fa8d1163f4e9a5a13a), lisans CC BY 4.0 "
        "(https://creativecommons.org/licenses/by/4.0/). Değişiklikler: yatay aynalama, sıfır "
        "görüntüler, benzerlik dönüşümleri ve ötelemeler.",
        "",
        "## Denenen başlangıçlar",
        *([f"- {n}" for n in notes] or ["- yok"]),
        "",
    ]
    return "\n".join(lines)


# --- hareket birim eşliği (Kabul 2) -----------------------------------------------------------


def maps_val(dev_root: Path, dets: dict, cfg: MotionCfg) -> dict:
    """Tam çözünürlük IR'den ego (yeniden konumlama dahil); maske takip katmanı tespitleri,
    `frame` dizi kare numarası. Sağlık / donmuş kare yok: kaynak her karede IR (val temiz)."""
    seqs = []
    for s in dets["sequences"]:
        if s["split"] == "test":
            raise ValueError("test bölümü fikstür olmaz (tasarım §5.9)")
        frames = load_sequence(Path(dev_root) / "sequences" / s["seq"], allow_test=False)
        ego = Ego(cfg, Relocalizer(cfg))
        rows = []
        for fr in s["frames"]:
            boxes = [tuple(d["box"]) for d in fr["dets"] if d["score_raw"] >= TRACK_FLOOR]
            r = ego.update(half_gray(frames.ir(fr["frame"])), "ir", boxes, fr["frame"])
            rows.append({"frame": fr["frame"], "ego": r.to_json()})
        seqs.append({"seq": s["seq"], "split": s["split"], "frames": rows})
    return {"source": "M3OT, CC BY 4.0", "motion_sha256": cfg.sha256, "sequences": seqs}


def ego_from_json(e: dict) -> EgoResult:
    def mat(v):
        return None if v is None else np.array(v, np.float64).reshape(2, 3)

    mg = e["merge"]
    return EgoResult(
        ok=e["ok"],
        source=e["source"],
        map_gen=e["map_gen"],
        provisional=e["provisional"],
        frame_to_map=mat(e["frame_to_map"]),
        inliers=e["inliers"],
        inlier_ratio=e["inlier_ratio"],
        keyframe=e["keyframe"],
        relinked=e["relinked"],
        relocalized=e["relocalized"],
        merge=None if mg is None else Merge(mg["from"], mg["to"], mat(mg["transform"])),
    )


def motion_expected(maps: dict, dets: dict, tracks: dict, cfg: MotionCfg) -> dict:
    """`maps_val` ego'su + takip fikstürü (takipçi kimlikleri, kimliği ≥ 0 olanlar) →
    kare başına `[track_id, motion, speed_rel, x, y]` (kimliğe göre sıralı)."""
    seqs = []
    for m, d, t in zip(maps["sequences"], dets["sequences"], tracks["sequences"], strict=True):
        if not m["seq"] == d["seq"] == t["seq"]:
            raise ValueError(f"dizi sırası farklı: {m['seq']}, {d['seq']}, {t['seq']}")
        ms = MotionState(cfg)
        rows = []
        for fm, fd, ft in zip(m["frames"], d["frames"], t["frames"], strict=True):
            if not fm["frame"] == fd["frame"] == ft["frame"]:
                raise ValueError(f"{m['seq']}: kare sırası farklı ({fm['frame']})")
            boxes = {x["index"]: tuple(x["box"]) for x in fd["dets"]}
            obs = [TrackObs(x["track_id"], boxes[x["index"]]) for x in ft["tracks"]
                   if x["track_id"] >= 0]  # fmt: skip
            res = ms.update(ego_from_json(fm["ego"]), obs)
            out = []
            for tid in sorted(res):
                r = res[tid]
                xy = r.map_xy or (None, None)
                out.append([tid, r.motion, r.speed_rel, xy[0], xy[1]])
            rows.append({"frame": fm["frame"], "tracks": out})
        seqs.append({"seq": m["seq"], "frames": rows})
    return {"motion_sha256": cfg.sha256, "sequences": seqs}


def gz_json(rec: dict) -> bytes:
    """Belirlenimci .json.gz (mtime 0)."""
    text = json.dumps(rec, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    return gzip.compress(text.encode("utf-8"), mtime=0)
