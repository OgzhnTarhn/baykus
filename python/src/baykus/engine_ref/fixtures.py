"""C++ eşlik fikstürleri (Faz 3 tasarımı §4.6, plan F3 Görev 14).

Yalnız M3OT geliştirme bölümünden (val; test asla). Sınır kuralı (üretici denetler): her sağlık
özelliği eşiğinden göreli ≥ 1e-3 uzak; `misalign_px` eşikten ≥ 5 px; `misalign_px ≥ t − 5` ise
`|response − r_min| ≥ 0,2`; klipte `S` karelerinde `|‖vᵢ − v̄‖ − 3| ≥ 2`. Yanıt `r_min − 0,2`'nin
altındaysa kayma kararı (`misaligned_frame`) hiçbir `misalign_px`'te değişmez: o karede kayma payı
aranmaz (karar güvenliği; M3OT'ta yanıt hep düşük, 2026-10-18 deney günlüğü). Tutmayan tek kare
yerine sonraki kare, tutmayan klip yerine bir sonraki başlangıç karesi seçilir. Ham skorlar için
eleme yok (Kabul 1'in satır düzeyindeki sınır kuralı). Seçim yalnız eşlik testi içindir.

Beklenen değerler bu Python referansından; girdiler dosya olarak (C++ rastgele sayı üretmez).
Her beklenti dosyası girdilerini yolla adlandırır; `fixtures/index.json` her dosyayı türüyle
listeler.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from baykus.engine_ref.compose import decode_pair
from baykus.engine_ref.models import Models
from baykus.engine_ref.pipeline import Pipeline, display
from baykus.health.calibrate import apply_isotonic
from baykus.health.features import frame_features
from baykus.health.gate import Thresholds, assess
from baykus.health.guard import GateModel
from baykus.health.m3ot import Alarms, alarm_states
from baykus.health.temporal import MisalignWindow
from baykus.robustness.corruptions import Condition, darken, gaussian_noise, image_rng, shift

REL_MARGIN = 1e-3  # 10 × göreli tolerans 1e-4
PX_MARGIN = 5.0  # 10 × 0,5 px
RESPONSE_MARGIN = 0.2
CLIP_DIST_MARGIN = 2.0
EVAL_FLOOR, TRACK_FLOOR, CONF = 0.001, 0.1, 0.25
N_FRAMES, CLIP_LEN, SHIFT_FROM = 10, 15, 8  # klipte 8. kareden (1'den sayınca) sonra kayma
SHIFT_PX, NOISE_SIGMA, DARKEN = 20, 20, 0.25
SEED = 42
SYNTHETIC = (
    "black_ir",
    "noise_ir",
    "shift_ir",
    "black_rgb",
    "darken_rgb",
    "resize_800x600",
    "tiny_33x17",
)


def check_margins(
    h: dict,
    gm: GateModel,
    th: Thresholds,
    *,
    rel_margin: float = REL_MARGIN,
    px_margin: float = PX_MARGIN,
    response_margin: float = RESPONSE_MARGIN,
) -> list[str]:
    """Varsayılan paylar Faz 3 §4.6 (10 × tolerans); Faz 4 fikstürleri ölçülmüş farkla daraltır."""
    rgb, ir, pair = h["rgb"], h["ir"], h["pair"]
    rel = (
        ("rgb_std", rgb["std"], th.rgb_lost),
        ("ir_std", ir["std"], th.ir_lost),
        ("ir_noise", ir["noise"], th.ir_noisy),
        ("rgb_blur", rgb["blur"], th.rgb_blurry),
        ("ir_blur", ir["blur"], th.ir_blurry),
        ("rgb_brightness", rgb["brightness"], th.dark),
    )
    out = [
        f"{name} {v:.6g} eşiğe ({t:.6g}) göreli {rel_margin:g}'ten yakın"
        for name, v, t in rel
        if v is not None and abs(v - t) < rel_margin * abs(t)
    ]
    px, resp = pair["misalign_px"], pair["response"]
    if resp is not None and resp + response_margin < gm.r_min:
        return out  # kare kuralı kaymayı her durumda reddeder: px kararı değiştiremez
    if px is not None and abs(px - th.misaligned) < px_margin:
        out.append(f"misalign_px {px:.3f} eşiğe ({th.misaligned:.3f}) {px_margin:g} px'ten yakın")
    if (
        px is not None
        and px >= th.misaligned - px_margin
        and resp is not None
        and abs(resp - gm.r_min) < response_margin
    ):
        out.append(f"response {resp:.4f} r_min'e ({gm.r_min:.4f}) {response_margin:g}'den yakın")
    return out


def clip_violations(
    healths: list[dict], gm: GateModel, *, dist_margin: float = CLIP_DIST_MARGIN
) -> list[str]:
    """Zaman kuralı penceresi boyunca S karelerinin ortancaya uzaklığı 3 px'e `dist_margin`'den
    (varsayılan 2 px) yakın mı."""
    buf: deque[tuple[bool, float | None, float | None]] = deque(maxlen=gm.window)
    out = []
    for i, h in enumerate(healths):
        p = h["pair"]
        buf.append((bool(p["misaligned_frame"]), p.get("dx_px"), p.get("dy_px")))
        s = np.array([(dx, dy) for f, dx, dy in buf if f], np.float64)
        if not len(s):
            continue
        dist = np.linalg.norm(s - np.median(s, axis=0), axis=1)
        if np.any(np.abs(dist - gm.tol_px) < dist_margin):
            out.append(f"kare {i}: S uzaklıkları {np.round(dist, 3).tolist()}")
    return out


def calibration_probe(x: np.ndarray, y: np.ndarray) -> list[dict]:
    """Tablonun her x'i için float32(x) ve iki float32 komşusu → kalibre skor (double)."""
    out = []
    for xi in x:
        f = np.float32(xi)
        probes = [np.nextafter(f, np.float32(-np.inf)), f, np.nextafter(f, np.float32(np.inf))]
        out.append(
            {
                "x": float(xi),
                "probes": [
                    {"score": float(p), "calibrated": float(apply_isotonic(np.float64(p), x, y))}
                    for p in probes
                ],
            }
        )
    return out


def _health(bgr: np.ndarray, ir: np.ndarray, m: Models) -> dict:
    return assess(frame_features(bgr, ir, vector=True), m.thresholds, m.gate.r_min)


def _layers(dets: list) -> dict:
    track = [d for d in dets if d.score_raw >= TRACK_FLOOR]
    return {
        "eval": [asdict(d) for d in dets],
        "track": [asdict(d) for d in track],
        "display": [asdict(d) for d in display(track, CONF)],
    }


def frame_expected(pipe: Pipeline, rgb_bytes: bytes, ir_bytes: bytes) -> dict:
    """İki blok: `modules` (kapı off, özgün pikseller) ve `pipeline` (kapı auto, oturumsuz)."""
    bgr, ir = decode_pair(rgb_bytes, ir_bytes)
    health = _health(bgr, ir, pipe.models)
    det = pipe.detect(bgr, ir, EVAL_FLOOR)
    lb = det["lb"]
    auto = pipe.run(rgb_bytes, ir_bytes, gate="auto", floor=EVAL_FLOOR)
    return {
        "modules": {
            "letterbox": {
                "gain": lb.gain,
                "pad_x": lb.pad_x,
                "pad_y": lb.pad_y,
                "sha256": det["tensor_sha256"],
            },
            "raw": det["raw"][0].astype(np.float64).tolist(),
            "dets": _layers(det["dets_all"]),
            "health": health,
        },
        "pipeline": {
            "applied": auto["health"]["applied"],
            "gate": auto["health"]["gate"],
            "letterbox_sha256": auto["tensor_sha256"],
            "dets": _layers(auto["dets_all"]),
        },
    }


def _synthetic(name: str, bgr: np.ndarray, ir: np.ndarray, stem: str) -> tuple:
    if name == "black_ir":
        return bgr, np.zeros_like(ir)
    if name == "noise_ir":
        rng = image_rng(SEED, stem, Condition("noise", float(NOISE_SIGMA), "ir"))
        return bgr, gaussian_noise(ir, NOISE_SIGMA, rng)
    if name == "shift_ir":
        return bgr, shift(ir, SHIFT_PX, 0)
    if name == "black_rgb":
        return np.zeros_like(bgr), ir
    if name == "darken_rgb":
        return darken(bgr, DARKEN), ir
    if name == "resize_800x600":
        size = (800, 600)
        return cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR), cv2.resize(
            ir, size, interpolation=cv2.INTER_LINEAR
        )
    if name == "tiny_33x17":
        size = (33, 17)
        return cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.resize(
            ir, size, interpolation=cv2.INTER_AREA
        )
    raise ValueError(f"bilinmeyen sentetik fikstür: {name}")


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise OSError("PNG kodlanamadı")
    return buf.tobytes()


class _Writer:
    def __init__(self, out: Path) -> None:
        self.out, self.index = Path(out), []

    def path(self, rel: str) -> Path:
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def bytes(self, rel: str, data: bytes) -> str:
        self.path(rel).write_bytes(data)
        return rel

    def json(self, rel: str, rec) -> str:
        text = json.dumps(rec, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
        self.path(rel).write_text(text, encoding="utf-8", newline="\n")
        return rel

    def add(self, name: str, kind: str, path: str, files: list[str] = ()) -> None:
        self.index.append({"name": name, "kind": kind, "path": path, "files": list(files)})


def _seq_frames(dev_root: Path, cfg: dict) -> list[tuple[str, int, Path]]:
    out = []
    for s in cfg["sequences"]:
        d = Path(dev_root) / "sequences" / s
        index = json.loads((d / "index.json").read_text(encoding="utf-8"))
        if index["split"] != cfg["split"]:
            raise ValueError(f"{s}: split {index['split']!r}, config {cfg['split']!r}")
        out += [(s, f, d) for f in index["frames"]]
    return out


def _read(d: Path, f: int) -> tuple[bytes, bytes]:
    return (d / "rgb" / f"{f:06d}.jpg").read_bytes(), (d / "ir" / f"{f:06d}.png").read_bytes()


def _pick_frames(frames, m: Models, n: int) -> tuple[list[int], list[str]]:
    """Eşit aralıklı n konum; tutmayan konumdan sonraki karelere geçilir."""
    chosen, notes = [], []
    for j in range(n):
        start = (2 * j + 1) * len(frames) // (2 * n)
        for k in range(start, len(frames)):
            if k in chosen:
                continue
            s, f, d = frames[k]
            v = check_margins(_health(*decode_pair(*_read(d, f)), m), m.gate, m.thresholds)
            if not v:
                chosen.append(k)
                break
            notes.append(f"{s} kare {f} atlandı: {v[0]}")
        else:
            raise ValueError(f"{j}. konumdan sonra sınır kuralını geçen kare yok")
    return chosen, notes


def _variant_ir(variant: str, ir: np.ndarray, pos: int) -> np.ndarray | None:
    """Klip IR değişkeni; değişmeyen karede None (temiz dosya kullanılır)."""
    if variant == "shift_all" or (variant == "shift_from8" and pos + 1 >= SHIFT_FROM):
        return shift(ir, SHIFT_PX, 0)
    return None


CLIP_VARIANTS = ("clean", "shift_all", "shift_from8")


def _clip_ok(seq_frames, m: Models) -> list[str]:
    decoded = [decode_pair(*_read(d, f)) for _, f, d in seq_frames]
    issues = []
    for v in CLIP_VARIANTS:
        hs = []
        for pos, (bgr, ir) in enumerate(decoded):
            alt = _variant_ir(v, ir, pos)
            h = _health(bgr, ir if alt is None else alt, m)
            issues += [f"{v} {pos + 1}: {x}" for x in check_margins(h, m.gate, m.thresholds)]
            hs.append(h)
        issues += [f"{v}: {x}" for x in clip_violations(hs, m.gate)]
    return issues


def _pick_clip(frames, m: Models) -> tuple[list, list[str]]:
    notes = []
    by_seq: dict[str, list] = {}
    for item in frames:
        by_seq.setdefault(item[0], []).append(item)
    for seq, items in by_seq.items():
        for k0 in range(len(items) - CLIP_LEN + 1):
            window = items[k0 : k0 + CLIP_LEN]
            nums = [f for _, f, _ in window]
            if nums != list(range(nums[0], nums[0] + CLIP_LEN)):
                continue  # ardışık değil
            issues = _clip_ok(window, m)
            if not issues:
                summary = [f"klip: {len(notes)} başlangıç karesi sınır kuralını geçmedi"]
                return window, summary + [f"  örnek {n}" for n in notes[:3]]
            notes.append(f"klip {seq} {nums[0]}: {issues[0]}")
    raise ValueError("sınır kuralını geçen 15 karelik klip yok: " + "; ".join(notes[-3:]))


def temporal_unit(gm: GateModel) -> list[dict]:
    """Zaman kuralı senaryoları (girdi: işaret, vektör, kayıp/gürültü; çıktı: misaligned, kapı)."""

    def h(flag=True, dx=20.0, dy=0.0, rgb_lost=False, ir_lost=False, ir_noisy=False):
        return {
            "rgb": {"flags": ["lost"] if rgb_lost else []},
            "ir": {"flags": (["lost"] if ir_lost else []) + (["noisy"] if ir_noisy else [])},
            "pair": {"misaligned_frame": flag, "dx_px": dx, "dy_px": dy, "misaligned": flag},
        }

    def cycle(pattern: list, n: int = 50) -> list:
        return [pattern[i % len(pattern)] for i in range(n)]

    def golden(k: int) -> dict:  # 40 px yarıçapta altın açı adımı: penceredeki iki vektör ≥ 35 px
        a = math.radians(137.5 * k)
        return h(dx=40.0 * math.cos(a), dy=40.0 * math.sin(a))

    off = h(flag=False, dx=0.0)
    # Tasarım §6.1 Kabul 3 b: 50 karelik diziler (sıfırlama adımları kare sayılmaz)
    scenarios = {
        "constant": [h() for _ in range(50)],
        "jumping": [golden(k) for k in range(50)],
        "onset_mid": [h(flag=False, dx=None, dy=None) for _ in range(10)]
        + [off for _ in range(15)]
        + [h() for _ in range(25)],
        # ortanca (0, 0); (3, 0) ve (1,8, 2,4) tam 3 px (sınır dahil), 3,000001 px dışarıda
        "boundary_3px": cycle([h(dx=0.0), h(dx=0.0), h(dx=3.0), off, off], 20)
        + cycle([h(dx=0.0), h(dx=0.0), h(dx=1.8, dy=2.4), off, off], 15)
        + cycle([h(dx=0.0), h(dx=0.0), h(dx=3.000001), off, off], 15),
        # |S| = 4: ortanca ortadaki iki değerin ortalaması (3 / 3,25 px)
        "even_S": cycle([h(dx=0.0), h(dx=0.0), h(dx=6.0), h(dx=6.0), off], 25)
        + cycle([h(dx=0.0), h(dx=0.0), h(dx=6.5), h(dx=6.5), off], 25),
        "unflagged_keeps": cycle([h(), h(), h(), off, off], 25) + cycle([h(), off], 25),
        "lost_overrides": [h(ir_lost=k in (10, 11), rgb_lost=k in (30, 41)) for k in range(50)],
        "noisy_priority": [h(ir_noisy=k in (5, 6, 7, 8, 9, 40)) for k in range(50)],
        "reset": [h() for _ in range(12)]
        + [None]  # None = oturum sıfırlama
        + [h() for _ in range(18)]
        + [None]
        + [h()]
        + [None, None]
        + [h() for _ in range(19)],
    }
    out = []
    for name, steps in scenarios.items():
        w = MisalignWindow(gm.window, gm.min_frames, gm.tol_px)
        recs = []
        for s in steps:
            if s is None:
                w.reset()
                recs.append({"reset": True})
                continue
            r = w.update(s)
            recs.append(
                {
                    "input": s,
                    "expected": {
                        "misaligned": r["pair"]["misaligned"],
                        "gate": r["gate"],
                        "reason_code": r["reason_code"],
                        "reason": r["reason"],
                    },
                }
            )
        out.append({"name": name, "steps": recs})
    return out


def raw_reference(pipe: Pipeline, fixtures: Path) -> dict:
    """Kare fikstürlerinin ham ORT çıktısı (`modules.raw` ile aynı yol: kapı off, özgün pikseller),
    bu makinede yeniden. CI Kabul 1 a'yı C++ ile aynı işlemcide karşılaştırır (ONNX Runtime'ın
    işlemciye göre yuvarlama farkı bizim kodumuz değil)."""
    import onnxruntime

    fixtures = Path(fixtures)
    index = json.loads((fixtures / "index.json").read_text(encoding="utf-8"))
    frames = {}
    for e in index:
        if e["kind"] != "frame":
            continue
        inputs = json.loads((fixtures / e["path"]).read_text(encoding="utf-8"))["inputs"]
        bgr, ir = decode_pair(
            (fixtures / inputs["rgb"]).read_bytes(), (fixtures / inputs["ir"]).read_bytes()
        )
        frames[e["name"]] = pipe.detect(bgr, ir, EVAL_FLOOR)["raw"][0].astype(np.float64).tolist()
    return {"onnxruntime": onnxruntime.__version__, "frames": frames}


def alarm_sequences() -> list[dict]:
    """§4.4 alarm sözleşmesi senaryoları (UI `lib/alarms.ts` ve motor aynı sonucu vermeli)."""

    def h(rgb_lost=False, ir_lost=False, ir_noisy=False, mis=False):
        return {
            "rgb": {"flags": ["lost"] if rgb_lost else []},
            "ir": {"flags": (["lost"] if ir_lost else []) + (["noisy"] if ir_noisy else [])},
            "pair": {"misaligned": mis},
        }

    lost, ok = h(ir_lost=True), h()
    scenarios = {
        "lost_open_clear": [lost, lost, lost, ok, ok, ok, ok],
        "noise": [h(ir_noisy=True)] * 4 + [ok] * 3,
        "multi": [h(rgb_lost=True, ir_noisy=True, mis=True)] * 3,
        "misaligned_current_frame": [h(mis=True), ok, h(mis=True)],
        "reset": [lost, lost, lost, None, lost, ok],
    }
    out = []
    for name, steps in scenarios.items():
        a = Alarms()
        recs = []
        for s in steps:
            if s is None:
                a.reset()
                recs.append({"reset": True})
            else:
                recs.append({"input": s, "expected": a.update(s)})
        out.append({"name": name, "steps": recs})
    return out


NOTICE = """# M3OT fikstürleri: atıf ve lisans

Bu klasördeki görüntüler (`frames/`, `clip/`) ve `../synthetic/`, `../tracks/` içindeki veriler
**M3OT** veri setinden türetilmiştir:

- Kaynak: M3OT, *Scientific Data* 2025 · figshare https://figshare.com/s/01fa8d1163f4e9a5a13a
- Lisans: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). Repo AGPL olsa da bu
  dosyalar CC BY 4.0 altında kalır.
- Yalnız geliştirme bölümü (val); test dizileri fikstür olmaz.
- Değişiklikler: RGB PNG → JPEG; IR tek kanal PNG; sentetik bozulmalar (siyah IR, IR gürültü σ 20,
  IR 20 px kaydırma, siyah RGB, RGB karartma × 0,25, 800×600 ve 33×17 yeniden boyutlandırma);
  takip fikstüründe yalnız modelin tespitleri (ham skor ≥ 0,1).
"""


def _readme(cfg: dict, frames: list[str], synth_base: str, clip: str, notes: list[str]) -> str:
    lines = [
        "# Eşlik fikstürleri (üretici: `baykus-engine fixtures`)",
        "",
        f"Bölüm: `{cfg['split']}`, diziler {cfg['sequences']}. Seçim yalnız eşlik testi için "
        "(tasarım §4.6 sınır kuralı); model ya da eşik seçimi değil.",
        "",
        "## Tek kareler",
        *[f"- `{f}`" for f in frames],
        "",
        f"## Sentetik durumlar\n\nTaban kare: `{synth_base}`. {', '.join(SYNTHETIC)}.",
        "",
        f"## Klip\n\n{clip}; değişkenler `clean`, `shift_all`, `shift_from8` "
        "(IR `shift(ir, 20, 0)`).",
        "",
        "## Atlananlar (sınır kuralı)",
        *([f"- {n}" for n in notes] or ["- yok"]),
        "",
    ]
    return "\n".join(lines)


def generate(
    models: Models,
    dev_root: Path,
    cfg: dict,
    out: Path,
    *,
    backend=None,
    n_frames: int = N_FRAMES,
) -> list[dict]:
    if cfg["split"] == "test":
        raise ValueError("test bölümü fikstür olmaz (tasarım §4.6)")
    out = Path(out)
    for sub in ("m3ot", "synthetic"):
        shutil.rmtree(out / sub, ignore_errors=True)
    for old in (out / "expected").glob("*.json") if (out / "expected").exists() else []:
        if old.name not in ("lsap_cases.json", "kalman_steps.json"):
            old.unlink()
    w = _Writer(out)
    pipe = Pipeline(models, backend=backend)
    frames = _seq_frames(Path(dev_root), cfg)

    chosen, notes = _pick_frames(frames, models, n_frames)
    names = []
    for k in chosen:
        s, f, d = frames[k]
        rgb, ir = _read(d, f)
        name = f"{s}_{f:06d}"
        names.append(name)
        files = [
            w.bytes(f"m3ot/frames/{name}_rgb.jpg", rgb),
            w.bytes(f"m3ot/frames/{name}_ir.png", ir),
        ]
        rec = {"inputs": {"rgb": files[0], "ir": files[1]}, **frame_expected(pipe, rgb, ir)}
        w.add(name, "frame", w.json(f"expected/{name}.json", rec), files)

    synth_base = None
    for k in chosen:
        s, f, d = frames[k]
        bgr, ir = decode_pair(*_read(d, f))
        stem = f"{s}_{f:06d}"
        encoded = {}
        for name in SYNTHETIC:
            b2, i2 = _synthetic(name, bgr, ir, stem)
            encoded[name] = (_png(b2), _png(i2))
        bad = {
            n: v
            for n, (rb, ib) in encoded.items()
            if (
                v := check_margins(
                    _health(*decode_pair(rb, ib), models), models.gate, models.thresholds
                )
            )
        }
        if not bad:
            synth_base = stem
            break
        notes.append(f"sentetik taban {stem} atlandı: {next(iter(bad.items()))}")
    if synth_base is None:
        raise ValueError("sentetik fikstürler için sınır kuralını geçen taban kare yok")
    for name, (rb, ib) in encoded.items():
        files = [w.bytes(f"synthetic/{name}_rgb.png", rb), w.bytes(f"synthetic/{name}_ir.png", ib)]
        rec = {"inputs": {"rgb": files[0], "ir": files[1]}, "base": synth_base,
               **frame_expected(pipe, rb, ib)}  # fmt: skip
        w.add(name, "frame", w.json(f"expected/{name}.json", rec), files)

    window, clip_notes = _pick_clip(frames, models)
    notes += clip_notes
    seq, f0 = window[0][0], window[0][1]
    clip_dir = "m3ot/clip/sequences/clip"
    base_files, var_files = [], {v: {} for v in CLIP_VARIANTS[1:]}
    for pos, (_, f, d) in enumerate(window):
        rgb, ir = _read(d, f)
        n = pos + 1
        base_files.append(
            (w.bytes(f"{clip_dir}/rgb/{n:06d}.jpg", rgb), w.bytes(f"{clip_dir}/ir/{n:06d}.png", ir))
        )
        ir_img = decode_pair(rgb, ir)[1]
        for v in CLIP_VARIANTS[1:]:
            alt = _variant_ir(v, ir_img, pos)
            if alt is not None:
                var_files[v][n] = w.bytes(f"m3ot/clip/variants/{v}/{n:06d}.png", _png(alt))
    size = list(decode_pair(*_read(window[0][2], window[0][1]))[0].shape[1::-1])
    index_rel = w.json(
        f"{clip_dir}/index.json",
        {
            "name": "clip",
            "split": cfg["split"],
            "frames": list(range(1, CLIP_LEN + 1)),
            "fps": 30,
            "size": size,
            "source": f"M3OT {cfg['split']} {seq} kare {f0}-{f0 + CLIP_LEN - 1}, CC BY 4.0",
        },
    )
    for v in CLIP_VARIANTS:
        win = MisalignWindow(models.gate.window, models.gate.min_frames, models.gate.tol_px)
        recs, healths = [], []
        files = [index_rel] if v == "clean" else []
        for pos, (rgb_rel, ir_rel) in enumerate(base_files):
            n = pos + 1
            ir_rel_v = var_files.get(v, {}).get(n, ir_rel)
            r = pipe.run(
                (out / rgb_rel).read_bytes(),
                (out / ir_rel_v).read_bytes(),
                gate="auto",
                floor=TRACK_FLOOR,
                window=win,
            )
            healths.append(r["health"])
            recs.append({"frame": n, "rgb": rgb_rel, "ir": ir_rel_v, "health": r["health"]})
            if v == "clean":
                files += [rgb_rel, ir_rel]
            elif ir_rel_v != ir_rel:
                files.append(ir_rel_v)
        for rec, alarms in zip(recs, alarm_states(healths), strict=True):
            rec["alarms"] = alarms
        frames_out = [{k: r[k] for k in ("frame", "health", "alarms")} for r in recs]
        inputs = [{"frame": r["frame"], "rgb": r["rgb"], "ir": r["ir"]} for r in recs]
        rel = w.json(
            f"expected/clip_{v}.json",
            {"variant": v, "gate": "auto", "inputs": inputs, "frames": frames_out},
        )
        w.add(f"clip_{v}", "clip", rel, files)

    probe = calibration_probe(models.calib_x, models.calib_y)
    w.add(
        "calibration_probe", "calibration_probe", w.json("expected/calibration_probe.json", probe)
    )
    w.add(
        "temporal_unit",
        "temporal_unit",
        w.json("expected/temporal_unit.json", temporal_unit(models.gate)),
    )
    w.add(
        "alarm_sequences",
        "alarm_sequences",
        w.json("expected/alarm_sequences.json", alarm_sequences()),
    )
    for name, kind, rel in (
        ("lsap_cases", "lsap", "expected/lsap_cases.json"),
        ("kalman_steps", "kalman", "expected/kalman_steps.json"),
        ("dev_dets", "tracks", "tracks/dev_dets.json.gz"),
        ("dev_tracks_py", "tracks", "tracks/dev_tracks_py.json.gz"),
    ):
        if (out / rel).exists():
            w.add(name, kind, rel)

    clip_txt = f"`{seq}` kare {f0}-{f0 + CLIP_LEN - 1}"
    (out / "m3ot").mkdir(parents=True, exist_ok=True)
    (out / "m3ot" / "NOTICE.md").write_text(NOTICE, encoding="utf-8", newline="\n")
    readme = _readme(cfg, names, synth_base, clip_txt, notes)
    (out / "m3ot" / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    text = json.dumps(w.index, ensure_ascii=False, indent=1) + "\n"
    (out / "index.json").write_text(text, encoding="utf-8", newline="\n")
    return w.index
