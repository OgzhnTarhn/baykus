"""Kör insan örneklemi (Faz 4 tasarımı §5.3, Kabul 4).

Nüfus etiketleyiciden bağımsız: aracın `t ± K` aralığında görünür olduğu bütün GT örnekleri.
Örneklem 80 düzgün rastgele (asıl ölçüt) + 40 hedefli (etiketleyicinin bant / kayıtsız dediği
ya da etiket değişiminin ±K çevresindeki örnekler; bilgi için), iz başına en çok 3, seed 42.
Sayfa yerel ve internetsiz; etiket, hız ve örnek türü gösterilmez; kırpıntılar telafisizdir
(bir kestiricinin kaydı kullanılsaydı doğrulama döngüsel olurdu). Puanlama yalnız 80 düzgün
örnekte:
(a) insan ve etiketleyicinin karar verdiği örneklerde uyum ≥ %95, (b) "emin değilim" ≤ %15,
(c) insan-kararlı örneklerin ≥ %80'i etiketleyicinin etiketli nüfusunda.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from baykus.io import imwrite_any
from baykus.motion.images import Box
from baykus.motion.labels import Frames, LabelRow

N_UNIFORM, N_TARGETED, PER_TRACK, SEED = 80, 40, 3, 42
STEP, CROP = 3, 192
A_MIN, B_MAX, C_MIN = 0.95, 0.15, 0.80
ANSWERS = ("moving", "stationary", "unsure")
DECIDED = ("moving", "stationary")


class Sample(NamedTuple):
    seq: str
    frame: int
    track_id: int
    kind: str  # "uniform" | "targeted" (yalnız anahtar dosyasında; sayfada yok)


Key = tuple[str, int, int]


def population(
    gt: dict[str, dict[int, dict[int, Box]]], frames: dict[str, list[int]], k: int = 15
) -> list[Key]:
    out: list[Key] = []
    for seq, tracks in gt.items():
        present = set(frames[seq])
        for tid, per in tracks.items():
            for t in per:
                if all(f in per and f in present for f in (t - k, t, t + k)):
                    out.append((seq, t, tid))
    return sorted(out)


def _targeted(key: Key, labels: dict[Key, LabelRow], k: int) -> bool:
    row = labels.get(key)
    if row is None:
        return False
    if row.label in ("band", "unregistered"):
        return True
    seq, t, tid = key
    for f in range(t - k, t + k + 1):
        other = labels.get((seq, f, tid))
        if other is not None and other.label in (*DECIDED, "band") and other.label != row.label:
            return True
    return False


def sample(
    pop: list[Key],
    labels: dict[Key, LabelRow],
    seed: int = SEED,
    n_uniform: int = N_UNIFORM,
    n_targeted: int = N_TARGETED,
    per_track: int = PER_TRACK,
    k: int = 15,
) -> list[Sample]:
    rng = random.Random(seed)
    count: dict[tuple[str, int], int] = {}
    chosen: set[Key] = set()

    def take(cands: list[Key], n: int, kind: str) -> list[Sample]:
        got: list[Sample] = []
        for key in cands:
            if len(got) == n:
                break
            tr = (key[0], key[2])
            if key in chosen or count.get(tr, 0) >= per_track:
                continue
            chosen.add(key)
            count[tr] = count.get(tr, 0) + 1
            got.append(Sample(key[0], key[1], key[2], kind))
        return got

    uni = list(pop)
    rng.shuffle(uni)
    out = take(uni, n_uniform, "uniform")
    tgt = [key for key in pop if _targeted(key, labels, k)]
    rng.shuffle(tgt)
    return out + take(tgt, n_targeted, "targeted")


def _window(cx: float, cy: float, w: int, h: int, size: int) -> tuple[int, int]:
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    return min(max(x0, 0), max(w - size, 0)), min(max(y0, 0), max(h - size, 0))


def _crop(img: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    c = img[y0 : y0 + size, x0 : x0 + size]
    if c.shape[0] == size and c.shape[1] == size:
        return c.copy()
    pad = np.zeros((size, size, *img.shape[2:]), img.dtype)
    pad[: c.shape[0], : c.shape[1]] = c
    return pad


def render(
    samples: list[Sample],
    seqs: dict[str, Frames],
    gts: dict[str, dict[int, dict[int, Box]]],
    out_dir: Path,
    k: int = 15,
    seed: int = SEED,
) -> dict[str, Sample]:
    """Sayfa + kırpıntılar; dönen sözlük örnek kimliği → örnek (anahtar dosyasına yazılır)."""
    out_dir = Path(out_dir)
    crops = out_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    order = list(samples)
    random.Random(seed + 1).shuffle(order)  # sayfa sırası türden bağımsız
    ids = {f"a{i + 1:03d}": s for i, s in enumerate(order)}
    offsets = list(range(-k, k + 1, STEP))
    for sid, s in ids.items():
        seq, per = seqs[s.seq], gts[s.seq][s.track_id]
        x1, y1, x2, y2 = per[s.frame]
        for j, off in enumerate(offsets):
            f = s.frame + off
            ir = seq.ir(f)
            rgb = seq.rgb(f)
            x0, y0 = _window((x1 + x2) / 2, (y1 + y2) / 2, ir.shape[1], ir.shape[0], CROP)
            b = per.get(f)
            for ch, img, color in (("ir", ir, 255), ("rgb", rgb, (0, 255, 255))):
                c = _crop(img, x0, y0, CROP)
                if b is not None:
                    p1 = (int(round(b[0] - x0)) - 2, int(round(b[1] - y0)) - 2)
                    p2 = (int(round(b[2] - x0)) + 1, int(round(b[3] - y0)) + 1)
                    cv2.rectangle(c, p1, p2, color, 1)
                imwrite_any(crops / f"{sid}_{j:02d}_{ch}.jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 90])
    page = PAGE.replace("__SAMPLES__", json.dumps([{"id": i} for i in ids]))
    page = page.replace("__FRAMES__", str(len(offsets)))
    (out_dir / "index.html").write_text(page, encoding="utf-8", newline="\n")
    return ids


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - m, c + m


def _ratio(k: int, n: int, ok: bool | None) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi], "pass": ok}


def score(ids: dict[str, Sample], labels: dict[Key, LabelRow], answers: dict[str, str]) -> dict:
    missing = [i for i in ids if answers.get(i) not in ANSWERS]
    if missing:
        raise ValueError(f"eksik ya da geçersiz cevap: {', '.join(missing[:10])}")
    uni = [(i, s) for i, s in ids.items() if s.kind == "uniform"]
    unsure = sum(answers[i] == "unsure" for i, _ in uni)
    human = [(i, s) for i, s in uni if answers[i] in DECIDED]
    lab = {i: labels[(s.seq, s.frame, s.track_id)].label for i, s in ids.items()}
    both = [i for i, _ in human if lab[i] in DECIDED]
    agree = sum(answers[i] == lab[i] for i in both)
    a_ok = bool(both) and agree / len(both) >= A_MIN
    b_ok = bool(uni) and unsure / len(uni) <= B_MAX
    c_ok = bool(human) and len(both) / len(human) >= C_MIN
    targeted: dict[str, dict[str, int]] = {}
    for i, s in ids.items():
        if s.kind == "targeted":
            row = targeted.setdefault(answers[i], {})
            row[lab[i]] = row.get(lab[i], 0) + 1
    return {
        "a_agreement": _ratio(agree, len(both), a_ok),
        "b_unsure": {**_ratio(unsure, len(uni), b_ok), "ci95": None},
        "c_covered": _ratio(len(both), len(human), c_ok),
        "targeted": targeted,
        "n_uniform": len(uni),
        "n_targeted": sum(s.kind == "targeted" for s in ids.values()),
        "thresholds": {"a_min": A_MIN, "b_max": B_MAX, "c_min": C_MIN},
        "verdict": "pass" if a_ok and b_ok and c_ok else "fail",
    }


PAGE = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hareket kör örneklemi</title>
<style>
:root { --bg: #f6f6f4; --fg: #1d1d1b; --muted: #6b6b66; --line: #d8d8d2; --accent: #2f5d8a; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161615; --fg: #ecece8; --muted: #a0a09a; --line: #3a3a37; --accent: #7fb0e0; }
}
* { box-sizing: border-box; }
body { margin: 0; font: 16px/1.45 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
main { max-width: 860px; margin: 0 auto; padding: 16px; }
h1 { font-size: 20px; margin: 0 0 4px; }
p { color: var(--muted); margin: 4px 0 12px; }
.views { display: flex; gap: 12px; flex-wrap: wrap; }
.views figure { margin: 0; }
.views img { width: 384px; max-width: 100%; height: auto; image-rendering: pixelated;
  border: 1px solid var(--line); display: block; }
figcaption { font-size: 13px; color: var(--muted); }
.bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 12px 0; }
button { font: inherit; padding: 8px 14px; border: 1px solid var(--line); background: transparent;
  color: var(--fg); border-radius: 6px; cursor: pointer; }
button.on { border-color: var(--accent); outline: 2px solid var(--accent); }
.grow { flex: 1; }
#done { color: var(--muted); }
</style>
</head>
<body>
<main>
<h1>Araç hareket ediyor mu?</h1>
<p>Her örnekte aynı aracın yaklaşık bir saniyelik görüntüsü oynar (kutu araçtır). Kamera da hareket
ediyor; aracı yol, çizgi ve park etmiş araçlar gibi arka plana göre değerlendir. Emin değilsen
"Emin değilim" seç. Kısayollar: 1 hareketli, 2 duran, 3 emin değilim, ← → gezin.</p>
<div class="bar"><strong id="pos"></strong><span class="grow"></span><span id="done"></span></div>
<div class="views">
  <figure><img id="ir" alt="IR"><figcaption>IR</figcaption></figure>
  <figure><img id="rgb" alt="RGB">
    <figcaption>RGB (kutu IR etiketinden; kayabilir)</figcaption></figure>
</div>
<div class="bar">
  <button data-a="moving">1 · Hareketli</button>
  <button data-a="stationary">2 · Duran</button>
  <button data-a="unsure">3 · Emin değilim</button>
  <span class="grow"></span>
  <button id="prev">←</button><button id="next">→</button>
</div>
<div class="bar"><button id="save">Cevapları indir (JSON)</button></div>
</main>
<script>
var SAMPLES = __SAMPLES__;
var FRAMES = __FRAMES__;
var STORE = "baykus-audit-" + SAMPLES.length + "-" + (SAMPLES[0] || {}).id;
var answers = {};
try { answers = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) { answers = {}; }
var cur = 0, tick = 0;
function pad(n) { return (n < 10 ? "0" : "") + n; }
function show() {
  var s = SAMPLES[cur];
  document.getElementById("pos").textContent = (cur + 1) + " / " + SAMPLES.length;
  var n = 0;
  for (var i = 0; i < SAMPLES.length; i++) { if (answers[SAMPLES[i].id]) n++; }
  document.getElementById("done").textContent = n + " cevaplandı";
  var btns = document.querySelectorAll("button[data-a]");
  for (var j = 0; j < btns.length; j++) {
    btns[j].className = answers[s.id] === btns[j].getAttribute("data-a") ? "on" : "";
  }
}
function frame() {
  var s = SAMPLES[cur];
  var f = pad(tick % FRAMES);
  document.getElementById("ir").src = "crops/" + s.id + "_" + f + "_ir.jpg";
  document.getElementById("rgb").src = "crops/" + s.id + "_" + f + "_rgb.jpg";
  tick++;
}
function answer(a) {
  answers[SAMPLES[cur].id] = a;
  try { localStorage.setItem(STORE, JSON.stringify(answers)); } catch (e) {}
  if (cur < SAMPLES.length - 1) { cur++; tick = 0; }
  show();
}
var btns = document.querySelectorAll("button[data-a]");
for (var b = 0; b < btns.length; b++) {
  btns[b].onclick = function () { answer(this.getAttribute("data-a")); };
}
document.getElementById("prev").onclick = function () { if (cur > 0) { cur--; tick = 0; show(); } };
document.getElementById("next").onclick = function () {
  if (cur < SAMPLES.length - 1) { cur++; tick = 0; show(); }
};
document.onkeydown = function (e) {
  if (e.key === "1") answer("moving");
  if (e.key === "2") answer("stationary");
  if (e.key === "3") answer("unsure");
  if (e.key === "ArrowLeft") document.getElementById("prev").onclick();
  if (e.key === "ArrowRight") document.getElementById("next").onclick();
};
document.getElementById("save").onclick = function () {
  var out = SAMPLES.map(function (s) { return { id: s.id, answer: answers[s.id] || null }; });
  var blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" });
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "audit_answers.json";
  a.click();
};
show();
frame();
setInterval(frame, 250);
</script>
</body>
</html>
"""
