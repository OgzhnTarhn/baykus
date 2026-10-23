"""Araç türü kör denetimi (Faz 4 tasarımı §3 B, öğretmen etiketi).

M3OT train izlerinden öğretmen etiketlerinin (E3c tahmini) insan denetimi. Örneklem: tahmini
`car` olmayan izlerden en çok `non_car_max` (sınıf başına sayılarıyla orantılı), kalanı `car`
izlerinden; iz anahtarları sıralı listeden seçilir, deterministik. Sayfa yerel ve internetsiz;
tahmin gösterilmez (kör); IR ve RGB kırpıntısı 4× büyütülür. Puanlama `unknown` hariç
cevaplananlarda tahminle aynı oranı; doğruluk ≥ 0,80 → öğretmen etiketleri, altında → bütün
M3OT kutuları `car` (tasarım §3 B).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from baykus.classes import CLASS_NAMES
from baykus.io import imwrite_any
from baykus.motion.audit import wilson

ANSWERS = ("car", "truck", "bus", "van", "freight_car", "unknown")
ACC_MIN = 0.80
SCALE = 4


def _allocate(counts: dict[int, int], total: int) -> dict[int, int]:
    """`total` yuvayı sınıf sayılarına orantılı dağıtır; yuvarlama artıkları en büyük artığa
    (eşitlikte büyük sınıfa, o da eşitse küçük `cls_id`'ye) gider, sınıf sayısını aşmaz."""
    alloc = {c: 0 for c in counts}
    if total <= 0 or not counts:
        return alloc
    s = sum(counts.values())
    exact = {c: total * v / s for c, v in counts.items()}
    for c in counts:
        alloc[c] = min(int(math.floor(exact[c])), counts[c])
    left = total - sum(alloc.values())
    order = sorted(counts, key=lambda c: (-(exact[c] - math.floor(exact[c])), -counts[c], c))
    while left > 0:
        progressed = False
        for c in order:
            if left == 0:
                break
            if alloc[c] < counts[c]:
                alloc[c] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def sample(
    teacher: dict[str, dict],
    n: int = 100,
    non_car_max: int = 60,
    seed: int = 42,
) -> list[str]:
    """`teacher`: iz anahtarı → {"cls_id": int, ...}. Dönen liste iz anahtarları (deterministik)."""
    rng = random.Random(seed)
    keys = sorted(teacher)
    non_car = [k for k in keys if teacher[k]["cls_id"] != 0]
    car = [k for k in keys if teacher[k]["cls_id"] == 0]
    by_cls: dict[int, list[str]] = {}
    for k in non_car:
        by_cls.setdefault(teacher[k]["cls_id"], []).append(k)
    want = min(non_car_max, len(non_car), n)
    alloc = _allocate({c: len(v) for c, v in by_cls.items()}, want)
    picked: list[str] = []
    for cls in sorted(alloc):
        pool = list(by_cls[cls])
        rng.shuffle(pool)
        picked.extend(pool[: alloc[cls]])
    rest = n - len(picked)
    pool = list(car)
    rng.shuffle(pool)
    picked.extend(pool[:rest])
    return picked


def render(
    keys: list[str],
    crops: dict[str, tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    seed: int = 42,
) -> dict[str, str]:
    """Sayfa + kırpıntılar; dönen sözlük örnek kimliği → iz anahtarı."""
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    order = list(keys)
    random.Random(seed + 1).shuffle(order)  # sayfa sırası tahminden bağımsız
    ids = {f"t{i + 1:03d}": k for i, k in enumerate(order)}
    for sid, key in ids.items():
        ir, rgb = crops[key]
        for name, img in (("ir", ir), ("rgb", rgb)):
            big = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
            imwrite_any(crops_dir / f"{sid}_{name}.jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # Tarayıcı kaydının anahtarı örnek kümesinden: başka bir turun sayfası bu kaydı açmaz.
    # Özet geri çevrilemez; tahmin ya da sınıf sızmaz.
    token = hashlib.sha256(json.dumps(sorted(ids.values())).encode("utf-8")).hexdigest()[:16]
    page = PAGE.replace("__SAMPLES__", json.dumps([{"id": i} for i in ids]))
    page = page.replace("__TOKEN__", token)
    (out_dir / "index.html").write_text(page, encoding="utf-8", newline="\n")
    return ids


def score(ids: dict[str, str], teacher: dict[str, dict], answers: dict[str, str]) -> dict:
    missing = [i for i in ids if answers.get(i) not in ANSWERS]
    if missing:
        raise ValueError(f"eksik ya da geçersiz cevap: {', '.join(missing[:10])}")
    per_class: dict[str, dict] = {}
    correct = answered = unknown = 0
    for sid, key in ids.items():
        pred = CLASS_NAMES[teacher[key]["cls_id"]]
        row = per_class.setdefault(pred, {"k": 0, "n": 0})
        if answers[sid] == "unknown":
            unknown += 1
            continue
        answered += 1
        row["n"] += 1
        if answers[sid] == pred:
            row["k"] += 1
            correct += 1
    for row in per_class.values():
        row["rate"] = row["k"] / row["n"] if row["n"] else None
    acc = correct / answered if answered else None
    lo, hi = wilson(correct, answered)
    return {
        "accuracy": acc,
        "ci95": [lo, hi],
        "answered": answered,
        "unknown": unknown,
        "per_class": per_class,
        "decision": "teacher" if acc is not None and acc >= ACC_MIN else "all_car",
    }


PAGE = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Araç türü kör denetimi</title>
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
<h1>Bu araç hangi tür?</h1>
<p>Her örnekte aynı aracın IR ve RGB kırpıntısı gösterilir. Aracın türünü seç; ayırt edemiyorsan
"Bilmiyorum" seç. Kısayollar: 1 otomobil, 2 kamyon, 3 otobüs, 4 minibüs, 5 yük aracı, 6 bilmiyorum,
← → gezin.</p>
<div class="bar"><strong id="pos"></strong><span class="grow"></span><span id="done"></span></div>
<div class="views">
  <figure><img id="ir" alt="IR"><figcaption>IR</figcaption></figure>
  <figure><img id="rgb" alt="RGB"><figcaption>RGB</figcaption></figure>
</div>
<div class="bar">
  <button data-a="car">1 · Otomobil</button>
  <button data-a="truck">2 · Kamyon</button>
  <button data-a="bus">3 · Otobüs</button>
  <button data-a="van">4 · Minibüs</button>
  <button data-a="freight_car">5 · Yük aracı</button>
  <button data-a="unknown">6 · Bilmiyorum</button>
  <span class="grow"></span>
  <button id="prev">←</button><button id="next">→</button>
</div>
<div class="bar"><button id="save">Cevapları indir (JSON)</button></div>
</main>
<script>
var SAMPLES = __SAMPLES__;
var STORE = "baykus-teacher-__TOKEN__";
var answers = {};
try { answers = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) { answers = {}; }
var cur = 0;
function show() {
  var s = SAMPLES[cur];
  document.getElementById("pos").textContent = (cur + 1) + " / " + SAMPLES.length;
  var n = 0;
  for (var i = 0; i < SAMPLES.length; i++) { if (answers[SAMPLES[i].id]) n++; }
  document.getElementById("done").textContent = n + " cevaplandı";
  document.getElementById("ir").src = "crops/" + s.id + "_ir.jpg";
  document.getElementById("rgb").src = "crops/" + s.id + "_rgb.jpg";
  var btns = document.querySelectorAll("button[data-a]");
  for (var j = 0; j < btns.length; j++) {
    btns[j].className = answers[s.id] === btns[j].getAttribute("data-a") ? "on" : "";
  }
}
function answer(a) {
  answers[SAMPLES[cur].id] = a;
  try { localStorage.setItem(STORE, JSON.stringify(answers)); } catch (e) {}
  if (cur < SAMPLES.length - 1) { cur++; }
  show();
}
var btns = document.querySelectorAll("button[data-a]");
for (var b = 0; b < btns.length; b++) {
  btns[b].onclick = function () { answer(this.getAttribute("data-a")); };
}
document.getElementById("prev").onclick = function () { if (cur > 0) { cur--; show(); } };
document.getElementById("next").onclick = function () {
  if (cur < SAMPLES.length - 1) { cur++; show(); }
};
document.onkeydown = function (e) {
  if (e.key === "1") answer("car");
  if (e.key === "2") answer("truck");
  if (e.key === "3") answer("bus");
  if (e.key === "4") answer("van");
  if (e.key === "5") answer("freight_car");
  if (e.key === "6") answer("unknown");
  if (e.key === "ArrowLeft") document.getElementById("prev").onclick();
  if (e.key === "ArrowRight") document.getElementById("next").onclick();
};
document.getElementById("save").onclick = function () {
  var out = {};
  for (var i = 0; i < SAMPLES.length; i++) {
    out[SAMPLES[i].id] = answers[SAMPLES[i].id] || null;
  }
  var blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" });
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "teacher_answers.json";
  a.click();
};
show();
</script>
</body>
</html>
"""
