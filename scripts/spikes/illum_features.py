"""Spike (ADR 0005, seçenek D): görüntü başına parlaklık özellikleri + etiketleme örnek sayfaları.

Ortalama parlaklık aydınlık geceyi gündüzden ayıramıyor (E0 mozaikleri). Bu script ek özellikler
(karanlık/aydınlık piksel oranları, yüzdelikler) hesaplar ve gündüz/gece sınırından örnekler basar.

Çıktılar:
  <out>/features.csv        split, stem, mean, p10, p25, p50, p75, p90, dark30, dark50, bright200
  <out>/samples.csv         idx, split, stem + özellikler (etiketlenecek örnekler)
  <out>/samples_NN.jpg      numaralı küçük resim sayfaları (8x5)

Çalıştırma (Colab, ham veri):
  python scripts/spikes/illum_features.py /content/raw/VisDrone-DroneVehicle /content/illum
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
CROP = 25  # IMREAD_REDUCED_COLOR_4 ile 100 px kenarlığın 1/4'ü
FIELDS = ["mean", "p10", "p25", "p50", "p75", "p90", "dark30", "dark50", "bright200"]
N_SAMPLES = 160
N_BINS = 8
COLS, ROWS = 8, 5
THUMB = (192, 154)


def load(path: Path) -> np.ndarray | None:
    buf = np.fromfile(str(path), np.uint8)
    if buf.size == 0:  # bozuk dosyaların bir kısmı 0 bayt (verify: unreadable_*)
        return None
    try:
        a = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_4)
    except cv2.error:
        return None
    return None if a is None else a[CROP:-CROP, CROP:-CROP]


def features(bgr: np.ndarray) -> dict:
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    p = np.percentile(v, [10, 25, 50, 75, 90])
    return {
        "mean": float(v.mean()),
        **{f"p{q}": float(x) for q, x in zip((10, 25, 50, 75, 90), p, strict=True)},
        "dark30": float((v < 30).mean()),
        "dark50": float((v < 50).mean()),
        "bright200": float((v > 200).mean()),
    }


def pick_samples(rows: list[dict], seed: int = 0) -> list[dict]:
    """Karanlık gece dışı (mean >= 50) görüntülerden örnek: dark50 sırasına göre eşit boyutlu
    dilimlerden eşit sayıda. Değer sınırı yerine sıra kullanılır; gündüzde dark50 çoğu zaman 0."""
    rng = np.random.default_rng(seed)
    pool = [r for r in rows if r["mean"] >= 50]
    if not pool:
        return []
    d = np.array([r["dark50"] for r in pool])
    order = np.lexsort((rng.random(len(pool)), d))  # eşitlikleri rastgele boz
    out = []
    for chunk in np.array_split(order, N_BINS):
        take = rng.choice(chunk, size=min(N_SAMPLES // N_BINS, len(chunk)), replace=False)
        out += [pool[i] for i in sorted(take)]
    return out


def sheet(raw: Path, items: list[tuple[int, dict]]) -> np.ndarray:
    tiles = []
    for i, r in items:
        a = load(raw / r["split"] / f"{r['split']}img" / f"{r['stem']}.jpg")
        t = np.zeros((THUMB[1], THUMB[0], 3), np.uint8) if a is None else cv2.resize(a, THUMB)
        cv2.rectangle(t, (0, 0), (52, 26), (0, 0, 0), -1)
        cv2.putText(t, str(i), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(t)
    tiles += [np.zeros((THUMB[1], THUMB[0], 3), np.uint8)] * (COLS * ROWS - len(tiles))
    rows = [np.hstack(tiles[k : k + COLS]) for k in range(0, COLS * ROWS, COLS)]
    return np.vstack(rows)


def main(raw: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in SPLITS:
        for p in sorted((raw / split / f"{split}img").glob("*.jpg")):
            a = load(p)
            if a is not None:
                rows.append({"split": split, "stem": p.stem, **features(a)})
    with (out / "features.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "stem", *FIELDS])
        w.writeheader()
        w.writerows(
            {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows
        )
    samples = pick_samples(rows)
    with (out / "samples.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "split", "stem", *FIELDS])
        w.writeheader()
        for i, r in enumerate(samples):
            w.writerow(
                {"idx": i, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}}
            )
    per = COLS * ROWS
    for page, start in enumerate(range(0, len(samples), per)):
        img = sheet(raw, list(enumerate(samples))[start : start + per])
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        enc.tofile(str(out / f"samples_{page:02d}.jpg"))
    print(f"görüntü: {len(rows)}, örnek: {len(samples)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
