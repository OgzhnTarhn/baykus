"""Spike B: DroneVehicle Kaggle kopyasının gerçek yapısını çıkarır.

Çalıştırma (python/ içinden):
  uv run python ../scripts/spikes/inspect_dronevehicle.py \
    ../data/raw/dronevehicle ../data/spikes/dv_inspect.json
"""

from __future__ import annotations

import collections
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main(root: Path, out: Path, n_sample: int = 400) -> None:
    rep: dict = {}
    files = [p for p in root.rglob("*") if p.is_file()]
    tree = collections.Counter((str(p.parent.relative_to(root)), p.suffix.lower()) for p in files)
    rep["files"] = [{"dir": d, "ext": e, "count": c} for (d, e), c in sorted(tree.items())]

    xmls = sorted(p for p in files if p.suffix.lower() == ".xml")
    rep["n_xml"] = len(xmls)
    rep["xml_samples"] = [
        {"path": str(p.relative_to(root)), "head": p.read_text("utf-8", errors="replace")[:1500]}
        for p in xmls[:2]
    ]
    names, tags, bad = collections.Counter(), collections.Counter(), []
    max_objs = 0
    for p in xmls:
        try:
            r = ET.parse(p).getroot()
        except ET.ParseError:
            bad.append(str(p.relative_to(root)))
            continue
        objs = list(r.iter("object"))
        max_objs = max(max_objs, len(objs))
        for o in objs:
            names[(o.findtext("name") or "").strip()] += 1
            for c in o:
                if c.tag not in {"name", "pose", "truncated", "difficult"}:
                    tags[c.tag] += 1
    rep.update(
        class_names=dict(names),
        shape_tags=dict(tags),
        n_bad_xml=len(bad),
        bad_xml=bad[:50],
        max_objects_per_file=max_objs,
    )

    imgs = [p for p in files if p.suffix.lower() in IMG_EXT]
    random.seed(0)
    stats = collections.Counter()
    for p in random.sample(imgs, min(n_sample, len(imgs))):
        a = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
        d = str(p.parent.relative_to(root))
        if a is None:
            stats[(d, "UNREADABLE")] += 1
            continue
        ch = 1 if a.ndim == 2 else a.shape[2]
        gray3 = ch == 3 and bool((a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all())
        stats[(d, f"{a.shape[1]}x{a.shape[0]}x{ch}{' gray3' if gray3 else ''}")] += 1
    rep["image_sample_stats"] = [
        {"dir": d, "kind": k, "count": c} for (d, k), c in sorted(stats.items())
    ]
    rep["other_files"] = [
        str(p.relative_to(root))
        for p in files
        if p.suffix.lower() in {".txt", ".csv", ".json", ".md", ".xlsx"}
    ][:100]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: v for k, v in rep.items() if k != "xml_samples"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
