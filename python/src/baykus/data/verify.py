"""RGB/IR çiftlerinin ve etiketlerin bütünlük kontrolü (FR-D1)."""

from __future__ import annotations

import collections
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from baykus.data.dronevehicle import parse_annotation
from baykus.data.layout import IMG_EXTS, RAW_SIZE, SplitDirs, find_image
from baykus.io import imread_any


@dataclass(frozen=True)
class Issue:
    split: str
    stem: str
    reason: str


def _stems(dirs: SplitDirs) -> list[str]:
    stems: set[str] = set()
    for folder in (dirs.rgb, dirs.ir):
        if folder.exists():
            stems |= {p.stem for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS}
    for folder in (dirs.label_rgb, dirs.label_ir):
        if folder.exists():
            stems |= {p.stem for p in folder.glob("*.xml")}
    return sorted(stems)


def _check(dirs: SplitDirs, stem: str) -> str | None:
    rgb_p, ir_p = find_image(dirs.rgb, stem), find_image(dirs.ir, stem)
    if rgb_p is None:
        return "missing_rgb"
    if ir_p is None:
        return "missing_ir"
    lr, li = dirs.label_rgb / f"{stem}.xml", dirs.label_ir / f"{stem}.xml"
    if not lr.exists():
        return "missing_label_rgb"
    if not li.exists():
        return "missing_label_ir"
    rgb, ir = imread_any(rgb_p), imread_any(ir_p)
    if rgb is None:
        return "unreadable_rgb"
    if ir is None:
        return "unreadable_ir"
    if rgb.shape[:2] != ir.shape[:2]:
        return "size_mismatch"
    if (rgb.shape[1], rgb.shape[0]) != RAW_SIZE:
        return "unexpected_size"
    for path, reason in ((lr, "bad_xml_rgb"), (li, "bad_xml_ir")):
        try:
            parse_annotation(path)
        except (ET.ParseError, ValueError):  # bozuk XML ya da eksik / sayısal olmayan koordinat
            return reason
    return None


def verify_split(dirs: SplitDirs, split: str) -> tuple[list[str], list[Issue]]:
    ok, issues = [], []
    for stem in _stems(dirs):
        reason = _check(dirs, stem)
        if reason is None:
            ok.append(stem)
        else:
            issues.append(Issue(split, stem, reason))
    return ok, issues


def write_outputs(meta_dir: Path, ok: dict[str, list[str]], issues: list[Issue]) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "excluded.txt").write_text(
        "".join(f"{i.split}/{i.stem}\t{i.reason}\n" for i in issues), encoding="utf-8"
    )
    (meta_dir / "ok_stems.json").write_text(json.dumps(ok, indent=1), encoding="utf-8")
    report = {
        "ok_counts": {k: len(v) for k, v in ok.items()},
        "issues_by_reason": dict(collections.Counter(i.reason for i in issues)),
        "issues": [asdict(i) for i in issues[:200]],
    }
    (meta_dir / "verify_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
