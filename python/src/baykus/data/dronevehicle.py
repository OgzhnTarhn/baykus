"""DroneVehicle VOC-benzeri XML etiket ayrıştırıcı (polygon ve bndbox)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from baykus.classes import normalize_class


@dataclass(frozen=True)
class RawObject:
    cls_name: str
    points: np.ndarray  # (N, 2), orijinal görüntü koordinatları


@dataclass
class ParseResult:
    objects: list[RawObject] = field(default_factory=list)
    unknown_classes: list[str] = field(default_factory=list)
    skipped_shapes: int = 0


def _points(obj: ET.Element) -> np.ndarray | None:
    poly = obj.find("polygon")
    if poly is not None:
        xs = [float(poly.findtext(f"x{i}")) for i in range(1, 5)]
        ys = [float(poly.findtext(f"y{i}")) for i in range(1, 5)]
        return np.stack([xs, ys], axis=1)
    bb = obj.find("bndbox")
    if bb is not None:
        x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
        x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return None


def parse_annotation(path: Path) -> ParseResult:
    root = ET.parse(path).getroot()
    res = ParseResult()
    for obj in root.iter("object"):
        raw = obj.findtext("name") or ""
        name = normalize_class(raw)
        if name is None:
            res.unknown_classes.append(raw.strip())
            continue
        pts = _points(obj)
        if pts is None:
            res.skipped_shapes += 1
            continue
        res.objects.append(RawObject(name, pts))
    return res
