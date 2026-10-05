"""Sabit sınıf listesi ve DroneVehicle ham ad varyantları."""

from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = ("car", "truck", "bus", "van", "freight_car")

# Spike B'de görülen gerçek varyantlarla güncellenir.
CLASS_ALIASES: dict[str, str] = {
    "car": "car",
    "truck": "truck",
    "truvk": "truck",
    "bus": "bus",
    "van": "van",
    "freight car": "freight_car",
    "freight_car": "freight_car",
    "feright car": "freight_car",
    "feright_car": "freight_car",
    "feright": "freight_car",
}


def normalize_class(raw: str) -> str | None:
    return CLASS_ALIASES.get(" ".join(raw.strip().lower().split()))


def class_id(name: str) -> int:
    return CLASS_NAMES.index(name)


def coco_category_id(cls_id: int) -> int:
    return cls_id + 1
