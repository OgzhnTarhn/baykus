"""Torch'suz testler için sahte `ultralytics`: val() GT kutularını tahmin olarak yazar.

Gerçek Ultralytics gibi: tahminler <project>/<name>/predictions.json, image_id = int(dosya adı),
category_id = sınıf + 1. Görüntü klasörü yoksa hata verir (yanlış data yolu yakalanır).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


class FakeYOLO:
    calls: list[dict] = []

    def __init__(self, weights: str):
        self.weights = weights

    def val(self, **kw):
        FakeYOLO.calls.append(kw)
        data = Path(kw["data"])
        cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
        img_dir = Path(cfg["path"]) / cfg[kw["split"]]
        if not any(img_dir.glob("*")):
            raise FileNotFoundError(f"görüntü yok: {img_dir}")
        gt = json.loads(
            (data.parent / "coco" / f"{kw['split']}_ir.json").read_text(encoding="utf-8")
        )
        stem = {im["id"]: Path(im["file_name"]).stem for im in gt["images"]}
        preds = [
            {
                "image_id": int(stem[a["image_id"]]),
                "category_id": a["category_id"],
                "bbox": a["bbox"],
                "score": 0.9,
            }
            for a in gt["annotations"]
        ]
        out = Path(kw["project"]) / kw["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "predictions.json").write_text(json.dumps(preds), encoding="utf-8")
        return SimpleNamespace(box=SimpleNamespace(map50=1.0))


def install(monkeypatch) -> type:
    FakeYOLO.calls = []
    mod = types.ModuleType("ultralytics")
    mod.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return FakeYOLO


def make_weights(run_dir: Path, run_id: str) -> Path:
    w = Path(run_dir) / "weights" / "best.pt"
    w.parent.mkdir(parents=True, exist_ok=True)
    w.write_bytes(b"")
    rec = {"run_id": run_id, "mlflow_run_id": None}
    (Path(run_dir) / "baykus_train.json").write_text(json.dumps(rec), encoding="utf-8")
    return w
