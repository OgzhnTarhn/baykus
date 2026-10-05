"""Spike A: Ultralytics YOLO26 + 4 kanallı TIFF uçtan uca çalışıyor mu? (CPU, sentetik veri)

Çalıştırma (python/ içinden):  uv run python ../scripts/spikes/ultralytics_4ch.py
Çıktı: data/spikes/u4ch/result_<layout>.json
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "spikes" / "u4ch"
SIZE = 128


LAYOUTS = ("hwc", "chw", "pages")


def write_tif(path: Path, img: np.ndarray, layout: str) -> None:
    """img HWC, kanallar R,G,B,T sırasıyla.

    hwc/chw: tek sayfa, RGB fotometrik; tifffile 4. örneği alfa olarak işaretler.
    pages: her kanal ayrı gri sayfa; Ultralytics sayfaları sırayla yığar.
    """
    if layout == "hwc":
        tifffile.imwrite(path, img, planarconfig="contig", compression="deflate")
    elif layout == "chw":
        tifffile.imwrite(
            path, np.transpose(img, (2, 0, 1)), planarconfig="separate", compression="deflate"
        )
    else:
        with tifffile.TiffWriter(path) as tw:
            for c in range(img.shape[2]):
                tw.write(img[..., c], photometric="minisblack", compression="deflate")


def make_dataset(root: Path, layout: str, n_train: int = 16, n_val: int = 4) -> Path:
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = rng.integers(0, 40, (SIZE, SIZE, 4), dtype=np.uint8)
            x1, y1 = (int(v) for v in rng.integers(10, SIZE - 50, 2))
            w, h = (int(v) for v in rng.integers(15, 40, 2))
            img[y1 : y1 + h, x1 : x1 + w, :3] = 200  # RGB'de parlak kutu
            img[y1 : y1 + h, x1 : x1 + w, 3] = 255  # termalde sıcak kutu
            write_tif(root / "images" / split / f"{i:04d}.tif", img, layout)
            cx, cy = (x1 + w / 2) / SIZE, (y1 + h / 2) / SIZE
            (root / "labels" / split / f"{i:04d}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {w / SIZE:.6f} {h / SIZE:.6f}\n", encoding="utf-8"
            )
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\n"
        "channels: 4\nnames:\n  0: car\n",
        encoding="utf-8",
    )
    return yaml_path


def probe_channel_order(root: Path, layout: str) -> dict:
    """Modelin gerçekte gördüğü kanal sırası (okuma + LetterBox + Format, augment yok)."""
    from ultralytics.data.dataset import YOLODataset

    d = root / "probe"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    img = np.zeros((SIZE, SIZE, 4), np.uint8)
    for c, v in enumerate((10, 20, 30, 40)):  # R=10, G=20, B=30, T=40
        img[..., c] = v
    write_tif(d / "images" / "probe.tif", img, layout)
    (d / "labels" / "probe.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    ds = YOLODataset(
        img_path=str(d / "images"),
        data={"names": {0: "car"}, "channels": 4},
        task="detect",
        imgsz=SIZE,
        augment=False,
    )
    t = ds[0]["img"]  # CHW, modele giden sıra
    seen = [int(t[c, SIZE // 2, SIZE // 2]) for c in range(t.shape[0])]
    # [10, 20, 30, 40] → RGBT yazmak doğru; [30, 20, 10, 40] → model BGRT görüyor
    return {"model_sees_values": seen}


def probe_first_conv_init() -> dict:
    """Eğitimsiz: COCO ağırlıkları 4 kanallı ilk katmana nasıl aktarılıyor?"""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    src = YOLO("yolo26n.pt").model
    w3 = src.model[0].conv.weight.detach().float().clone()  # (out, 3, k, k)
    m4 = DetectionModel("yolo26n.yaml", ch=4, nc=1, verbose=False)
    m4.load(src, verbose=True)
    w4 = m4.model[0].conv.weight.detach().float()  # (out, 4, k, k)
    return {
        "w3_shape": list(w3.shape),
        "w4_shape": list(w4.shape),
        "ch0_2_equal_coco": bool(torch_allclose(w4[:, :3], w3)),
        "ch3_equals_rgb_mean": bool(torch_allclose(w4[:, 3], w3.mean(1))),
        "ch3_std": float(w4[:, 3].std()),
        "coco_std": float(w3.std()),
    }


def torch_allclose(a, b) -> bool:
    import torch

    return torch.allclose(a, b, atol=1e-6)


def set_rgb_mean(trainer) -> None:
    """Callback: ilk katmanın 4. kanalını RGB ağırlıklarının ortalamasıyla başlat (model + EMA)."""
    import torch

    with torch.no_grad():
        for m in (trainer.model, trainer.ema.ema if trainer.ema else None):
            if m is None:
                continue
            w = m.model[0].conv.weight
            w[:, 3] = w[:, :3].mean(1)
    trainer.rgb_mean_applied = True


def run(layout: str) -> dict:
    import cv2
    import ultralytics
    from ultralytics import YOLO

    res: dict = {"layout": layout, "ultralytics": ultralytics.__version__}
    root = OUT / layout
    yaml_path = make_dataset(root, layout)
    sample = root / "images" / "train" / "0000.tif"
    key = slice(None) if layout == "pages" else None  # çok sayfalıda bütün sayfaları oku
    original = tifffile.imread(sample, key=key)
    res["tifffile_shape"] = list(original.shape)

    try:
        from ultralytics.utils.patches import imread

        arr = imread(str(sample), cv2.IMREAD_UNCHANGED)
        res["ultralytics_imread_shape"] = None if arr is None else list(arr.shape)
        if arr is not None and arr.ndim == 3 and arr.shape[2] == 4:
            ref = original if layout == "hwc" else np.transpose(original, (1, 2, 0))
            res["channel_order_matches_RGBT"] = bool((arr == ref).all())
            res["channel_order_matches_BGRT"] = bool((arr[..., [2, 1, 0, 3]] == ref).all())
    except Exception:
        res["imread_error"] = traceback.format_exc(limit=3)

    for key, fn in (
        ("channel_probe", lambda: probe_channel_order(root, layout)),
        ("first_conv_init", probe_first_conv_init),
    ):
        try:
            res[key] = fn()
        except Exception:
            res[f"{key}_error"] = traceback.format_exc(limit=5)

    stages = {}
    try:
        model = YOLO("yolo26n.pt")
        model.train(
            data=str(yaml_path),
            epochs=1,
            imgsz=SIZE,
            batch=4,
            device="cpu",
            workers=0,
            plots=False,
            project=str(OUT / "runs"),
            name=layout,
            exist_ok=True,
            verbose=False,
        )
        stages["train"] = "ok"
        res["first_conv_weight_shape"] = list(model.model.model[0].conv.weight.shape)
        metrics = model.val(data=str(yaml_path), imgsz=SIZE, batch=4, device="cpu", plots=False)
        stages["val"] = "ok"
        res["val_map50"] = float(metrics.box.map50)
        import onnx
        import onnxruntime as ort

        def dims(v):
            return [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]

        # nms=None: Ultralytics varsayılanı (ham çıktı, harici NMS); nms=False: NMS'siz uçtan uca
        for nms in (None, False):
            key = "e2e" if nms is False else "raw"
            onnx_path = model.export(format="onnx", imgsz=SIZE, nms=nms)
            stages[f"export_{key}"] = "ok"
            g = onnx.load(onnx_path).graph
            res[f"onnx_{key}"] = {
                "inputs": [{v.name: dims(v)} for v in g.input],
                "outputs": [{v.name: dims(v)} for v in g.output],
                "topk_nodes": sum(n.op_type == "TopK" for n in g.node),
            }
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            outs = sess.run(None, {name: np.zeros((1, 4, SIZE, SIZE), np.float32)})
            res[f"onnx_{key}"]["onnxruntime_output_shapes"] = [list(o.shape) for o in outs]
        stages["onnxruntime"] = "ok"
    except Exception:
        res["pipeline_error"] = traceback.format_exc(limit=5)

    # rgb_mean başlatması callback ile (Ultralytics'e yama yok)
    try:
        m = YOLO("yolo26n.pt")
        m.add_callback("on_pretrain_routine_end", set_rgb_mean)
        m.train(
            data=str(yaml_path),
            epochs=1,
            imgsz=SIZE,
            batch=4,
            device="cpu",
            workers=0,
            plots=False,
            project=str(OUT / "runs"),
            name=f"{layout}_rgbmean",
            exist_ok=True,
            verbose=False,
        )
        res["rgb_mean_callback_applied"] = bool(getattr(m.trainer, "rgb_mean_applied", False))
        stages["train_rgb_mean"] = "ok"
    except Exception:
        res["rgb_mean_error"] = traceback.format_exc(limit=5)
    res["stages"] = stages
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for layout in LAYOUTS:
        res = run(layout)
        (OUT / f"result_{layout}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
