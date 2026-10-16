"""Uçtan uca edge testleri (slow): Ultralytics LetterBox ile birebir uyum."""

import numpy as np
import pytest

from baykus.edge.frames import preprocess

pytestmark = pytest.mark.slow


def test_letterbox_matches_ultralytics():
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import LetterBox

    for shape in [(512, 640, 3), (200, 300, 4)]:
        img = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
        ref = LetterBox(new_shape=(640, 640), auto=False)(image=img)
        x, _ = preprocess(img)
        mine = x[0].transpose(1, 2, 0) * 255
        if shape[2] == 3:
            mine = mine[..., ::-1]
        assert np.array_equal(mine.round().astype(np.uint8), ref)


def test_export_onnx_variants_with_real_ultralytics(tmp_path):
    """yolo26n (COCO) ile sentetik veride dört ONNX varyantı: uçtan uca çıktı (1, 300, 6)."""
    pytest.importorskip("ultralytics")
    import shutil
    from pathlib import Path

    import onnxruntime as ort

    from baykus.edge.calib import calibration_stems
    from baykus.edge.export import export_model
    from baykus.edge.variants import ONNX_VARIANTS
    from tests.synthetic import make_synthetic_dataset

    src = Path(__file__).resolve().parents[1] / "yolo26n.pt"
    if not src.exists():
        pytest.skip("yolo26n.pt yok (python/ altına indirilir)")
    w = tmp_path / "w" / "best.pt"
    w.parent.mkdir()
    shutil.copy(src, w)
    root = make_synthetic_dataset(tmp_path / "d", n_train=4, with_ir=True)
    out = export_model(w, tmp_path / "m", "R", "rgb", list(ONNX_VARIANTS), imgsz=640,
                       root=root, stems=calibration_stems(root, n=4))  # fmt: skip
    assert set(out) == set(ONNX_VARIANTS)
    assert out["onnx_fp16"]["input"]["dtype"] == "float32"  # keep_io_types
    x = np.zeros((1, 3, 640, 640), np.float32)
    for v in ONNX_VARIANTS:
        s = ort.InferenceSession(out[v]["file"], providers=["CPUExecutionProvider"])
        assert s.run(None, {"images": x})[0].shape == (1, 300, 6)
    assert out["onnx_int8_headfp32"]["quant"]["excluded_nodes"] > out["onnx_int8"]["quant"][
        "excluded_nodes"]  # fmt: skip


def test_evaluate_variant_with_real_onnx(tmp_path):
    """Gerçek ONNX FP32 (yolo26n) + ORT CPU ile sentetik val'de kayıt yazılıyor."""
    pytest.importorskip("ultralytics")
    import json
    import shutil
    from pathlib import Path

    from baykus.edge.evaluate import evaluate_variant
    from baykus.edge.export import export_model
    from baykus.train.config import load_config
    from tests.synthetic import make_synthetic_dataset

    src = Path(__file__).resolve().parents[1] / "yolo26n.pt"
    if not src.exists():
        pytest.skip("yolo26n.pt yok")
    w = tmp_path / "w" / "best.pt"
    w.parent.mkdir()
    shutil.copy(src, w)
    out = export_model(w, tmp_path / "m", "R", "rgb", ["onnx_fp32"], imgsz=640)
    root = make_synthetic_dataset(tmp_path / "d")
    cfg = Path(__file__).resolve().parents[2] / "configs/e1_rgb.yaml"
    paths = evaluate_variant(model_cfg=load_config(cfg), run_id="R", config="configs/e1_rgb.yaml",
                             variant="onnx_fp32", model_path=Path(out["onnx_fp32"]["file"]),
                             data_root=root, meta=root / "meta", split="val",
                             project=tmp_path / "p", results=tmp_path / "r", threads=2)  # fmt: skip
    rec = json.loads(paths[0].read_text(encoding="utf-8"))
    assert isinstance(rec["metrics"]["map50"], float) and rec["edge"]["threads"] == 2
    assert rec["format"] == "onnx" and rec["edge"]["seconds_per_image"] > 0


def test_bench_single_pipeline_with_real_onnx(tmp_path):
    """Gerçek ONNX FP32 (yolo26n) + ORT CPU: bench kaydı aşama süreleriyle yazılıyor."""
    pytest.importorskip("ultralytics")
    import shutil
    from pathlib import Path

    from baykus.bench.core import SinglePipeline, bench_record, time_pipeline
    from baykus.edge.backends import OrtBackend
    from baykus.edge.export import export_model

    src = Path(__file__).resolve().parents[1] / "yolo26n.pt"
    if not src.exists():
        pytest.skip("yolo26n.pt yok")
    w = tmp_path / "w" / "best.pt"
    w.parent.mkdir()
    shutil.copy(src, w)
    f = export_model(w, tmp_path / "m", "R", "rgb", ["onnx_fp32"], imgsz=640)["onnx_fp32"]["file"]
    b = OrtBackend(Path(f), threads=2)
    p = SinglePipeline(b, [np.zeros((512, 640, 3), np.uint8)], conf=0.25)
    times = time_pipeline(p.step, warmup=2, runs=5)
    rec = bench_record(pipeline="p", variant="onnx_fp32", hardware="laptop-cpu", models=[],
                       times=times, protocol={"warmup": 2, "runs": 5}, threads=2)  # fmt: skip
    assert rec["stages"]["infer"]["n"] == 5 and rec["stages"]["infer"]["p50"] > 0
    assert rec["fps"] > 0 and rec["system"]["logical_cores"]
