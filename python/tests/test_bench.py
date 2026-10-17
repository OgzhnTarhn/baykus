"""baykus-bench (H5 Görev 10, spec §14): istatistik, zamanlama, pipeline'lar, kayıt, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from baykus.bench.core import (
    LateFusionPipeline,
    SinglePipeline,
    bench_record,
    summarize,
    time_pipeline,
)
from baykus.bench.sysinfo import peak_rss_mb, sysinfo
from tests.synthetic import SIZE, make_synthetic_dataset

REPO = Path(__file__).resolve().parents[2]
SEL = {"selected": {"iou_thr": 0.7, "w_rgb": 1.0, "ir_weight": 3.0}, "skip_box_thr": 0.001}


class BoxBackend:
    """Sabit kutular döndürür (letterbox'lı uzayda); kaç kez çağrıldığını sayar."""

    def __init__(self, rows: list[list[float]], imgsz: int = SIZE, dtype=np.float32):
        self.rows = np.array([rows], np.float32).reshape(1, -1, 6)
        self.imgsz, self.channels, self.input_dtype = imgsz, 3, np.dtype(dtype)
        self.provider, self.threads, self.calls = "cpu", None, 0

    def infer(self, x: np.ndarray) -> np.ndarray:
        assert x.dtype == self.input_dtype
        self.calls += 1
        return self.rows


def test_summarize_exact():
    s = summarize(list(range(1, 101)))
    assert s == {"p50": 50.5, "p95": pytest.approx(95.05), "mean": 50.5, "n": 100}


def test_time_pipeline_drops_warmup_and_sums_total():
    t = time_pipeline(lambda i: {"pre": 1.0, "infer": float(i)}, warmup=3, runs=5)
    assert t["infer"] == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert t["total"] == [4.0, 5.0, 6.0, 7.0, 8.0] and t["pre"] == [1.0] * 5


def test_single_step_stages_and_cycles_frames():
    b = BoxBackend([[10, 10, 30, 30, 0.9, 0]], dtype=np.float16)
    frames = [np.zeros((SIZE, SIZE, 3), np.uint8)] * 2
    p = SinglePipeline(b, frames, conf=0.25)
    for i in range(5):
        t = p.step(i)
    assert set(t) == {"pre", "infer", "post"} and all(v >= 0 for v in t.values())
    assert b.calls == 5 and len(p.last) == 1


def test_late_fusion_step_stages_and_conf_filter():
    rgb = BoxBackend([[10, 10, 30, 30, 0.9, 0]])
    ir = BoxBackend([[10, 10, 30, 30, 0.9, 0], [40, 40, 60, 60, 0.2, 1]])
    frame = (np.zeros((SIZE, SIZE, 3), np.uint8), np.zeros((SIZE, SIZE, 3), np.uint8))
    p = LateFusionPipeline(rgb, ir, [frame], SEL, conf=0.25)
    t = p.step(0)
    assert set(t) == {"pre", "infer_rgb", "infer_ir", "post", "fuse"}
    assert len(p.last) == 1 and p.last[0]["score"] >= 0.25  # IR'nin 0.2'lik kutusu süzüldü


def test_late_fusion_rejects_per_illumination_weights():
    sel = {**SEL, "selected": {**SEL["selected"], "ir_weight": {"day": 1.0}}}
    with pytest.raises(ValueError):
        LateFusionPipeline(BoxBackend([]), BoxBackend([]), [], sel, conf=0.25)


def test_bench_record_schema_and_fps():
    times = {"pre": [1.0] * 4, "infer": [9.0] * 4, "total": [10.0] * 4}
    rec = bench_record(pipeline="p", variant="onnx_int8", hardware="laptop-cpu",
                       models=[{"name": "m", "file": "f", "size_mb": 1.0, "sha256": "x"}],
                       times=times, protocol={"warmup": 1, "runs": 4}, threads=None)  # fmt: skip
    assert rec["fps"] == pytest.approx(100.0) and rec["protocol"]["batch"] == 1
    assert rec["stages"]["total"]["p95"] == pytest.approx(10.0)
    assert "power_plugged" in rec["system"]  # None da olabilir
    assert rec["peak_rss_mb"] is None or rec["peak_rss_mb"] > 0


def test_peak_rss_positive():
    v = peak_rss_mb()
    assert v is None or v > 0


def test_sysinfo_survives_missing_libs(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("yok")

    monkeypatch.setattr("cpuinfo.get_cpu_info", boom)
    monkeypatch.setattr("psutil.sensors_battery", boom)
    s = sysinfo()
    assert s["cpu_brand"] is None and s["cpu_flags"] is None and s["power_plugged"] is None
    assert "onnxruntime" in s["software"]


def test_sysinfo_warns_on_battery(monkeypatch, capsys):
    from types import SimpleNamespace

    monkeypatch.setattr("psutil.sensors_battery", lambda: SimpleNamespace(power_plugged=False))
    assert sysinfo()["power_plugged"] is False
    assert "pil" in capsys.readouterr().err


def _inputs(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """bench-inputs ile girdi klasörü; E1/E2 tahminleri onnx_int8 klasörlerinde."""
    from baykus.edge.cli import main
    from baykus.eval.predictions import save_predictions

    root = make_synthetic_dataset(tmp_path / "dv", with_ir=True)
    cfg = REPO / "configs/edge/e9_edge.yaml"
    gt = json.loads((root / "coco/val_ir.json").read_text(encoding="utf-8"))
    preds = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"],
              "score": 0.9} for a in gt["annotations"]]  # fmt: skip
    for run in ("2026-10-11_e1-rgb_s_49da0cb", "2026-10-14_e2-ir_s_02c4098"):
        save_predictions(tmp_path / "proj" / f"{run}_val_onnx_int8" / "predictions_coco.json.gz",
                         preds)  # fmt: skip
    out = tmp_path / "inputs"
    args = ["bench-inputs", "--config", str(cfg), "--data-root", str(root), "--split", "val",
            "--preds-dir", str(tmp_path / "proj"), "--out", str(out),
            "--modalities", "rgb", "ir"]  # fmt: skip
    assert main(args) == 0
    return root, out


def test_bench_inputs_layout(tmp_path: Path, monkeypatch):
    _, out = _inputs(tmp_path, monkeypatch)
    images = json.loads((out / "images.json").read_text(encoding="utf-8"))
    assert len(images) == 4  # 20 istendi, sentetik val'de 4 görüntü var
    s = images[0]["stem"]
    assert (out / "rgb/images/val" / f"{s}.jpg").exists()
    assert (out / "ir/images/val" / f"{s}.png").exists()
    assert not (out / "npy").exists()  # .npy yalnız --npy ile (NPU denemesi; 20 görüntüde ≈ 340 MB)
    refs = json.loads((out / "refs/2026-10-11_e1-rgb_s_49da0cb_onnx_int8.json").read_text(
        encoding="utf-8"))  # fmt: skip
    assert refs and {d["image_id"] for d in refs} <= {im["image_id"] for im in images}


def test_parity_and_bench_cli_with_fake_backend(tmp_path: Path, monkeypatch):
    import baykus.bench.cli as bc
    import baykus.edge.cli as ec
    from baykus.edge.variants import model_file

    root, out = _inputs(tmp_path, monkeypatch)
    models = tmp_path / "models"
    for run in ("2026-10-11_e1-rgb_s_49da0cb", "2026-10-14_e2-ir_s_02c4098"):
        f = model_file(models, run, "onnx_int8")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(run.encode())
    # sentetik kutu 16..48 × 20..40 (64 px görüntü, 640'a 10 kat büyür, dolgu 0)
    box = [[160, 200, 480, 400, 0.9, 0]]
    monkeypatch.setattr(ec, "open_backend", lambda p, threads=None: BoxBackend(box, imgsz=640))
    monkeypatch.setattr(bc, "open_backend", lambda p, threads=None: BoxBackend(box, imgsz=640))
    cfg = str(REPO / "configs/edge/e9_edge.yaml")
    res = tmp_path / "res"
    par = ["parity", "--config", cfg, "--inputs", str(out), "--models-dir", str(models),
           "--variant", "onnx_int8", "--hardware", "laptop-cpu", "--results", str(res)]  # fmt: skip
    assert ec.main(par) == 0
    rep = json.loads((res / "bench/parity_laptop-cpu_onnx_int8.json").read_text(encoding="utf-8"))
    assert rep["models"]["e1_rgb"]["ok"] is True and "e3c_modality_dropout" not in rep["models"]
    run = ["run", "--config", cfg, "--pipeline", "e4_late_fusion", "--variant", "onnx_int8",
           "--models-dir", str(models), "--inputs", str(out), "--hardware", "laptop-cpu",
           "--results", str(res), "--warmup", "2", "--runs", "5"]  # fmt: skip
    assert bc.main(run) == 0
    rec = json.loads((res / "bench/e4_late_fusion_onnx_int8_laptop-cpu.json").read_text(
        encoding="utf-8"))  # fmt: skip
    assert rec["stages"]["total"]["n"] == 5 and set(rec["stages"]) >= {"fuse", "infer_ir"}
    assert [m["name"] for m in rec["models"]] == ["e1_rgb", "e2_ir"]
    assert bc.main(run) == 0  # aynı modellerle tekrar: atlanır


def test_sysinfo_flag_names_windows_and_linux(monkeypatch):
    # py-cpuinfo Windows'ta CPUID adlarını (avx512vnni), Linux'ta /proc/cpuinfo adlarını
    # (avx512_vnni) verir; ikisi de yakalanmalı
    info = {"brand_raw": "x", "flags": ["avx2", "avx512f", "avx512vnni", "avx_vnni", "sse4_2"]}
    monkeypatch.setattr("cpuinfo.get_cpu_info", lambda: info)
    assert sysinfo()["cpu_flags"] == ["avx2", "avx512f", "avx512vnni", "avx_vnni"]


def test_bench_cli_prints_on_cp1254_console(tmp_path: Path, monkeypatch):
    """Windows konsolu (cp1254) ASCII dışı her karakteri basamaz; ölçüm kaydı yazıldıktan sonra
    yazdırma hatası dönüş kodunu 2 yapmamalı."""
    import io
    import sys

    import baykus.bench.cli as bc
    from baykus.edge.variants import model_file

    _, out = _inputs(tmp_path, monkeypatch)
    models = tmp_path / "models"
    f = model_file(models, "2026-10-15_e3c-modality-dropout_s_d4e1280", "onnx_fp32")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"m")
    monkeypatch.setattr(bc, "open_backend", lambda p, threads=None: BoxBackend([], imgsz=640))
    monkeypatch.setattr("baykus.bench.cli.load_input",
                        lambda *a, **k: np.zeros((SIZE, SIZE, 4), np.uint8))  # fmt: skip
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1254")
    monkeypatch.setattr(sys, "stdout", console)
    run = ["run", "--config", str(REPO / "configs/edge/e9_edge.yaml"), "--pipeline",
           "e3c_modality_dropout", "--variant", "onnx_fp32", "--models-dir", str(models),
           "--inputs", str(out), "--hardware", "laptop-cpu", "--results", str(tmp_path / "r"),
           "--warmup", "1", "--runs", "2"]  # fmt: skip
    assert bc.main(run) == 0


def test_bench_rerun_with_other_protocol_is_error(tmp_path: Path, monkeypatch, capsys):
    """Deneme ölçümü (az tur) asıl ölçümü 'zaten var' diye gölgelememeli."""
    import baykus.bench.cli as bc
    from baykus.edge.variants import model_file

    _, out = _inputs(tmp_path, monkeypatch)
    models = tmp_path / "models"
    f = model_file(models, "2026-10-15_e3c-modality-dropout_s_d4e1280", "onnx_fp32")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"m")
    monkeypatch.setattr(bc, "open_backend", lambda p, threads=None: BoxBackend([], imgsz=640))
    monkeypatch.setattr("baykus.bench.cli.load_input",
                        lambda *a, **k: np.zeros((SIZE, SIZE, 4), np.uint8))  # fmt: skip
    base = ["run", "--config", str(REPO / "configs/edge/e9_edge.yaml"), "--pipeline",
            "e3c_modality_dropout", "--variant", "onnx_fp32", "--models-dir", str(models),
            "--inputs", str(out), "--hardware", "laptop-cpu",
            "--results", str(tmp_path / "r")]  # fmt: skip
    assert bc.main([*base, "--warmup", "1", "--runs", "2"]) == 0
    assert bc.main([*base, "--warmup", "1", "--runs", "2"]) == 0  # aynı protokol: atlanır
    assert bc.main([*base, "--warmup", "1", "--runs", "3"]) == 2
    assert bc.main([*base, "--warmup", "1", "--runs", "2", "--threads", "4"]) == 2
    assert "protokol" in capsys.readouterr().err


def test_bench_inputs_npy_only_on_request(tmp_path: Path, monkeypatch):
    from baykus.edge.cli import main

    root = make_synthetic_dataset(tmp_path / "dv", with_ir=True)
    out = tmp_path / "inputs"
    args = ["bench-inputs", "--config", str(REPO / "configs/edge/e9_edge.yaml"), "--data-root",
            str(root), "--split", "val", "--preds-dir", str(tmp_path / "yok"), "--out", str(out),
            "--modalities", "rgb", "--npy"]  # fmt: skip
    assert main(args) == 0
    x = np.load(next((out / "npy/rgb").glob("*.npy")))
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
