"""Edge tablosu ve sevk kuralı (H5 Görev 11, ADR 0007)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baykus.edge.config import load_edge_config
from baykus.edge.variants import VARIANTS
from baykus.report.edge import apply_rule, edge_rows, edge_table, load_edge

REPO = Path(__file__).resolve().parents[2]
CFG = load_edge_config(REPO / "configs/edge/e9_edge.yaml")
E1 = CFG.models["e1_rgb"].run_id
E2 = CFG.models["e2_ir"].run_id
E3C = CFG.models["e3c_modality_dropout"].run_id
E4 = "2026-10-16_e4-late-fusion_s_b346199"
ONNX = ("onnx_fp32", "onnx_fp16", "onnx_int8", "onnx_int8_headfp32")


def _single(run_id: str, variant: str | None, map50: float, size: float = 10.0) -> dict:
    v = VARIANTS.get(variant) if variant else None
    rec = {"run_id": run_id, "split": "val", "label_set": "ir",
           "format": v.fmt if v else "pytorch", "precision": v.precision if v else "fp32",
           "metrics": {"map50": map50}}  # fmt: skip
    if variant:
        rec["edge"] = {"variant": variant, "file_size_mb": size}
    return rec


def _fused(variant: str | None, map50: float, selection: str = "e4_late_fusion") -> dict:
    v = VARIANTS.get(variant) if variant else None
    return {"run_id": E4, "split": "val", "label_set": "ir",
            "format": v.fmt if v else "pytorch", "precision": v.precision if v else "fp32",
            "metrics": {"map50": map50},
            "fusion": {"selection": selection, "rgb_run_id": E1, "ir_run_id": E2,
                       **({"variant": variant} if variant else {})}}  # fmt: skip


def _bench(pipeline: str, variant: str, p95: float, hw: str = "laptop-cpu") -> dict:
    return {"pipeline": pipeline, "variant": variant, "hardware": hw, "fps": 1000 / (p95 - 5),
            "stages": {"total": {"p50": p95 - 5, "p95": p95, "mean": p95 - 5}}}  # fmt: skip


def data(e4_map=None, e4_p95=None, e3c_p95=None):
    """E4: fp32 0.848 · fp16 0.848 · int8 0.840 · int8_head 0.845; E3c: 0.851 · 0.851 · 0.845 ·
    0.848. Laptop p95 varsayılanı E4 120/250/70/80, E3c 60/140/30/35."""
    e4_map = {"onnx_fp32": 0.848, "onnx_fp16": 0.848, "onnx_int8": 0.840,
              "onnx_int8_headfp32": 0.845, **(e4_map or {})}  # fmt: skip
    e3c_map = dict(zip(ONNX, (0.851, 0.851, 0.845, 0.848), strict=True))
    e4_p95 = {**dict(zip(ONNX, (120, 250, 70, 80), strict=True)), **(e4_p95 or {})}
    e3c_p95 = {**dict(zip(ONNX, (60, 140, 30, 35), strict=True)), **(e3c_p95 or {})}
    acc = [_fused(v, m) for v, m in e4_map.items()]
    acc += [_single(E3C, v, m) for v, m in e3c_map.items()]
    acc += [_single(r, v, 0.8, size=s) for v in ONNX for r, s in ((E1, 9.0), (E2, 11.0))]
    bench = [_bench("e4_late_fusion", v, p) for v, p in e4_p95.items() if p is not None]
    bench += [_bench("e3c_modality_dropout", v, p) for v, p in e3c_p95.items() if p is not None]
    bench.append(_bench("e4_late_fusion", "trt_fp16", 12.0, hw="colab-t4"))
    runs = [_fused(None, 0.8478), _fused(None, 0.8504, "e4b_late_fusion_illum"),
            _single(E3C, None, 0.8506)]  # fmt: skip
    return acc, bench, runs


def rows(**kw):
    return edge_rows(*data(**kw), CFG)


def _row(rs, pipeline, variant, hw="laptop-cpu"):
    return next(r for r in rs if (r["pipeline"], r["variant"], r["hardware"]) ==
                (pipeline, variant, hw))  # fmt: skip


def test_rows_delta_size_and_pytorch_reference():
    rs = rows()
    r = _row(rs, "e4_late_fusion", "onnx_int8")
    assert r["delta_map50"] == pytest.approx(-0.008) and r["p95"] == 70
    assert r["size_mb"] == pytest.approx(20.0)  # iki modelin toplamı
    ref = _row(rs, "e4_late_fusion", "pytorch", None)
    assert ref["map50"] == pytest.approx(0.8478)  # E4b değil
    assert _row(rs, "e3c_modality_dropout", "onnx_int8")["size_mb"] == pytest.approx(10.0)
    assert _row(rs, "e4_late_fusion", "trt_fp16", "colab-t4")["map50"] is None


def test_rule_ships_fastest_eligible_of_first_passing_candidate():
    r = apply_rule(rows(), CFG)
    assert r["shipped"]["pipeline"] == "e4_late_fusion"
    assert r["shipped"]["variant"] == "onnx_int8" and r["shipped"]["p95"] == 70


def test_rule_excludes_variant_beyond_tolerance():
    r = apply_rule(rows(e4_map={"onnx_int8": 0.835}), CFG)  # Δ −1,3 puan
    assert r["shipped"]["variant"] == "onnx_int8_headfp32"


def test_tolerance_boundary_passes():
    r = apply_rule(rows(e4_map={"onnx_int8": 0.838}), CFG)  # Δ tam −1,0 puan
    assert r["shipped"]["variant"] == "onnx_int8"


def test_rule_falls_back_to_second_candidate_then_none():
    slow = dict.fromkeys(ONNX, 150)
    r = apply_rule(rows(e4_p95=slow), CFG)
    assert r["shipped"]["pipeline"] == "e3c_modality_dropout"
    assert r["shipped"]["variant"] == "onnx_int8"
    assert [t["passes"] for t in r["trace"]] == [False, True]
    r = apply_rule(rows(e4_p95=slow, e3c_p95=slow), CFG)
    assert r["shipped"] is None and [t["passes"] for t in r["trace"]] == [False, False]


def test_missing_bench_not_eligible_and_dash_in_table():
    rs = rows(e4_p95={"onnx_int8": None})
    r = _row(rs, "e4_late_fusion", "onnx_int8", None)
    assert r["p95"] is None
    assert apply_rule(rs, CFG)["shipped"]["variant"] == "onnx_int8_headfp32"
    table = edge_table(rs)
    assert "—" in table and table.startswith("| Pipeline |")


def test_load_edge_skips_parity(tmp_path: Path):
    (tmp_path / "edge").mkdir()
    (tmp_path / "bench").mkdir()
    (tmp_path / "edge/a.json").write_text(json.dumps(_single(E3C, "onnx_fp32", 0.8)), "utf-8")
    (tmp_path / "bench/x.json").write_text(json.dumps(_bench("p", "onnx_fp32", 50)), "utf-8")
    (tmp_path / "bench/parity_laptop-cpu_onnx_fp32.json").write_text("{}", "utf-8")
    acc, bench = load_edge(tmp_path)
    assert len(acc) == 1 and len(bench) == 1


def test_report_cli_adds_edge_section(tmp_path: Path):
    from baykus.report.cli import main

    acc, bench, runs = data()
    for folder, recs in (("edge", acc), ("bench", bench)):
        (tmp_path / folder).mkdir()
        for i, r in enumerate(recs):
            (tmp_path / folder / f"{i}.json").write_text(json.dumps(r), encoding="utf-8")
    (tmp_path / "runs").mkdir()
    for r in runs:  # ana tablo için gereken alanlar
        r.setdefault("model", {"mode": "late_fusion"})
    out = tmp_path / "results.md"
    args = ["--runs", str(tmp_path / "runs"), "--compare", str(tmp_path / "yok"),
            "--m3ot", str(tmp_path / "yok"), "--out", str(out), "--fig", str(tmp_path / "f.png"),
            "--edge-results", str(tmp_path),
            "--edge-config", str(REPO / "configs/edge/e9_edge.yaml")]  # fmt: skip
    for i, r in enumerate(runs):
        (tmp_path / "runs" / f"{i}.json").write_text(json.dumps(r), encoding="utf-8")
    assert main(args) == 0
    text = out.read_text(encoding="utf-8")
    assert "## Edge (E9)" in text and "### Sevk kuralı (ADR 0007)" in text
    assert "**Sevk:** e4_late_fusion · onnx_int8" in text


def test_report_cli_without_edge_results_has_no_section(tmp_path: Path):
    from baykus.report.cli import main

    (tmp_path / "runs").mkdir()
    out = tmp_path / "results.md"
    args = ["--runs", str(tmp_path / "runs"), "--compare", str(tmp_path / "yok"),
            "--m3ot", str(tmp_path / "yok"), "--out", str(out), "--fig", str(tmp_path / "f.png"),
            "--edge-results", str(tmp_path / "bos")]  # fmt: skip
    assert main(args) == 0 and "Edge" not in out.read_text(encoding="utf-8")


def test_report_cli_test_split_has_no_edge_section(tmp_path: Path):
    """Sevk kuralı val'de uygulanır; test raporunda edge bölümü ve kural izi olmamalı."""
    from baykus.report.cli import main

    acc, bench, runs = data()
    for folder, recs in (("edge", acc), ("bench", bench)):
        (tmp_path / folder).mkdir()
        for i, r in enumerate(recs):
            (tmp_path / folder / f"{i}.json").write_text(json.dumps(r), encoding="utf-8")
    (tmp_path / "runs").mkdir()
    out = tmp_path / "results-test.md"
    args = ["--runs", str(tmp_path / "runs"), "--compare", str(tmp_path / "yok"),
            "--m3ot", str(tmp_path / "yok"), "--out", str(out), "--fig", str(tmp_path / "f.png"),
            "--split", "test", "--edge-results", str(tmp_path)]  # fmt: skip
    assert main(args) == 0 and "Sevk kuralı" not in out.read_text(encoding="utf-8")
