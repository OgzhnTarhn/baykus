from pathlib import Path

from baykus.health.fit import STATES
from baykus.health.report import calibration_plot, health_markdown

FL0 = {"lost": 0.0, "noisy": 0.0, "dark": 0.0, "blurry": 0.0}
TH = {
    "states": {
        n: {
            "feature": s.feature,
            "direction": s.direction,
            "threshold": 9.5,
            "train": {"tpr": 0.98, "fpr": 0.01},
        }
        for n, s in STATES.items()
    },
    "dark_brightness": 48.66,
    "train_clean_gate_rate": 0.004,
    "source": {"split": "train", "n": 500, "seed": 42},
}


def _row(cond, base, gate, diff, ci, rate=0.0):
    return {"condition": cond, "n_images": 1469, "gate_rate": rate, "changed": 0,
            "base_map50": base, "gate_map50": gate, "diff": diff, "ci95": ci,
            "significant": ci[0] is not None and (ci[0] > 0 or ci[1] < 0)}  # fmt: skip


def _gate(split: str, acc: bool) -> dict:
    rows = [
        _row("clean", 0.851, 0.850, -0.001, [-0.003, 0.0], 0.01),
        _row("noise-20", 0.535, 0.700, 0.165, [0.15, 0.18], 0.97),
        _row("misalign-20", 0.266, 0.720, 0.454, [0.43, 0.47], 0.95),
    ]
    det = {r["condition"]: {"gate_rate": r["gate_rate"], "rgb": FL0, "ir": FL0,
                            "misaligned": 0.0} for r in rows}  # fmt: skip
    return {
        "split": split,
        "n_boot": 1000,
        "rows": rows,
        "harmful": {"conditions": ["noise-20", "misalign-20"], "diff": 0.31, "ci95": [0.3, 0.32]},
        "detection": det,
        "acceptance": {
            "k1": {"value": -0.001, "limit": -0.005, "passed": True},
            "k2": {"ci95": [0.3, 0.32], "passed": True},
            "k3": {"value": 0.01, "limit": 0.02, "passed": True},
        }
        if acc
        else None,
    }


BINS = [{"lo": i / 15, "hi": (i + 1) / 15, "n": 0, "conf": None, "acc": None} for i in range(15)]
BINS[14] = {"lo": 14 / 15, "hi": 1.0, "n": 10, "conf": 0.95, "acc": 0.5}
CALIB = {
    "table": {"x": [0.95], "y": [0.5]},
    "test": {
        "n_eval": 20,
        "ece_before": 0.45,
        "ece_after": 0.0,
        "map50_before": 0.8144,
        "map50_after": 0.8143,
        "delta_map50": -0.0001,
        "bins_before": BINS,
        "bins_after": BINS,
    },  # fmt: skip
    "k4": {"value": 0.0001, "limit": 0.001, "passed": True},
}


def test_markdown_with_all_parts_has_tables_and_marks():
    md = health_markdown(TH, _gate("val", True), _gate("test", False), CALIB, "fig.png")
    for head in (
        "## Eşikler (train)",
        "## Arıza tespiti (val)",
        "## Kapı (val)",
        "## Kabul ölçütleri",
        "## Test kontrolü",
        "## Kalibrasyon (test)",
        "## Sınırlamalar",
    ):
        assert head in md
    assert md.count("✅") == 4 and "❌" not in md
    assert "+45.4 [+43.0, +47.0]" in md  # misalign-20 farkı, puan
    assert "| misaligned | misalign_px | ≥ | 9.500 | 98.0 | 1.0 |" in md
    assert "0.450 → 0.000" in md and "![" in md and "fig.png" in md


def test_failed_criterion_is_marked():
    g = _gate("val", True)
    g["acceptance"]["k3"] = {"value": 0.035, "limit": 0.02, "passed": False}
    md = health_markdown(TH, g, None, CALIB, "fig.png")
    assert "❌" in md and "3.50%" in md


def test_markdown_without_test_and_calibration():
    md = health_markdown(TH, _gate("val", True), None, None, "x.png")
    assert md.count("henüz ölçülmedi") == 2 and "Kabul 4" in md


def test_plot_written(tmp_path: Path):
    assert calibration_plot(CALIB, tmp_path / "c.png").stat().st_size > 0


def test_cli_report(tmp_path: Path):
    from baykus.health.cli import main
    from baykus.runinfo import write_json

    h = tmp_path / "health"
    write_json(h / "thresholds.json", TH)
    write_json(h / "gate_val.json", _gate("val", True))
    out, fig = tmp_path / "docs" / "health.md", tmp_path / "docs" / "figures" / "cal.png"
    args = ["report", "--health", str(h), "--out", str(out), "--fig", str(fig)]
    assert main(args) == 0
    md = out.read_text(encoding="utf-8")
    assert "## Kapı (val)" in md and md.count("henüz ölçülmedi") == 2 and not fig.exists()
    write_json(h / "calibration_e3c_onnx_fp32.json", CALIB)
    assert main(args) == 0
    assert fig.exists() and "figures/cal.png" in out.read_text(encoding="utf-8")


def test_cli_report_without_thresholds_is_error(tmp_path: Path, capsys):
    from baykus.health.cli import main

    assert main(["report", "--health", str(tmp_path), "--out", str(tmp_path / "x.md")]) == 2
    assert "thresholds.json" in capsys.readouterr().err
