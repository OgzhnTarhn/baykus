import json
from pathlib import Path

from baykus.report.cli import main
from baykus.report.plots import illumination_plot
from baykus.report.tables import fmt, load_runs, main_table


def _run(run_id, mode, map50, ci, ill, split="val", ls="ir"):
    return {
        "run_id": run_id,
        "split": split,
        "label_set": ls,
        "model": {"family": "yolo26", "size": "s", "mode": mode},
        "metrics": {
            "map50": map50,
            "map50_95": None if map50 is None else map50 / 2,
            "per_illumination": ill,
            "ci95": {"map50": ci, "per_illumination": {k: [None, None] for k in ill}},
        },
    }


R1 = _run(
    "2026-10-14_e1-rgb_s_abc1234",
    "rgb",
    0.6234,
    [0.6011, 0.6402],
    {"day": 0.7, "night": 0.5, "dark_night": None},
)


def test_fmt():
    assert fmt(None) == "—"
    assert fmt(0.6234) == "62.3"
    assert fmt(0.6234, [0.6011, 0.6402]) == "62.3 [60.1, 64.0]"
    assert fmt(0.5, [None, None]) == "50.0"


def test_main_table_exact():
    expected = (
        "| Çalıştırma | Mod | Split | Etiket | mAP50 [%95 GA] | mAP50-95 | Gündüz | Gece "
        "| Karanlık gece |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        "| 2026-10-14_e1-rgb_s_abc1234 | rgb | val | ir | 62.3 [60.1, 64.0] | 31.2 | 70.0 "
        "| 50.0 | — |"
    )
    assert main_table([R1]) == expected


def test_missing_metric_is_dash_not_zero():
    # Review Focus 5
    r = _run("x", "ir", None, [None, None], {"day": None, "night": None, "dark_night": None})
    row = main_table([r]).splitlines()[-1]
    assert "0.0" not in row and row.count("—") == 5


def test_load_runs_filters(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps(R1), encoding="utf-8")
    other = _run(
        "b", "ir", 0.5, [0.4, 0.6], {"day": 0.5, "night": 0.5, "dark_night": 0.5}, ls="rgb"
    )
    (tmp_path / "b.json").write_text(json.dumps(other), encoding="utf-8")
    assert [r["run_id"] for r in load_runs(tmp_path)] == [R1["run_id"], "b"]
    assert [r["run_id"] for r in load_runs(tmp_path, label_set="ir")] == [R1["run_id"]]


def test_plot_and_cli(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "a.json").write_text(json.dumps(R1), encoding="utf-8")
    p = illumination_plot([R1], tmp_path / "f" / "ill.png")
    assert p.exists() and p.stat().st_size > 0
    out, fig = tmp_path / "results.md", tmp_path / "fig.png"
    assert main(["--runs", str(runs), "--out", str(out), "--fig", str(fig)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "62.3 [60.1, 64.0]" in text and "results/runs" in text and fig.exists()
