import json
from pathlib import Path

from baykus.report.cli import main
from baykus.report.plots import illumination_plot
from baykus.report.tables import (
    compare_table,
    fmt,
    label_set_table,
    load_compares,
    load_runs,
    main_table,
)


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


def _rec(run_id, ls, map50, ill=None):
    return {
        "run_id": run_id,
        "split": "val",
        "label_set": ls,
        "model": {"mode": "rgb"},
        "metrics": {
            "map50": map50,
            "ci95": {"map50": [map50 - 0.01, map50 + 0.01]},
            "per_illumination": ill or {"day": 0.80, "night": 0.70, "dark_night": None},
        },
    }


def test_label_set_table_delta_and_missing_pair():
    runs = [
        _rec("a", "ir", 0.761),
        _rec("a", "rgb", 0.780, {"day": 0.83, "night": 0.70, "dark_night": None}),
        _rec("b", "ir", 0.5),  # rgb kaydı yok → satır yok
    ]
    table = label_set_table(runs)
    rows = [r for r in table.splitlines() if r.startswith("| a") or r.startswith("| b")]
    assert rows == ["| a | rgb | 76.1 [75.1, 77.1] | 78.0 [77.0, 79.0] | +1.9 | +3.0 | +0.0 | — |"]


def test_compare_table_marks_significance():
    c = {
        "name": "e3-vs-e1_val_ir",
        "overall": {"diff": 0.05, "ci95": [0.02, 0.08], "significant": True},
        "per_illumination": {
            "day": {"diff": -0.01, "ci95": [-0.03, 0.01], "significant": False},
            "night": None,
            "dark_night": {"diff": 0.1, "ci95": [0.05, 0.15], "significant": True},
        },
    }
    row = compare_table([c]).splitlines()[-1]
    assert row == (
        "| e3-vs-e1_val_ir | +5.0 [+2.0, +8.0] * | -1.0 [-3.0, +1.0] | — | +10.0 [+5.0, +15.0] * |"
    )


def test_load_compares_missing_folder(tmp_path: Path):
    assert load_compares(tmp_path / "yok") == []


def test_cli_adds_e5_and_compare_sections(tmp_path: Path):
    runs, comp = tmp_path / "runs", tmp_path / "compare"
    runs.mkdir()
    comp.mkdir()
    for r in (_rec("a", "ir", 0.761), _rec("a", "rgb", 0.78)):
        (runs / f"a_{r['label_set']}.json").write_text(json.dumps(r), encoding="utf-8")
    c = {"name": "e3-vs-e1_val_ir", "split": "val", "overall": None, "per_illumination": {}}
    (comp / "x.json").write_text(json.dumps(c), encoding="utf-8")
    out = tmp_path / "results.md"
    argv = ["--runs", str(runs), "--compare", str(comp), "--out", str(out)]
    assert main([*argv, "--fig", str(tmp_path / "f.png")]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## Etiket seti karşılaştırması (E5)" in text and "+1.9" in text
    assert "## Eşleştirilmiş farklar" in text and "e3-vs-e1_val_ir" in text


def test_load_compares_filters_split(tmp_path: Path):
    for name, split in (("a", "val"), ("b", "test"), ("c", None)):
        c = {"name": name, "overall": None, "per_illumination": {}}
        if split:
            c["split"] = split
        (tmp_path / f"{name}.json").write_text(json.dumps(c), encoding="utf-8")
    assert [c["name"] for c in load_compares(tmp_path, split="val")] == ["a"]
    assert len(load_compares(tmp_path)) == 3
