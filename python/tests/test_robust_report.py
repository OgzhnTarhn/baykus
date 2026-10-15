import json
from pathlib import Path

from baykus.report.robust import load_robust, robust_markdown, robust_plot
from baykus.robustness.cli import main


def _rec(config, mode, cid, corruption, severity, target, map50):
    return {
        "run_id": "r",
        "config": config,
        "split": "val",
        "label_set": "ir",
        "model": {"mode": mode},
        "metrics": {"map50": map50, "ci95": {"map50": [0.1, 0.9]}},
        "robustness": {
            "id": cid,
            "corruption": corruption,
            "severity": severity,
            "target": target,
            "seed": 42,
        },
    }


RECS = [
    _rec("configs/e1_rgb.yaml", "rgb", "clean", "clean", 0.0, "none", 0.76),
    _rec("configs/e1_rgb.yaml", "rgb", "darken-0.5", "darken", 0.5, "rgb", 0.6),
    _rec("configs/e2_ir.yaml", "ir", "clean", "clean", 0.0, "none", 0.8),
    _rec("configs/e2_ir.yaml", "ir", "noise-5", "noise", 5.0, "ir", 0.7),
]


def _rows(md: str, title: str) -> list[str]:
    block = md.split(f"## {title}")[1].split("##")[0]
    head = ("| Şiddet", "| Koşul")
    return [ln for ln in block.splitlines() if ln.startswith("| ") and not ln.startswith(head)]


def test_markdown_cells_applicability_and_missing():
    md = robust_markdown(RECS, "figures/results/robustness.png")
    dark = _rows(md, "Karartma")
    assert dark[0] == "| 1 (temiz) | 76.0 [10.0, 90.0] | 80.0 [10.0, 90.0] |"
    assert dark[1] == "| 0.5 | 60.0 [10.0, 90.0] | uygulanmaz |"
    assert dark[2] == "| 0.25 | — | uygulanmaz |"  # ölçülmedi ≠ uygulanmaz
    noise = _rows(md, "Gauss gürültüsü")
    assert noise[0] == "| 0 (temiz) | 76.0 [10.0, 90.0] | 80.0 [10.0, 90.0] |"
    assert noise[1] == "| 5 | uygulanmaz | 70.0 [10.0, 90.0] |"
    mis = _rows(md, "Hizasızlık")
    assert mis[0] == "| 0 (temiz) | 76.0 [10.0, 90.0] | 80.0 [10.0, 90.0] |"
    assert all(r.endswith("| uygulanmaz | uygulanmaz |") for r in mis[1:])
    loss = _rows(md, "Kamera kaybı")
    assert loss == [
        "| Temiz | 76.0 [10.0, 90.0] | 80.0 [10.0, 90.0] |",
        "| RGB kaybı | — | uygulanmaz |",
        "| IR kaybı | uygulanmaz | — |",
    ]
    assert "![Sağlamlık eğrileri](figures/results/robustness.png)" in md


def test_load_robust_ignores_plain_runs(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps(RECS[0]), encoding="utf-8")
    plain = {k: v for k, v in RECS[0].items() if k != "robustness"}
    (tmp_path / "b.json").write_text(json.dumps(plain), encoding="utf-8")
    assert len(load_robust(tmp_path)) == 1


def test_plot_and_cli(tmp_path: Path):
    assert robust_plot(RECS, tmp_path / "f.png").stat().st_size > 0
    rob = tmp_path / "robust"
    rob.mkdir()
    for i, r in enumerate(RECS):
        (rob / f"{i}.json").write_text(json.dumps(r), encoding="utf-8")
    out, fig = tmp_path / "docs" / "robustness.md", tmp_path / "docs" / "figs" / "r.png"
    assert main(["report", "--robust", str(rob), "--out", str(out), "--fig", str(fig)]) == 0
    assert fig.exists() and "figs/r.png" in out.read_text(encoding="utf-8")
    assert main(["report", "--robust", str(tmp_path / "bos"), "--out", str(out)]) == 2


def test_each_model_keeps_one_color_across_panels():
    from baykus.report.robust import robust_figure

    e3 = ("configs/e3_early_fusion.yaml", "early_fusion")
    recs = RECS + [
        _rec(*e3, "clean", "clean", 0.0, "none", 0.85),
        _rec(*e3, "noise-5", "noise", 5.0, "ir", 0.8),
        _rec(*e3, "darken-0.5", "darken", 0.5, "rgb", 0.8),
    ]
    fig = robust_figure(recs)
    colors: dict[str, set] = {}
    for ax in fig.axes:
        for line in ax.get_lines():
            colors.setdefault(line.get_label(), set()).add(line.get_color())
    assert set(colors) == {"e1_rgb", "e2_ir", "e3_early_fusion"}
    assert all(len(c) == 1 for c in colors.values()), colors
    assert len({next(iter(c)) for c in colors.values()}) == 3


def test_load_robust_filters_split(tmp_path: Path):
    test_rec = dict(RECS[0], split="test")
    (tmp_path / "a.json").write_text(json.dumps(RECS[0]), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(test_rec), encoding="utf-8")
    assert [r["split"] for r in load_robust(tmp_path)] == ["val"]
    assert [r["split"] for r in load_robust(tmp_path, split="test")] == ["test"]


def test_duplicate_model_condition_is_an_error():
    import pytest

    dup = dict(RECS[0], run_id="r2")
    with pytest.raises(ValueError, match="r2"):
        robust_markdown([*RECS, dup], "f.png")


def test_cli_report_split(tmp_path: Path):
    rob = tmp_path / "robust"
    rob.mkdir()
    (rob / "a.json").write_text(json.dumps(RECS[0]), encoding="utf-8")
    out = tmp_path / "r.md"
    args = ["report", "--robust", str(rob), "--out", str(out), "--fig", str(tmp_path / "f.png")]
    assert main([*args, "--split", "test"]) == 2
    assert main(args) == 0 and "(val, IR etiketleri)" in out.read_text(encoding="utf-8")
