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
    assert dark[0] == "| 1 (temiz) | 76.0 [10.0, 90.0] | uygulanmaz |"
    assert dark[1] == "| 0.5 | 60.0 [10.0, 90.0] | uygulanmaz |"
    assert dark[2] == "| 0.25 | — | uygulanmaz |"  # ölçülmedi ≠ uygulanmaz
    noise = _rows(md, "Gauss gürültüsü")
    assert noise[0] == "| 0 (temiz) | uygulanmaz | 80.0 [10.0, 90.0] |"
    mis = _rows(md, "Hizasızlık")
    assert mis and all(r.endswith("| uygulanmaz | uygulanmaz |") for r in mis)
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
