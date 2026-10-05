import numpy as np
import pytest

from baykus.data.illumination import assign, fit_thresholds, mean_brightness


def test_mean_brightness():
    assert mean_brightness(np.full((4, 4, 3), 200, np.uint8)) == pytest.approx(200)
    assert mean_brightness(np.zeros((4, 4, 3), np.uint8)) == 0


def test_fit_thresholds_matches_proportions():
    values = np.arange(100, dtype=float)  # 0..99
    t_dark, t_night = fit_thresholds(values, {"dark_night": 30, "night": 20, "day": 50})
    labels = assign(values, (t_dark, t_night))
    assert labels.count("dark_night") == 30
    assert labels.count("night") == 20
    assert labels.count("day") == 50
    assert labels[0] == "dark_night" and labels[-1] == "day"


def test_fit_thresholds_scales_to_sample_size():
    values = np.linspace(0, 255, 1000)
    t = fit_thresholds(values, {"dark_night": 8493, "night": 5468, "day": 14478})
    labels = assign(values, t)
    assert labels.count("dark_night") == pytest.approx(1000 * 8493 / 28439, abs=2)


def test_cli_illum_writes_csv_thresholds_and_mosaics(raw_root, tmp_path):
    import csv
    import json

    from baykus.data.cli import main
    from tests.conftest import make_raw_dataset

    bright = {"00001": 10, "00002": 120, "00003": 240}
    make_raw_dataset(raw_root, "train", list(bright), brightness=bright)
    for split in ("val", "test"):
        make_raw_dataset(raw_root, split, [])
    meta, out, fig = tmp_path / "meta", tmp_path / "out", tmp_path / "fig"
    main(["verify", "--raw", str(raw_root), "--meta", str(meta)])
    main(["convert", "--raw", str(raw_root), "--meta", str(meta), "--out", str(out)])
    assert main(["illum", "--out", str(out), "--meta", str(meta), "--fig", str(fig)]) == 0

    with (meta / "illumination.csv").open(encoding="utf-8") as f:
        rows = {r["stem"]: r for r in csv.DictReader(f)}
    assert set(rows) == set(bright)
    assert rows["00001"]["illumination"] == "dark_night"
    assert rows["00003"]["illumination"] == "day"
    th = json.loads((meta / "illumination_thresholds.json").read_text(encoding="utf-8"))
    assert th["t_dark"] <= th["t_night"]
    assert (fig / "illum_samples_day.jpg").exists()
