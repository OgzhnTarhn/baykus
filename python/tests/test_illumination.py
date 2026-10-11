import numpy as np
import pytest

from baykus.data.illumination import (
    DAY_MIN_BRIGHTNESS,
    assign,
    fit_thresholds,
    mean_brightness,
    thresholds,
)


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


def test_thresholds_fixed_day_boundary_fitted_dark():
    # ADR 0005: gündüz sınırı 160 elle etiketli örnekten sabit; karanlık gece resmi orana uydurulur
    values = np.linspace(0, 255, 1000)
    t_dark, t_day = thresholds(values)
    assert t_day == DAY_MIN_BRIGHTNESS
    assert t_dark == fit_thresholds(values, {"dark_night": 8493, "night": 5468, "day": 14478})[0]
    labels = assign(values, (t_dark, t_day))
    assert labels[int(np.searchsorted(values, 100))] == "night"  # eski eşikte (82.6) gündüzdü


def test_thresholds_day_never_below_dark():
    t_dark, t_day = thresholds(np.linspace(150, 255, 100))  # hepsi parlak
    assert t_day >= t_dark


def test_cli_illum_relabel_rewrites_labels_without_images(tmp_path):
    import csv
    import json

    from baykus.data.cli import main

    meta = tmp_path / "meta"
    meta.mkdir()
    rows = [("train", f"{i:05d}", b) for i, b in enumerate([5, 20, 60, 100, 113, 115, 200], 1)]
    with (meta / "illumination.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "stem", "brightness", "illumination"])
        for s, st, b in rows:
            w.writerow([s, st, f"{b:.2f}", "day"])  # eski etiketler yanlış/önemsiz
    assert main(["illum-relabel", "--meta", str(meta)]) == 0
    with (meta / "illumination.csv").open(encoding="utf-8") as f:
        got = {r["stem"]: (r["brightness"], r["illumination"]) for r in csv.DictReader(f)}
    assert got["00004"] == ("100.00", "night") and got["00005"][1] == "night"
    assert got["00006"][1] == "day" and got["00007"][1] == "day" and got["00001"][1] == "dark_night"
    th = json.loads((meta / "illumination_thresholds.json").read_text(encoding="utf-8"))
    assert th["t_day"] == DAY_MIN_BRIGHTNESS and th["t_dark"] < th["t_day"]
    assert "ADR 0005" in th["method"]


def test_cli_slices_from_coco_and_csv(tmp_path):
    import csv
    import json

    from baykus.data.cli import main

    out, meta = tmp_path / "out", tmp_path / "meta"
    (out / "coco").mkdir(parents=True)
    meta.mkdir()
    # 00002 dışlanmış: kimlikler dosya adından okunmalı, sıradan değil
    images = [{"id": 1, "file_name": "00001"}, {"id": 2, "file_name": "00003"}]
    for split in ("train", "val", "test"):
        imgs = images if split == "val" else []
        coco = {"images": imgs, "annotations": [], "categories": []}
        (out / "coco" / f"{split}_ir.json").write_text(json.dumps(coco), encoding="utf-8")
    with (meta / "illumination.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "stem", "brightness", "illumination"])
        w.writerow(["val", "00001", "150.00", "day"])
        w.writerow(["val", "00003", "60.00", "night"])
    assert main(["slices", "--out", str(out), "--meta", str(meta)]) == 0
    s = json.loads((meta / "slices_val.json").read_text(encoding="utf-8"))
    assert s == {"illumination": {"day": [1], "night": [2], "dark_night": []}}
    assert (meta / "slices_train.json").exists()


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
    assert th["t_dark"] <= th["t_day"]
    assert (fig / "illum_samples_day.jpg").exists()
