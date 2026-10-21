import pytest

from baykus.motion.audit import Sample
from baykus.motion.labels import LabelRow
from baykus.motion.measure import FrameOut, k9, score, vs_human

EGO_OK = {"ok": True, "map_gen": 1, "frame_to_map": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}


def _box(i: int) -> tuple[float, float, float, float]:
    return (100.0 * i, 50.0, 100.0 * i + 40, 80.0)


def _run(per_frame: dict[int, list[tuple[int, tuple, str]]]) -> list[FrameOut]:
    return [
        FrameOut(
            f,
            EGO_OK,
            [
                {"track_id": tid, "box": box, "motion": m, "map_xy": (box[0], box[1])}
                for tid, box, m in per_frame[f]
            ],
        )
        for f in sorted(per_frame)
    ]


def _gt(tracks: dict[int, range]) -> dict[int, dict[int, tuple]]:
    return {tid: {f: _box(tid) for f in frames} for tid, frames in tracks.items()}


def test_denominators_count_unknown_and_missed():
    gt = {"s": _gt({1: range(10, 11), 2: range(10, 11), 3: range(10, 11), 4: range(10, 11)})}
    labels = [LabelRow("s", 10, t, 0.2, "moving") for t in (1, 2, 3, 4)]
    # Motor: kendi kimlikleri (100+), kutular GT ile aynı; 4. araç kaçırıldı.
    run = {"s": _run({10: [(101, _box(1), "moving"), (102, _box(2), "moving"),
                           (103, _box(3), "unknown")]})}  # fmt: skip
    s = score(labels, run, gt, matched=True)
    lab = s["labeled"]
    assert lab["R_mov"] == 0.5 and lab["n_moving"] == 4
    assert lab["confusion"]["moving"] == {"moving": 2, "unknown": 1, "missed": 1}
    assert lab["R_sta"] is None and lab["BA"] is None
    assert lab["coverage"] == 0.5 and lab["BA_decided"] is None


def test_oracle_uses_gt_ids_and_ba():
    gt = {"s": _gt({1: range(1, 3), 2: range(1, 3)})}
    labels = [LabelRow("s", f, 1, 0.2, "moving") for f in (1, 2)]
    labels += [LabelRow("s", f, 2, 0.0, "stationary") for f in (1, 2)]
    run = {"s": _run({1: [(1, _box(1), "moving"), (2, _box(2), "moving")],
                      2: [(1, _box(1), "moving"), (2, _box(2), "stationary")]})}  # fmt: skip
    lab = score(labels, run, gt, matched=False)["labeled"]
    assert lab["R_mov"] == 1.0 and lab["R_sta"] == 0.5 and lab["BA"] == 0.75
    assert lab["BA_decided"] == 0.75 and lab["coverage"] == 1.0


def test_unmatched_moving_counts_against_precision():
    gt = {"s": _gt({1: range(5, 6), 2: range(5, 6)})}
    labels = [LabelRow("s", 5, 1, 0.2, "moving"), LabelRow("s", 5, 2, None, "band")]
    run = {"s": _run({5: [(7, _box(1), "moving"), (8, _box(2), "moving"),
                          (9, _box(5), "moving")]})}  # fmt: skip
    p = score(labels, run, gt, matched=True)["precision"]
    assert p["moving"] == {"total": 3, "correct": 1, "wrong_label": 0, "excluded": 1,
                           "unmatched": 1, "P": pytest.approx(1 / 3)}  # fmt: skip
    assert p["false_moving_per_frame"] == 1.0  # yanlış etiket + eşleşmeyen, 1 kare


def test_band_excluded_but_reported():
    gt = {"s": _gt({1: range(3, 4), 2: range(3, 4)})}
    labels = [LabelRow("s", 3, 1, 0.0, "stationary"), LabelRow("s", 3, 2, 0.05, "band")]
    run = {"s": _run({3: [(1, _box(1), "stationary"), (2, _box(2), "moving")]})}
    s = score(labels, run, gt, matched=False)
    assert s["labeled"]["n"] == 1 and s["labeled"]["R_sta"] == 1.0
    assert s["excluded"] == {"band": {"n": 1, "pred": {"moving": 1}}}


def _switch_track(change: int, n: int, switch_pred: int | None):
    """Tek iz: 1..change−1 hareketli, change..n duran; ürün `switch_pred`'de duran'a geçer."""
    gt = {"s": _gt({1: range(1, n + 1)})}
    labels = [
        LabelRow("s", f, 1, 0.2 if f < change else 0.0, "moving" if f < change else "stationary")
        for f in range(1, n + 1)
    ]
    run = {
        "s": _run(
            {
                f: [(1, _box(1), "stationary" if switch_pred and f >= switch_pred else "moving")]
                for f in range(1, n + 1)
            }
        )
    }
    return labels, run, gt


def test_stable_part_excludes_transitions():
    labels, run, gt = _switch_track(31, 60, 36)
    s = score(labels, run, gt, matched=False, k=15)
    # Geçiş 31: 16..46 kararlı bölüm dışı → 15 hareketli (1..15) + 14 duran (47..60).
    assert s["stable"]["n_moving"] == 15 and s["stable"]["n_stationary"] == 14
    assert s["stable"]["BA"] == 1.0
    assert s["labeled"]["n"] == 60 and s["labeled"]["R_sta"] == pytest.approx(25 / 30)


def test_decision_delay():
    labels, run, gt = _switch_track(31, 60, 36)
    d = score(labels, run, gt, matched=False)["delay"]
    assert d == {"transitions": 1, "reached": 1, "not_reached": 0, "median": 5.0, "p90": 5.0}
    labels, run, gt = _switch_track(31, 60, None)
    d = score(labels, run, gt, matched=False)["delay"]
    assert d["not_reached"] == 1 and d["median"] is None


def test_bootstrap_by_track_deterministic():
    gt = {"s": _gt({t: range(1, 11) for t in range(1, 7)})}
    labels, frames = [], {f: [] for f in range(1, 11)}
    for t in range(1, 7):
        lab = "moving" if t % 2 else "stationary"
        hit = t not in (5, 6)
        for f in range(1, 11):
            labels.append(LabelRow("s", f, t, None, lab))
            pred = lab if hit else ("stationary" if lab == "moving" else "moving")
            frames[f].append((t, _box(t), pred))
    run = {"s": _run(frames)}
    a = score(labels, run, gt, matched=False)["bootstrap"]
    b = score(labels, run, gt, matched=False)["bootstrap"]
    assert a == b and a["resamples"] == 1000 and a["seed"] == 42 and a["n_tracks"] == 6
    lo, hi = a["BA"]
    assert lo <= 2 / 3 <= hi and lo < hi


def test_all_unknown_fails_k9():
    gt = {"s": _gt({1: range(1, 3), 2: range(1, 3)})}
    labels = [LabelRow("s", f, 1, 0.2, "moving") for f in (1, 2)]
    labels += [LabelRow("s", f, 2, 0.0, "stationary") for f in (1, 2)]
    run = {"s": _run({f: [(1, _box(1), "unknown"), (2, _box(2), "unknown")] for f in (1, 2)})}
    g = k9(score(labels, run, gt, matched=False))
    assert g == {"BA": 0.0, "R_mov": 0.0, "R_sta": 0.0, "pass": False}
    run = {"s": _run({f: [(1, _box(1), "moving"), (2, _box(2), "stationary")] for f in (1, 2)})}
    assert k9(score(labels, run, gt, matched=False))["pass"] is True


def test_vs_human_wilson():
    gt = {"s": _gt({1: range(1, 4)})}
    run = {"s": _run({1: [(1, _box(1), "moving")], 2: [(1, _box(1), "stationary")],
                      3: [(1, _box(1), "unknown")]})}  # fmt: skip
    samples = {"a": Sample("s", 1, 1, "uniform"), "b": Sample("s", 2, 1, "uniform"),
               "c": Sample("s", 3, 1, "uniform"), "d": Sample("s", 1, 1, "uniform")}  # fmt: skip
    answers = {"a": "moving", "b": "moving", "c": "moving", "d": "unsure"}
    h = vs_human(samples, answers, run, gt, matched=False)
    assert h["n"] == 3 and h["agree"] == 1 and h["product_unknown"] == 1
    assert (h["decided"]["n"], h["decided"]["agree"], h["decided"]["rate"]) == (2, 1, 0.5)
    assert 0 < h["ci95"][0] < 1 / 3 < h["ci95"][1] < 1
