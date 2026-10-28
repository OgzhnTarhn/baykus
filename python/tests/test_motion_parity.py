import copy

from baykus.motion.parity import compare

FM = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _ego(frame_to_map=FM, keyframe=False, relocalized=False, merge=None) -> dict:
    return {
        "ok": True,
        "source": "ir",
        "map_gen": 1,
        "provisional": False,
        "frame_to_map": list(frame_to_map),
        "keyframe": keyframe,
        "relinked": False,
        "relocalized": relocalized,
        "merge": merge,
    }


def _health() -> dict:
    return {
        "rgb": {"flags": [], "frozen": False},
        "ir": {"flags": [], "frozen": False},
        "pair": {"misaligned": False},
    }


def _track(tid: int, motion: str, v: float | None) -> dict:
    return {"track_id": tid, "motion": motion, "speed_rel": v, "map_xy": [10.0, 20.0]}


def _frame(n: int, tracks: list[dict], ego: dict | None = None) -> dict:
    return {
        "frame": n,
        "reset_before": None,
        "tracks": tracks,
        "ego": _ego() if ego is None else ego,
        "health": _health(),
    }


def _run(rows: list[tuple[str, float]]) -> list[dict]:
    """Tek iz, her kare bir (motion, v); 100 karelik %99 oranı için doldurma yok."""
    return [_frame(i + 1, [_track(1, m, v)]) for i, (m, v) in enumerate(rows)]


def test_parity_identical_passes():
    ref = {"scenarios": {"clean": _run([("moving", 0.5)] * 5)}}
    out = compare(ref, copy.deepcopy(ref))
    assert out["verdict"] == "pass"
    assert out["per_scenario"]["clean"]["compared"] == 5


def test_parity_ego_decision_mismatch_fails():
    ref = {"scenarios": {"clean": _run([("moving", 0.5)] * 3)}}
    cpp = copy.deepcopy(ref)
    cpp["scenarios"]["clean"][1]["ego"]["keyframe"] = True
    out = compare(ref, cpp)
    assert out["verdict"] == "fail" and out["per_scenario"]["clean"]["errors"] == 1


def test_parity_free_row_propagation():
    # Bantta ayrışma serbest; sonraki satır bant dışında ama kararlar hâlâ ayrı → yine serbest;
    # kararlar eşitlenince karşılaştırma döner. Bantta kararları aynı olan satır karşılaştırılır.
    ref = _run([("stationary", 0.0301), ("stationary", 0.05), ("moving", 0.2), ("moving", 0.2)])
    cpp = _run([("unknown", 0.0299), ("unknown", 0.05), ("moving", 0.2), ("moving", 0.2)])
    rows = 200
    ref += _run([("moving", 0.5)] * rows)[4:]
    cpp += _run([("moving", 0.5)] * rows)[4:]
    out = compare({"scenarios": {"s": ref}}, {"scenarios": {"s": cpp}})
    s = out["per_scenario"]["s"]
    assert s["free"] == 2 and s["errors"] == 0 and out["verdict"] == "pass"
    same = _run([("stationary", 0.0301)])
    out = compare({"scenarios": {"s": same}}, {"scenarios": {"s": copy.deepcopy(same)}})
    assert out["per_scenario"]["s"]["in_band"] == 1
    assert out["per_scenario"]["s"]["compared"] == 1 and out["per_scenario"]["s"]["free"] == 0
    bad = _run([("moving", 0.5)])
    worse = _run([("stationary", 0.5)])
    out = compare({"scenarios": {"s": bad}}, {"scenarios": {"s": worse}})
    assert out["verdict"] == "fail"


def _reloc_case(dx: float) -> dict:
    clean = [_frame(n, []) for n in (1, 2, 3)]
    moved = [FM[0], FM[1], FM[2] + dx, FM[3], FM[4], FM[5]]
    merge = {"from": 2, "to": 1, "transform": FM}
    seek = [_frame(1, []), _frame(2, [], _ego(moved, relocalized=True, merge=merge))]
    return {"scenarios": {"clean": clean, "seek": seek}}


def test_parity_reloc_tolerance():
    ok = _reloc_case(1.5)
    assert compare(ok, copy.deepcopy(ok))["kabul10b"]["pass"]
    far = _reloc_case(2.5)
    out = compare(far, copy.deepcopy(far))
    assert not out["kabul10b"]["pass"] and out["verdict"] == "fail"


def test_parity_kabul1_separate_from_10b():
    far = _reloc_case(2.5)
    out = compare(far, copy.deepcopy(far))
    assert out["kabul1"] == "pass"
    assert out["kabul10b"]["pass"] is False
    assert out["verdict"] == "fail"
    ref = {"scenarios": {"clean": _run([("moving", 0.5)] * 3)}}
    cpp = copy.deepcopy(ref)
    cpp["scenarios"]["clean"][1]["ego"]["keyframe"] = True
    assert compare(ref, cpp)["kabul1"] == "fail"
