import gzip
import hashlib
import json
from pathlib import Path

import pytest

from baykus.eval.cli import load_coco
from baykus.health.gate import assess
from baykus.health.guard import (
    GateModel,
    download_list,
    guard_val_report,
    load_gate_model,
    select_r_min,
    splice_condition,
    write_gate_model,
)
from baykus.health.suite import HealthError
from tests.test_health_gate import OK, TH

QUIET = {**OK, "misalign_px": 3.0, "misalign_response": 0.9}


def _clean(alarm_responses, n=100):
    rows = [{**QUIET, "condition": "clean"} for _ in range(n - len(alarm_responses))]
    rows += [
        {**OK, "condition": "clean", "misalign_px": 30.0, "misalign_response": r}
        for r in alarm_responses
    ]
    return rows


def test_select_smallest_r_meeting_criterion():
    sel = select_r_min(_clean([0.05, 0.3, 0.6]), TH)
    # r=.05 → 3 alarm (150 > 100); r=.3 → 2 alarm (100 ≤ 100)
    assert sel["r_min"] == 0.3
    assert sel["criterion"] == "clean_alarm*50<=n"
    assert sel["n_clean"] == 100 and sel["train_clean_alarm_rate"] == pytest.approx(0.02)


def test_selected_r_matches_assess():
    rows = _clean([0.05, 0.3, 0.6])
    sel = select_r_min(rows, TH)
    alarms = sum(assess(r, TH, sel["r_min"])["gate"] != "none" for r in rows)
    assert alarms == round(sel["train_clean_alarm_rate"] * sel["n_clean"])


def test_select_zero_when_already_ok():
    assert select_r_min(_clean([0.05]), TH)["r_min"] == 0.0


def test_criterion_is_integer():
    assert select_r_min(_clean([0.05, 0.1]), TH)["r_min"] == 0.0  # 2/100 kabul
    assert select_r_min(_clean([0.05, 0.1, 0.2]), TH)["r_min"] == 0.1  # 3/100 değil


def test_non_misalignment_alarms_count():
    rows = _clean([0.6])
    rows[0] = {**rows[0], "ir_noise": 12.0}  # gürültü alarmı r_min'den bağımsız
    rows[1] = {**rows[1], "ir_noise": 12.0}
    sel = select_r_min(rows, TH)  # 2 gürültü + 1 hizasız: hizasızı kaldırmak gerekir
    assert sel["r_min"] == 0.9 and sel["train_clean_alarm_rate"] == pytest.approx(0.02)


def test_no_candidate_meets_criterion_is_error():
    rows = [{**QUIET, "condition": "clean", "ir_noise": 12.0}] * 3 + _clean([], n=97)
    with pytest.raises(ValueError, match="r_min"):
        select_r_min(rows, TH)


def test_tpr_split_dark_bright():
    # pozitif yanıtlar (.05, .3] dışında: aday olarak seçimi değiştirmesinler
    pos = [
        {**OK, "condition": c, "misalign_px": 20.0, "rgb_brightness": b, "misalign_response": r}
        for c, b, r in [
            ("misalign-20", 20.0, 0.04),
            ("misalign-40", 20.0, 0.5),
            ("misalign-20", 120.0, 0.5),
            ("misalign-40", 120.0, 0.5),
        ]
    ]
    pos.append({**OK, "condition": "noise-20", "misalign_px": 20.0, "misalign_response": 0.01})
    sel = select_r_min(_clean([0.05, 0.3, 0.6]) + pos, TH)
    assert sel["r_min"] == 0.3 and sel["n_pos"] == 4
    assert sel["train_tpr"] == {"all": 0.75, "dark": 0.5, "bright": 1.0}


def test_csv_strings_are_parsed():
    rows = [
        {k: ("" if v is None else str(v)) for k, v in r.items()} for r in _clean([0.05, 0.3, 0.6])
    ]
    assert select_r_min(rows, TH)["r_min"] == 0.3


def _p(img, score, box=(1.0, 2.0, 3.0, 4.0)):
    return {"image_id": img, "category_id": 1, "bbox": list(box), "score": score}


def _h(stem, reason="", response=0.5, changed=False):
    return {
        "stem": stem,
        "gate": "ir_off" if reason else "none",
        "reason": reason,
        "changed": changed,
        "pair": {"response": response},
    }


HEALTH = [
    _h("a", "IR hizasız", 0.1, True),
    _h("b", "IR hizasız", 0.5, True),
    _h("c"),
]
IDS = {"a": 1, "b": 2, "c": 3}
BASE = [_p(1, 0.9), _p(2, 0.8), _p(3, 0.7)]
GATE = [_p(1, 0.4, (9, 9, 9, 9)), _p(2, 0.3, (9, 9, 9, 9)), _p(3, 0.7)]


def test_splice_takes_base_only_for_removed():
    preds, info = splice_condition(HEALTH, GATE, BASE, IDS, 0.3, True)
    by = {p["image_id"]: p for p in preds}
    assert by[1] == BASE[0] and by[2] == GATE[1] and by[3] == GATE[2]
    assert len(preds) == 3
    assert info == {"removed": 1, "precondition_ok": True, "source": "spliced"}


def test_splice_unevaluated_uses_base():
    preds, info = splice_condition([_h("c")], None, BASE, IDS, 0.3, False)
    assert preds == BASE
    assert info == {"removed": 0, "precondition_ok": True, "source": "base"}


def test_splice_precondition_detects_mismatch():
    gate = [*GATE[:2], _p(3, 0.71)]  # kapının dokunmadığı karede farklı skor
    preds, info = splice_condition(HEALTH, gate, BASE, IDS, 0.3, True)
    assert info["precondition_ok"] is False and preds == []


def test_splice_precondition_detects_missing_box():
    preds, info = splice_condition(HEALTH, GATE[:2], BASE, IDS, 0.3, True)
    assert info["precondition_ok"] is False


def test_splice_unknown_stem_is_error():
    with pytest.raises(ValueError, match="zz"):
        splice_condition([_h("zz")], GATE, BASE, IDS, 0.3, True)


RUN = "2026-10-15_e3c-modality-dropout_s_d4e1280"


def test_download_list_skips_unevaluated(tmp_path: Path):
    h = tmp_path / "health"
    h.mkdir()
    for cid, ev in (("clean", True), ("drop_ir", False)):
        (h / f"{RUN}_val_{cid}_health.json").write_text(
            json.dumps({"evaluated": ev}), encoding="utf-8"
        )
    got = download_list(tmp_path, RUN)
    assert got == [
        f"{RUN}_val_clean/predictions_coco.json.gz",
        f"{RUN}_val_clean_gate/health.jsonl.gz",
        f"{RUN}_val_clean_gate/predictions_coco.json.gz",
        f"{RUN}_val_drop_ir/predictions_coco.json.gz",
        f"{RUN}_val_drop_ir_gate/health.jsonl.gz",
    ]


def test_gate_model_roundtrip_and_sha(tmp_path: Path):
    csv_path = tmp_path / "f.csv"
    csv_path.write_text("stem,condition\n", encoding="utf-8")
    sel = select_r_min(_clean([0.05, 0.3, 0.6]), TH)
    out = write_gate_model(tmp_path / "g.json", sel, TH, csv_path)
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["temporal"] == {"window": 5, "min_frames": 3, "tol_px": 3.0}
    assert raw["thresholds_sha256"] == TH.sha256
    assert raw["train_features_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    gm = load_gate_model(out)
    assert (gm.r_min, gm.window, gm.min_frames, gm.tol_px) == (0.3, 5, 3, 3.0)
    assert gm.sha256 == hashlib.sha256(out.read_bytes()).hexdigest()
    assert gm.thresholds_sha256 == TH.sha256


def test_load_gate_model_missing_key_is_error(tmp_path: Path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"r_min": 0.2}), encoding="utf-8")
    with pytest.raises(ValueError, match="temporal"):
        load_gate_model(p)


def _write_val(tmp_path: Path) -> tuple[Path, Path, Path]:
    """İki karelik val: clean + HARMFUL koşulları; 'a' düşük yanıtlı hizasız alarm."""
    gt = tmp_path / "val_ir.json"
    imgs = [{"id": i, "file_name": s, "width": 64, "height": 64} for s, i in IDS.items()]
    anns = [
        {"id": i, "image_id": i, "category_id": 1, "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0}
        for i in IDS.values()
    ]
    cats = [{"id": 1, "name": "car"}]
    gt.write_text(json.dumps({"images": imgs, "annotations": anns, "categories": cats}), "utf-8")
    results, robust = tmp_path / "results", tmp_path / "robust"
    (results / "health").mkdir(parents=True)
    for cid in ("clean", "noise-20", "misalign-20", "misalign-40"):
        (results / "health" / f"{RUN}_val_{cid}_health.json").write_text(
            json.dumps(
                {
                    "thresholds_sha256": TH.sha256,
                    "evaluated": True,
                    "n_images": 3,
                    "changed": 2,
                    "gate": {"none": 1, "ir_off": 2, "rgb_off": 0},
                }
            ),
            encoding="utf-8",
        )
        base = robust / f"{RUN}_val_{cid}"
        gate = robust / f"{RUN}_val_{cid}_gate"
        for d, preds in ((base, BASE), (gate, GATE)):
            d.mkdir(parents=True)
            (d / "predictions_coco.json.gz").write_bytes(gzip.compress(json.dumps(preds).encode()))
        frames = [
            {
                **h,
                "pair": {**h["pair"], "misaligned": bool(h["reason"])},
                "rgb": {"brightness": 100},
            }
            for h in HEALTH
        ]
        (gate / "health.jsonl.gz").write_bytes(
            gzip.compress("".join(json.dumps(f) + "\n" for f in frames).encode())
        )
    return gt, results, robust


def test_guard_val_report_splices_and_accepts(tmp_path: Path):
    gt_path, results, robust = _write_val(tmp_path)
    gm = GateModel(0.3, 5, 3, 3.0, "g", TH.sha256)
    rep = guard_val_report(
        run_id=RUN, robust_preds=robust, gt=load_coco(gt_path), results=results,
        gate_model=gm, th=TH, n_boot=0,
    )  # fmt: skip
    clean = next(r for r in rep["rows"] if r["condition"] == "clean")
    assert clean["source"] == "spliced" and clean["removed"] == 1
    assert clean["gate_rate"] == pytest.approx(1 / 3)  # 2 alarm − 1 kaldırılan
    assert rep["label"] == "val keşfi, PyTorch E3c" and rep["r_min"] == 0.3
    assert rep["harmful"]["conditions"] and rep["acceptance"] is not None
    assert rep["misaligned_catch"]["misalign-20"]["all"] == pytest.approx(1 / 3)
    json.dumps(rep, allow_nan=False)


def test_guard_val_report_rejects_other_thresholds(tmp_path: Path):
    gt_path, results, robust = _write_val(tmp_path)
    gm = GateModel(0.3, 5, 3, 3.0, "g", "baska")
    with pytest.raises(HealthError):
        guard_val_report(
            run_id=RUN, robust_preds=robust, gt=load_coco(gt_path), results=results,
            gate_model=gm, th=TH, n_boot=0,
        )  # fmt: skip
