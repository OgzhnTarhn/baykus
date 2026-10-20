import json
from pathlib import Path

from baykus.system.report import EVIDENCE, acceptance, exit_ok, load_results, system_markdown


def _write(root: Path, rel: str, rec: dict) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec), encoding="utf-8")


def _stages(p95: float) -> dict:
    names = ("decode", "health", "pre", "infer", "post", "track", "total")
    return {n: {"p50": p95 / 2, "p95": p95} for n in names}


def _passing(root: Path) -> Path:
    frame = {"letterbox_sha_equal": True, "raw": {"max_diff": 0.0}, "ok": True}
    _write(root, EVIDENCE["k1"], {"passed": True, "n_frames": 17, "frames": [frame] * 17})
    _write(
        root,
        EVIDENCE["k3"],
        {
            "passed": True,
            "single_frames": {"equal": 17, "frames": 17, "max_px": 0.0, "max_rel": 0.0},
            "clip": {"passed": True},
            "temporal_unit": {"passed": True},
        },
    )
    seq = {"frames": 600, "compared": 600, "first_diff": None, "ratio": 1.0}
    _write(root, EVIDENCE["k4"], {"verdict": "pass", "per_seq": {"1_08": seq, "2_08": seq}})
    _write(
        root,
        EVIDENCE["k5"],
        {
            "kabul5": {"passed": True, "total_p95_ms": 82.19, "threshold_ms": 100.0},
            "stages": _stages(82.19),
            "peak_rss_mb": 287.1,
            "fps": 14.8,
        },
    )
    _write(
        root,
        EVIDENCE["k6"],
        {"passed": True, "browser_requests_external": 0, "scan": {"violations": []}},
    )
    _write(
        root,
        EVIDENCE["k7"],
        {"wifi_off": True, "items": [{"id": "FR-U1 oynat", "file": "a.jpg", "ok": True}]},
    )
    _write(
        root,
        EVIDENCE["k8"],
        {
            "cases": [{"name": "missing_part", "ok": True}] * 15,
            "fuzz": {"n": 1000, "healthy_after": True, "server_errors": 0},
        },
    )
    det = {"ap50": 0.8612, "ece_raw": 0.361, "ece_calibrated": 0.347, "duplicate_rate": 0.012}
    _write(root, "engine/m3ot_test_detect.json", {**det, "n_images": 960, "per_seq": {}})
    trk = {"HOTA": 0.4892, "MOTA": 0.5469, "IDF1": 0.6774}
    _write(root, "track/m3ot_test_engine.json", {"per_seq": {"1_03": trk}, "combined": trk})
    return root


def test_report_renders_acceptance_table_from_jsons(tmp_path):
    res = load_results(_passing(tmp_path))
    rows = acceptance(res)
    assert [r["kabul"] for r in rows] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert all(r["ok"] for r in rows) and exit_ok(rows)
    md = system_markdown(res)
    assert "| 5 |" in md and "82.2 ms" in md and "✅" in md and "❌" not in md
    assert "0.861" in md and "0.489" in md  # M3OT test AP50 ve HOTA JSON'dan
    assert "—" in md  # eksik isteğe bağlı sayı (ör. val provası) tire


def test_missing_evidence_file_is_fail_not_pass(tmp_path):
    root = _passing(tmp_path)
    (root / EVIDENCE["k8"]).unlink()
    res = load_results(root)
    rows = acceptance(res)
    k8 = next(r for r in rows if r["kabul"] == 8)
    assert k8["ok"] is False and "kanıt yok" in k8["detail"]
    assert not exit_ok(rows)
    assert "kanıt yok" in system_markdown(res)


def test_failing_value_is_fail(tmp_path):
    root = _passing(tmp_path)
    _write(root, EVIDENCE["k7"], {"wifi_off": None, "items": [{"id": "x", "ok": True}]})
    rows = acceptance(load_results(root))
    assert next(r for r in rows if r["kabul"] == 6)["ok"] is False  # Wi-Fi kapalı demo yok
    assert next(r for r in rows if r["kabul"] == 7)["ok"] is True


def test_exploration_label_present(tmp_path):
    md = system_markdown(load_results(_passing(tmp_path)))
    assert "val'de keşif, PyTorch E3c" in md


def test_report_protocol_from_json(tmp_path):
    """K7: bench protokolü ve sürümler kanıt dosyasından; sabit metin değil."""
    root = _passing(tmp_path)
    k5 = json.loads((root / EVIDENCE["k5"]).read_text(encoding="utf-8"))
    k5["protocol"] = {"warmup": 7, "runs": 33, "images": 4, "gate": "auto", "track": False}
    k5["system"] = {"onnxruntime": "9.8.7", "opencv": "6.5.4"}
    _write(root, EVIDENCE["k5"], k5)
    md = system_markdown(load_results(root))
    assert "7 ısınma + 33 ölçüm, 4 görüntü, `gate=auto`, takip kapalı" in md
    assert "50 ısınma" not in md
    assert "ONNX Runtime 9.8.7 + OpenCV 6.5.4" in md


def test_report_protocol_sequence_inputs(tmp_path):
    root = _passing(tmp_path)
    k5 = json.loads((root / EVIDENCE["k5"]).read_text(encoding="utf-8"))
    k5["protocol"] = {
        "warmup": 50,
        "runs": 500,
        "inputs": "sequence",
        "seq": "1_08",
        "gate": "advise",
        "track": True,
    }
    _write(root, EVIDENCE["k5"], k5)
    md = system_markdown(load_results(root))
    assert "50 ısınma + 500 ölçüm, `1_08` dizisinin ardışık kareleri" in md


def test_report_protocol_missing_is_dash(tmp_path):
    md = system_markdown(load_results(_passing(tmp_path)))
    assert "Protokol —" in md
