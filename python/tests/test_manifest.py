import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_manifest.py"


def _module():
    spec = importlib.util.spec_from_file_location("make_manifest", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _models(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    names = ("e3c.onnx", "thresholds.json", "calibration.json", "gate_model.json")
    for i, name in enumerate(names):
        (d / name).write_bytes(bytes([i]) * (i + 3))
    return d


def test_manifest_matches_files(tmp_path: Path):
    d = _models(tmp_path)
    out = tmp_path / "models.json"
    _module().main([str(d), str(out)])
    m = json.loads(out.read_text(encoding="utf-8"))
    assert m["release"] == "models-v1"
    assert m["base_url"].endswith("/releases/download/models-v1/")
    assert sorted(m["files"]) == sorted(p.name for p in d.iterdir())
    for name, rec in m["files"].items():
        data = (d / name).read_bytes()
        assert rec == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def test_manifest_missing_file_is_error(tmp_path: Path):
    d = _models(tmp_path)
    (d / "gate_model.json").unlink()
    with pytest.raises(FileNotFoundError, match="gate_model.json"):
        _module().main([str(d), str(tmp_path / "models.json")])
