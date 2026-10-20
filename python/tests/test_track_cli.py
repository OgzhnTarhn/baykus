import gzip
import json
from pathlib import Path

from baykus.track.cli import main


def _write(path: Path, ids_per_frame: list[list[tuple[int, int]]]) -> Path:
    seq = {
        "seq": "1_08",
        "split": "val",
        "frames": [
            {"frame": f + 1, "tracks": [{"index": i, "track_id": t} for i, t in fr]}
            for f, fr in enumerate(ids_per_frame)
        ],
    }
    path.write_bytes(gzip.compress(json.dumps({"sequences": [seq]}).encode("utf-8")))
    return path


def test_compare_exit_code_on_fail(tmp_path: Path):
    """S6: karar `fail` ise çıkış kodu ≠ 0 (CI'daki kanıt adımı başarısızlığı gizlemesin)."""
    py = _write(tmp_path / "py.json.gz", [[(0, 1)], [(0, 1)]])
    cpp = _write(tmp_path / "cpp.json.gz", [[(0, 5)], [(0, 6)]])
    out = tmp_path / "parity.json"
    code = main(["compare", "--py", str(py), "--cpp", str(cpp), "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "fail"
    assert code == 1


def test_compare_exit_code_on_pass(tmp_path: Path):
    py = _write(tmp_path / "py.json.gz", [[(0, 1)], [(0, 1)]])
    cpp = _write(tmp_path / "cpp.json.gz", [[(0, 4)], [(0, 4)]])
    out = tmp_path / "parity.json"
    code = main(["compare", "--py", str(py), "--cpp", str(cpp), "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "pass"
    assert code == 0
