"""C++ motorundaki ve Python referansındaki OpenCV derleme bilgisi (Faz 3 tasarımı §4.1).

Kullanım (python/ içinden): uv run python ../scripts/opencv_build_info.py
    [--image baykus-engine-build]
Çıktı: results/engine/opencv_build.json = {"cpp": {...}, "python": {...}, "compare": {...}}.
`compare` JPEG/PNG/zlib sürümlerini, IPP'yi ve CPU taban/dağıtım satırlarını yan yana koyar:
JPEG çözümünün bayt bayt aynı olması (Kabul 2) bu kütüphanelerin aynı olmasına bağlı.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import onnxruntime

KEYS = ("ZLib", "JPEG", "PNG", "Intel IPP", "Baseline", "Dispatched code generation")
OUT = Path(__file__).resolve().parents[1] / "results" / "engine" / "opencv_build.json"


def _line(info: str, key: str) -> str | None:
    m = re.search(rf"^\s*{re.escape(key)}:\s*(.+)$", info, re.MULTILINE)
    return m.group(1).strip() if m else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="baykus-engine-build")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args(argv)
    run = subprocess.run(
        ["docker", "run", "--rm", a.image, "/build/bin/baykus_buildinfo"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    cpp = json.loads(run.stdout)
    py_info = cv2.getBuildInformation()
    rec = {
        "cpp": cpp,
        "python": {
            "opencv_version": cv2.__version__,
            "onnxruntime_version": onnxruntime.__version__,
            "opencv_build_information": py_info,
        },
        "compare": {
            k: {"cpp": _line(cpp["opencv_build_information"], k), "python": _line(py_info, k)}
            for k in KEYS
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n")
    print(out)
    for k, v in rec["compare"].items():
        print(f"{k:28s} C++ {v['cpp']!s:40s} Python {v['python']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
