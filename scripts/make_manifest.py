"""Model manifesti: configs/engine/models.json (Faz 3 tasarımı §4.6, plan F3 Görev 8).

Kullanım: python scripts/make_manifest.py <models_dir> <out>
Dört dosyanın sha256'sı ve bayt sayısı; CI ve kurulum Release'ten indirip bunlarla doğrular
(scripts/fetch_models.sh).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

RELEASE = "models-v1"
BASE_URL = f"https://github.com/OgzhnTarhn/baykus/releases/download/{RELEASE}/"
FILES = ("e3c.onnx", "thresholds.json", "calibration.json", "gate_model.json")


def manifest(models_dir: Path) -> dict:
    files = {}
    for name in FILES:
        p = Path(models_dir) / name
        if not p.is_file():
            raise FileNotFoundError(f"{p} yok")
        data = p.read_bytes()
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return {"release": RELEASE, "base_url": BASE_URL, "files": files}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(args[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest(Path(args[0])), indent=2, ensure_ascii=False) + "\n"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
