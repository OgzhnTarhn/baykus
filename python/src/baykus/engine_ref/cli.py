"""baykus-engine: C++ motorunun Python referansı.

detect — bir dizi klasöründen (operatör ekranı düzeni: rgb/, ir/, index.json) tespit dosyası
         (değerlendirme katmanı, varsayılan ham skor ≥ 0,001). Test dizisi bayraksız okunmaz.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path

from baykus.edge.backends import OrtBackend
from baykus.engine_ref.models import load_models
from baykus.engine_ref.pipeline import GATE_MODES, Pipeline

TEST_FLAG = "--i-know-this-is-test"


def read_index(seq_dir: Path, allow_test: bool) -> dict:
    index = json.loads((Path(seq_dir) / "index.json").read_text(encoding="utf-8"))
    if index.get("split") == "test" and not allow_test:
        raise ValueError(f"{seq_dir}: test dizisi; yalnız son ölçümde {TEST_FLAG} ile okunur")
    return index


def _cmd_detect(a: argparse.Namespace) -> int:
    seq = Path(a.seq_dir)
    index = read_index(seq, a.i_know_this_is_test)  # modellerden önce: test hiç yüklenmesin
    models = load_models(Path(a.models), Path(a.manifest) if a.manifest else None)
    pipe = Pipeline(models, backend=OrtBackend(models.onnx))
    frames = []
    for f in index["frames"]:
        rgb = (seq / "rgb" / f"{f:06d}.jpg").read_bytes()
        ir = (seq / "ir" / f"{f:06d}.png").read_bytes()
        r = pipe.run(rgb, ir, gate=a.gate, floor=a.floor)
        frames.append({"frame": f, "dets": [asdict(d) for d in r["dets_all"]]})
    rec = {
        "seq": index["name"],
        "split": index["split"],
        "floor": a.floor,
        "gate": a.gate,
        "frames": frames,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.compress(json.dumps(rec, ensure_ascii=False).encode("utf-8")))
    print(f"{out} ({len(frames)} kare)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-engine")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="dizi klasörü → tespit dosyası (.json.gz)")
    d.add_argument("--models", required=True, help="model dizini (e3c.onnx + JSON'lar)")
    d.add_argument("--manifest", default=None, help="configs/engine/models.json (sha256 denetimi)")
    d.add_argument("--seq-dir", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--floor", type=float, default=0.001, help="ham skor taban süzgeci")
    d.add_argument("--gate", default="advise", choices=GATE_MODES)
    d.add_argument(TEST_FLAG, action="store_true", help="test dizisi (yalnız son ölçüm)")
    d.set_defaults(func=_cmd_detect)
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows konsolu (cp1254) → yazamaz: kodlanamayan karakter '?' olur, komut çökmez
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, FileNotFoundError, OSError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 2
