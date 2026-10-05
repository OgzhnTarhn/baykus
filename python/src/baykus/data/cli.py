"""baykus-data: veri hattı alt komutları."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baykus.data.layout import SPLITS, split_dirs


def _cmd_verify(a: argparse.Namespace) -> int:
    from baykus.data.verify import verify_split, write_outputs

    ok, issues = {}, []
    for split in SPLITS:
        s_ok, s_issues = verify_split(split_dirs(Path(a.raw), split), split)
        ok[split], issues = s_ok, issues + s_issues
    write_outputs(Path(a.meta), ok, issues)
    print(json.dumps({k: len(v) for k, v in ok.items()}), f"excluded={len(issues)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-data")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="RGB/IR çift ve etiket bütünlüğü")
    v.add_argument("--raw", required=True)
    v.add_argument("--meta", required=True)
    v.set_defaults(func=_cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
