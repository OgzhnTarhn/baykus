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


def _cmd_convert(a: argparse.Namespace) -> int:
    from baykus.data.convert import convert_split

    ok = json.loads((Path(a.meta) / "ok_stems.json").read_text(encoding="utf-8"))
    stats = {
        s: convert_split(split_dirs(Path(a.raw), s), s, ok.get(s, []), Path(a.out)) for s in SPLITS
    }
    (Path(a.meta) / "convert_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))
    return 0


def _cmd_activate(a: argparse.Namespace) -> int:
    from baykus.data.convert import activate_label_set

    print(activate_label_set(Path(a.out), a.modality, a.label_set))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-data")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="RGB/IR çift ve etiket bütünlüğü")
    v.add_argument("--raw", required=True)
    v.add_argument("--meta", required=True)
    v.set_defaults(func=_cmd_verify)
    c = sub.add_parser("convert", help="kenarlık kırp, etiketleri YOLO formatına çevir")
    c.add_argument("--raw", required=True)
    c.add_argument("--meta", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_convert)
    al = sub.add_parser(
        "activate-labels", help="bir etiket setini <modality>/labels altına kopyala"
    )
    al.add_argument("--out", required=True)
    al.add_argument("--modality", required=True, choices=["rgb", "ir", "rgbt4"])
    al.add_argument("--label-set", required=True, choices=["ir", "rgb"])
    al.set_defaults(func=_cmd_activate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
