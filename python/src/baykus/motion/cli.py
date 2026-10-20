"""`baykus-motion`: Faz 4 referans etiketleri ve keşif (docs/design/faz4.md §4, §5.3).

Komutlar `python/` içinden; yollar varsayılan olarak repo düzenine göre (`../data/m3ot_dev`).
Test dizisi okuyan komutlar `--i-know-this-is-test` ister (yalnız son ölçüm, tasarım §6.3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from baykus.engine_ref.cli import TEST_FLAG
from baykus.motion.explore import METHOD, explore_sequence
from baykus.motion.images import load_sequence
from baykus.motion.labels import gt_tracks, label_sequence, write_labels
from baykus.runinfo import git_info, iso, utc_now, write_json

K, LO, HI = 15, 0.03, 0.10  # tasarım §5.2 / §5.3 (ölçümden önce sabit)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cmd_labels(a: argparse.Namespace) -> int:
    seq = load_sequence(Path(a.seq_dir), a.i_know_this_is_test)  # test korumalı: önce
    rows = label_sequence(seq, gt_tracks(Path(a.tracks), seq.name), k=K, lo=LO, hi=HI)
    out = write_labels(Path(a.out), rows)
    counts = Counter(r.label for r in rows)
    meta = {
        "seq": seq.name,
        "split": seq.split,
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "k": K,
        "lo": LO,
        "hi": HI,
        "csv_sha256": _sha256(out),
        "tracks_sha256": _sha256(Path(a.tracks)),
        "labeler_commit": git_info()["git_commit"],
        "git_dirty": git_info()["git_dirty"],
        "created_at": iso(utc_now()),
    }
    write_json(out.with_suffix(".json"), meta)
    print(f"{seq.name}: {len(rows)} satır {dict(sorted(counts.items()))} → {out}")
    return 0


def _cmd_explore(a: argparse.Namespace) -> int:
    cfg = json.loads(Path(a.dev).read_text(encoding="utf-8"))
    data = Path(a.data)
    tracks = data / "labels" / "tracks_ir.csv"
    per_seq = {}
    for name in cfg["sequences"]:
        seq = load_sequence(data / "sequences" / name, allow_test=False)
        per_seq[name] = explore_sequence(seq, seq.rgb, gt_tracks(tracks, name))
        s = per_seq[name]
        print(
            f"{name}: IR adım ortanca {s['ir']['step_px']['median']:.2f} px, "
            f"zincir ölçek {s['ir']['chained_scale']:.2f}, v_hist {s['v_hist']}"
        )
    rec = {
        "split": cfg["split"],
        "sequences": cfg["sequences"],
        "method": METHOD,
        "per_seq": per_seq,
        "tracks_sha256": _sha256(tracks),
        **git_info(),
        "created_at": iso(utc_now()),
    }
    write_json(Path(a.out), rec)
    print(f"→ {a.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-motion")
    sub = p.add_subparsers(required=True)
    lab = sub.add_parser("labels", help="dizi + GT izleri → referans hareket etiketleri (CSV)")
    lab.add_argument("--seq-dir", required=True)
    lab.add_argument("--tracks", required=True, help="pack_labels tracks_ir.csv")
    lab.add_argument("--out", required=True)
    lab.add_argument(TEST_FLAG, action="store_true", dest="i_know_this_is_test")
    lab.set_defaults(func=_cmd_labels)
    ex = sub.add_parser("explore", help="tasarım §4 val keşfini yeniden üretir")
    ex.add_argument("--dev", default="../configs/m3ot_dev.json")
    ex.add_argument("--data", default="../data/m3ot_dev")
    ex.add_argument("--out", default="../results/motion/explore_val.json")
    ex.set_defaults(func=_cmd_explore)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, FileNotFoundError, OSError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 2
