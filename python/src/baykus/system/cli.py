"""baykus-system: Faz 3 raporu.

report — results/**/*.json → docs/system.md (Kabul 1-8 kanıt dosyalarından;
         çıkış kodu 1 = faz çıkışı yok).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baykus.system.report import acceptance, exit_ok, load_results, system_markdown


def _cmd_report(a: argparse.Namespace) -> int:
    res = load_results(Path(a.results))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(system_markdown(res), encoding="utf-8", newline="\n")
    rows = acceptance(res)
    for r in rows:
        print(f"Kabul {r['kabul']}: {'geçti' if r['ok'] else 'DÜŞTÜ'} — {r['detail']}")
    print(out)
    return 0 if exit_ok(rows) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baykus-system")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="results/ → docs/system.md")
    r.add_argument("--results", default="../results")
    r.add_argument("--out", default="../docs/system.md")
    r.set_defaults(func=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows konsolu (cp1254) → yazamaz: kodlanamayan karakter '?' olur, komut çökmez
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, OSError) as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
