"""Calculate pass rates for quality.toml rubric files.

Detects unscored templates and incomplete rubrics, emitting clear warnings
instead of silently printing 0.0000. Use --strict to make these warnings
fatal (non-zero exit code).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ctpipe.toml_utils import safe_calc_passrate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate pass rate for quality.toml files.",
    )
    parser.add_argument("paths", nargs="+", help="TOML file or directory paths")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any file is unscored or incomplete",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.toml")))
        else:
            files.append(path)

    has_issue = False
    for path in files:
        rate, err = safe_calc_passrate(path)
        if rate is None:
            print(f"{path}: ERROR: {err}", file=sys.stderr)
            has_issue = True
            continue
        print(f"{path}: {rate:.4f}")

    if args.strict and has_issue:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
