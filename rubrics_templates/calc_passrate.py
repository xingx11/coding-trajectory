"""Calculate pass rates for quality.toml rubric files.

Detects unscored templates and incomplete rubrics, emitting clear warnings
instead of silently printing 0.0000. Use --strict to make these warnings
fatal (non-zero exit code).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ctpipe.toml_utils import (
    calc_passrate,
    is_complete_rubric,
    is_unscored_template,
    read_quality_toml,
)


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
        try:
            criteria = read_quality_toml(path)
        except Exception as exc:
            print(f"{path}: ERROR: {exc}", file=sys.stderr)
            return 1

        if not criteria:
            print(f"{path}: ERROR: no [[criterion]] entries", file=sys.stderr)
            return 1

        if is_unscored_template(criteria):
            print(
                f"{path}: UNSCORED — all {len(criteria)} criteria have "
                f"score=0 and empty rationale",
                file=sys.stderr,
            )
            has_issue = True
            continue

        if not is_complete_rubric(criteria):
            scored_n = sum(1 for c in criteria if c.score >= 1 and c.rationale)
            print(
                f"{path}: INCOMPLETE — {scored_n}/{len(criteria)} criteria "
                f"fully scored",
                file=sys.stderr,
            )
            has_issue = True
            continue

        try:
            rate = calc_passrate(criteria)
        except Exception as exc:
            print(f"{path}: ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"{path}: {rate:.4f}")

    if args.strict and has_issue:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
