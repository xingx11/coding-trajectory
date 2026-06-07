from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


def calc_passrate(path: Path) -> float:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    criteria = data.get("criterion", [])
    if not criteria:
        raise ValueError(f"{path} has no [[criterion]] entries")

    total_score = 0.0
    total_points = 0.0
    for item in criteria:
        score = float(item.get("score", 0))
        points = float(item.get("points", 5))
        weight = float(item.get("weight", 1))
        total_score += score * weight
        total_points += points * weight

    if total_points <= 0:
        raise ValueError(f"{path} has non-positive total points")
    return total_score / total_points


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate pass rate for quality.toml files.")
    parser.add_argument("paths", nargs="+", help="TOML file or directory paths")
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.toml")))
        else:
            files.append(path)

    for path in files:
        try:
            rate = calc_passrate(path)
        except Exception as exc:
            print(f"{path}: ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"{path}: {rate:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
