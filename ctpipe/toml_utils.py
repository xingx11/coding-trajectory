"""TOML reading and writing utilities.

Python's tomllib is read-only. For writing scored TOML files, we use
string templating since the structure is fixed (7 [[criterion]] blocks).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Criterion:
    name: str
    description: str
    type: str
    points: int
    weight: float
    score: int
    rationale: str


def read_quality_toml(path: Path) -> list[Criterion]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return [
        Criterion(
            name=c["name"],
            description=c["description"],
            type=c.get("type", "likert"),
            points=int(c.get("points", 5)),
            weight=float(c.get("weight", 1.0)),
            score=int(c.get("score", 0)),
            rationale=c.get("rationale", ""),
        )
        for c in data.get("criterion", [])
    ]


def _escape_toml_basic(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")


def _escape_toml_multiline(s: str) -> str:
    s = s.replace("\\", "\\\\")
    while '"""' in s:
        s = s.replace('"""', '""\\"')
    if s.endswith('"'):
        s = s[:-1] + '\\"'
    return s


def write_quality_toml(path: Path, criteria: list[Criterion]) -> None:
    parts: list[str] = []
    for c in criteria:
        parts.append(
            f'[[criterion]]\n'
            f'name = "{_escape_toml_basic(c.name)}"\n'
            f'description = "{_escape_toml_basic(c.description)}"\n'
            f'type = "{_escape_toml_basic(c.type)}"\n'
            f'points = {c.points}\n'
            f'weight = {c.weight}\n'
            f'score = {c.score}\n'
            f'rationale = """{_escape_toml_multiline(c.rationale)}"""\n'
        )
    path.write_text("\n".join(parts), encoding="utf-8")


def calc_passrate(criteria: list[Criterion]) -> float:
    total_score = sum(c.score * c.weight for c in criteria)
    total_points = sum(c.points * c.weight for c in criteria)
    if total_points <= 0:
        raise ValueError("Non-positive total points")
    return total_score / total_points


EXPECTED_CRITERIA_COUNT = 7


def is_unscored_template(criteria: list[Criterion]) -> bool:
    return all(c.score == 0 and not c.rationale for c in criteria)


def is_complete_rubric(criteria: list[Criterion]) -> bool:
    if len(criteria) != EXPECTED_CRITERIA_COUNT:
        return False
    scored_count = sum(1 for c in criteria if c.score > 0 or c.rationale)
    return scored_count == EXPECTED_CRITERIA_COUNT
