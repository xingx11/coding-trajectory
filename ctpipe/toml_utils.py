"""TOML reading and writing utilities.

Python's tomllib is read-only. For writing scored TOML files, we use
string templating since the structure is a list of [[criterion]] blocks.
"""

from __future__ import annotations

import os
import re
import tempfile
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


def escape_toml_basic(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def escape_toml_multiline(s: str) -> str:
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
            f'name = "{escape_toml_basic(c.name)}"\n'
            f'description = "{escape_toml_basic(c.description)}"\n'
            f'type = "{escape_toml_basic(c.type)}"\n'
            f'points = {c.points}\n'
            f'weight = {c.weight}\n'
            f'score = {c.score}\n'
            f'rationale = "{escape_toml_basic(c.rationale)}"\n'
        )
    content = "\n".join(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        closed = True
        os.replace(tmp, path)
    except BaseException:
        if not closed:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_rubric_pair(rubrics_dir: Path, task_id: str, criteria: list[Criterion]) -> None:
    """Write rubric TOML templates for both qwen and claude."""
    for model in ("qwen", "claude"):
        tpl_dir = rubrics_dir / model
        tpl_dir.mkdir(parents=True, exist_ok=True)
        write_quality_toml(tpl_dir / f"{task_id}.quality.toml", criteria)


def calc_passrate(criteria: list[Criterion]) -> float:
    total_score = sum(c.score * c.weight for c in criteria)
    total_points = sum(c.points * c.weight for c in criteria)
    if total_points <= 0:
        raise ValueError("Non-positive total points")
    return total_score / total_points


def is_unscored_template(criteria: list[Criterion]) -> bool:
    return all(c.score == 0 and not c.rationale for c in criteria)


def is_complete_rubric(criteria: list[Criterion]) -> bool:
    from ctpipe.config import MAX_CRITERIA_COUNT, MIN_CRITERIA_COUNT

    if not (MIN_CRITERIA_COUNT <= len(criteria) <= MAX_CRITERIA_COUNT):
        return False
    return all(
        1 <= c.score <= 5
        and c.points == 5
        and c.type == "likert"
        and c.rationale
        for c in criteria
    )


def has_custom_descriptions(criteria: list[Criterion]) -> bool:
    """Check if criteria have customized (non-template) descriptions.

    Returns True if all criteria descriptions contain proper 1-5 score
    tier definitions, indicating they have been customized.
    """
    return all(has_score_tiers(c.description) for c in criteria)


def has_score_tiers(description: str) -> bool:
    """Check if a description contains all 5 score tier definitions (1-5分).

    Each tier must have a marker like "1分：" / "1分:" and be followed by
    at least 5 characters of substantive definition before the next tier
    marker or end of string.
    """
    tier_pattern = re.compile(r"(?<!\d)([1-5])\s*分\s*[：:]")
    matches = list(tier_pattern.finditer(description))
    found_tiers = {int(m.group(1)) for m in matches}
    if found_tiers != {1, 2, 3, 4, 5}:
        return False

    # Verify each tier has substantive content after the marker
    for m in matches:
        tier_end = m.end()
        # Find next tier marker or end of string
        next_match = tier_pattern.search(description, tier_end)
        segment_end = next_match.start() if next_match else len(description)
        content = description[tier_end:segment_end].strip().rstrip("。.，,")
        if len(content) < 5:
            return False
    return True
