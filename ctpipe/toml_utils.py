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


def safe_calc_passrate(score_path: Path) -> tuple[float | None, str]:
    """Read a score file and compute passrate; return (passrate, error_reason).

    Returns (float, "") on success.
    Returns (None, reason) when the file is missing, unreadable, empty,
    unscored (all criteria have score=0 and empty rationale), or incomplete
    (not all criteria are fully scored).
    """
    if not score_path.exists():
        return None, "score file missing"
    try:
        criteria = read_quality_toml(score_path)
    except Exception as exc:
        return None, f"score file unreadable: {exc}"
    if not criteria:
        return None, "score file has no criteria"
    if is_unscored_template(criteria):
        return None, "score file is still an unscored template"
    if not is_complete_rubric(criteria):
        from ctpipe.config import MAX_CRITERIA_COUNT, MIN_CRITERIA_COUNT
        n = len(criteria)
        if not (MIN_CRITERIA_COUNT <= n <= MAX_CRITERIA_COUNT):
            return None, f"wrong criteria count: {n} (expected {MIN_CRITERIA_COUNT}-{MAX_CRITERIA_COUNT})"
        bad_scores = [c for c in criteria if not (1 <= c.score <= 5)]
        if bad_scores:
            return None, f"score {bad_scores[0].score} out of range 1-5 in '{bad_scores[0].name}'"
        missing_rationale = [c for c in criteria if not c.rationale]
        if missing_rationale:
            return None, f"missing rationale in '{missing_rationale[0].name}'"
        scored = sum(1 for c in criteria if 1 <= c.score <= 5 and c.rationale)
        return None, f"incomplete scoring: {scored}/{n} criteria scored"
    try:
        return calc_passrate(criteria), ""
    except Exception as exc:
        return None, f"passrate calculation error: {exc}"


def read_complete_score(score_path: Path) -> tuple[list[Criterion] | None, float | None, str]:
    """Read and validate a score file, returning validated criteria and passrate.

    Consolidates the repeated pattern found across score.py and health.py of:
      1. Read the TOML score file
      2. Skip if invalid, unscored, or incomplete
      3. Calculate the passrate

    Validation chain (each step short-circuits on failure):
      - File must exist and be parseable as TOML
      - Must contain at least one criterion
      - Must not be an unscored template (all score=0 with empty rationale)
      - Must pass ``is_complete_rubric()``: criterion count within
        MIN–MAX range, every score in 1–5, points=5, type="likert",
        and non-empty rationale
      - Every criterion name must be valid snake_case
        (``is_valid_criterion_name()``)

    Args:
        score_path: Path to the ``.quality.toml`` score file.

    Returns:
        ``(criteria, passrate, "")`` on success — *criteria* is the fully
        validated list of :class:`Criterion` objects and *passrate* is the
        weighted pass rate (0.0–1.0).

        ``(None, None, reason)`` on any validation failure — *reason* is a
        human-readable string such as ``"score file missing"``,
        ``"score 0 out of range 1-5 in 'delivery'"``, or
        ``"incomplete scoring: 5/7 criteria scored"``.
    """
    from ctpipe.config import is_valid_criterion_name

    if not score_path.exists():
        return None, None, "score file missing"
    try:
        criteria = read_quality_toml(score_path)
    except Exception as exc:
        return None, None, f"score file unreadable: {exc}"
    if not criteria:
        return None, None, "score file has no criteria"
    if is_unscored_template(criteria):
        return None, None, "score file is still an unscored template"
    if not is_complete_rubric(criteria):
        from ctpipe.config import MAX_CRITERIA_COUNT, MIN_CRITERIA_COUNT
        n = len(criteria)
        if not (MIN_CRITERIA_COUNT <= n <= MAX_CRITERIA_COUNT):
            return None, None, f"wrong criteria count: {n} (expected {MIN_CRITERIA_COUNT}-{MAX_CRITERIA_COUNT})"
        bad_scores = [c for c in criteria if not (1 <= c.score <= 5)]
        if bad_scores:
            return None, None, f"score {bad_scores[0].score} out of range 1-5 in '{bad_scores[0].name}'"
        missing_rationale = [c for c in criteria if not c.rationale]
        if missing_rationale:
            return None, None, f"missing rationale in '{missing_rationale[0].name}'"
        scored = sum(1 for c in criteria if 1 <= c.score <= 5 and c.rationale)
        return None, None, f"incomplete scoring: {scored}/{n} criteria scored"
    if not all(is_valid_criterion_name(c.name) for c in criteria):
        return None, None, "invalid criterion name"
    try:
        passrate = calc_passrate(criteria)
    except Exception as exc:
        return None, None, f"passrate calculation error: {exc}"
    return criteria, passrate, ""
