"""Shared test helpers for ctpipe tests."""

from __future__ import annotations

import json
from pathlib import Path

from ctpipe.config import (
    BatchConfig,
    ModelConfig,
    REFERENCE_CRITERION_DESCRIPTIONS,
    REFERENCE_CRITERION_NAMES,
    TaskConfig,
)
from ctpipe.toml_utils import Criterion, write_quality_toml

# Fake paths used as constructor arguments (not accessed on disk).
FAKE_RUNS_ROOT = Path("D:/runs")
FAKE_PROJECT_PATH = Path("D:/projects/demo")


def build_config(
    tasks: list[TaskConfig] | None = None,
    person_id: str = "99",
    delivery_date: str = "20990101",
) -> BatchConfig:
    """Build a minimal BatchConfig for testing."""
    return BatchConfig(
        delivery_date=delivery_date,
        runs_root=FAKE_RUNS_ROOT,
        max_parallel=2,
        tasks=tasks or [],
        qwen=ModelConfig(auth_token="", base_url="", model="qwen-test"),
        claude=ModelConfig(auth_token="", base_url="", model="claude-test"),
        person_id=person_id,
    )


def make_task(
    task_id: str = "CT-0001",
    task_type: str = "bug-fix",
    domain: str = "web_frontend",
    language: str = "ts",
    bad_pattern: str = "",
    followups_qwen: list[str] | None = None,
    followups_claude: list[str] | None = None,
) -> TaskConfig:
    """Build a minimal TaskConfig for testing."""
    return TaskConfig(
        id=task_id,
        project_path=FAKE_PROJECT_PATH,
        clone_method="git",
        task_type=task_type,
        domain=domain,
        language=language,
        prompt="test prompt",
        bad_pattern=bad_pattern,
        followups_qwen=followups_qwen or [],
        followups_claude=followups_claude or [],
    )


def write_trajectory(
    jsonl_path: Path, session_id: str, model_name: str, user_turns: int = 3
) -> None:
    """Write a minimal valid trajectory JSONL (>= 10 lines, valid session/model)."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"sessionId": session_id, "type": "system"})]
    for i in range(user_turns):
        lines.append(
            json.dumps({"type": "user", "message": {"role": "user", "content": f"msg {i}"}})
        )
        lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "model": f"{model_name}-model-v1",
                        "content": f"reply {i}",
                    },
                }
            )
        )
    while len(lines) < 12:
        lines.append(json.dumps({"type": "system", "info": "padding"}))
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_score(score_path: Path, score_per_criterion: int = 3) -> None:
    """Write a complete score TOML with valid criterion names."""
    score_path.parent.mkdir(parents=True, exist_ok=True)
    names = REFERENCE_CRITERION_NAMES[:7]
    criteria = [
        Criterion(
            name, REFERENCE_CRITERION_DESCRIPTIONS[name], "likert", 5,
            1.0, score_per_criterion, "评分理由"
        )
        for name in names
    ]
    write_quality_toml(score_path, criteria)
