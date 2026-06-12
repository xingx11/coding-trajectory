"""Tests for ctpipe.health.health_check() — the three-way overall_status verdict.

Verdict rules under test (see ctpipe/health.py):
  * critical  — any permanently_failed entry exists in pipeline state
  * degraded  — threshold violations or integrity issues (and no permanent failures)
  * healthy   — none of the above
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctpipe.config import BatchConfig, ModelConfig, TaskConfig
from ctpipe.health import health_check
from ctpipe.state import PipelineState


def _model() -> ModelConfig:
    return ModelConfig(auth_token="t", base_url="https://example.invalid", model="m")


def _task(task_id: str = "T1") -> TaskConfig:
    return TaskConfig(
        id=task_id,
        project_path=Path("proj"),
        clone_method="git",
        task_type="feature",
        domain="d",
        language="py",
        prompt_qwen="q",
        prompt_claude="c",
    )


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BatchConfig:
    # base_dir is a hardcoded property pointing at the real repo; redirect it to
    # an isolated tmp dir so the test never touches the real delivery tree.
    monkeypatch.setattr(BatchConfig, "base_dir", property(lambda self: tmp_path))
    cfg = BatchConfig(
        delivery_date="20990101",
        runs_root=tmp_path,
        max_parallel=1,
        tasks=[_task()],
        qwen=_model(),
        claude=_model(),
    )
    cfg.delivery_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _state(config: BatchConfig) -> PipelineState:
    return PipelineState(config.state_path)


def test_returns_four_top_level_fields(config: BatchConfig) -> None:
    result = health_check(config)
    for key in ("overall_status", "stage_summary", "threshold_violations", "integrity_issues"):
        assert key in result


def test_healthy_when_clean(config: BatchConfig) -> None:
    # No state at all: no permanent failures, no passrates, no 'done' stages.
    result = health_check(config)
    assert result["overall_status"] == "healthy"
    assert result["permanently_failed"] == []
    assert result["threshold_violations"] == []
    assert result["integrity_issues"] == []


def test_critical_on_permanently_failed(config: BatchConfig) -> None:
    state = _state(config)
    state.set("T1", "run", "qwen", status="permanently_failed")
    state.save()
    result = health_check(config)
    assert result["overall_status"] == "critical"
    assert any(e["task_id"] == "T1" for e in result["permanently_failed"])


def test_degraded_on_threshold_violation(config: BatchConfig) -> None:
    # qwen passrate >= THRESHOLD_QWEN_MAX (0.7) is a threshold violation.
    state = _state(config)
    state.set("T1", "finalize", qwen_passrate=0.85)
    state.save()
    result = health_check(config)
    assert result["overall_status"] == "degraded"
    assert result["permanently_failed"] == []
    assert result["threshold_violations"]


def test_degraded_on_integrity_issue(config: BatchConfig) -> None:
    # run marked 'done' but no trajectory file on disk -> integrity issue.
    state = _state(config)
    state.set("T1", "run", "qwen", status="done")
    state.save()
    result = health_check(config)
    assert result["overall_status"] == "degraded"
    assert result["permanently_failed"] == []
    assert result["integrity_issues"]


def test_critical_overrides_degraded(config: BatchConfig) -> None:
    # A permanent failure AND a threshold violation -> critical takes priority.
    state = _state(config)
    state.set("T1", "run", "qwen", status="permanently_failed")
    state.set("T1", "finalize", qwen_passrate=0.85)
    state.save()
    result = health_check(config)
    assert result["overall_status"] == "critical"
