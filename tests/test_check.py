"""Regression tests for check.py CSV cross-validation.

Verifies that check() uses mapped submission IDs (from _assign_submission_ids)
when looking up CSV rows, so that session_id and passrate cross-validation
actually fires against formatted IDs like "56-608-bug-fix-01".
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import PropertyMock, patch

from ctpipe.check import _check_model_identity, _count_turns, check
from ctpipe.config import (
    REFERENCE_CRITERION_DESCRIPTIONS,
    SUBMISSION_FIELDNAMES,
    REFERENCE_CRITERION_NAMES,
    BatchConfig,
    ModelConfig,
    TaskConfig,
    model_stem,
    write_task_manifest,
)
from ctpipe.toml_utils import Criterion, write_quality_toml
from conftest import build_config, make_task, write_trajectory, write_score


def _write_score_custom(score_path: Path, criteria: list[Criterion]) -> None:
    """Write a score TOML with exactly the given criteria list."""
    score_path.parent.mkdir(parents=True, exist_ok=True)
    write_quality_toml(score_path, criteria)


def _write_submission_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _run_check(config: BatchConfig) -> tuple[bool, str]:
    """Run check() and capture stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = check(config)
    return result, buf.getvalue()


def _make_csv_row(
    sub_id: str,
    task_id: str,
    qwen_session: str = "qw-real-session",
    claude_session: str = "cl-real-session",
    qwen_passrate: str = "0.6000",
    claude_passrate: str = "0.8000",
) -> dict[str, str]:
    return {
        "id": sub_id,
        "qwen 本地trajectory": f"trajectories/qwen/{model_stem(task_id, 'qwen')}.jsonl",
        "qwen session id": qwen_session,
        "qwen rubrics 人工评分": f"scores/qwen/{model_stem(task_id, 'qwen')}.quality.toml",
        "claude 本地trajectory": f"trajectories/claude/{model_stem(task_id, 'claude')}.jsonl",
        "claude session id": claude_session,
        "claude rubrics 人工评分": f"scores/claude/{model_stem(task_id, 'claude')}.quality.toml",
        "qwen passrate": qwen_passrate,
        "claude passrate": claude_passrate,
        "任务类型": "bug-fix",
        "应用领域": "web_frontend",
        "编程语言": "ts",
        "命中QwenBad Pattern": "",
    }


def _setup_delivery(
    config: BatchConfig,
    tasks: list[TaskConfig],
    csv_rows: list[dict[str, str]],
    qwen_session: str = "qw-real-session",
    claude_session: str = "cl-real-session",
    qwen_score: int = 3,
    claude_score: int = 4,
) -> None:
    """Create a complete delivery directory with trajectories, scores, metadata, CSV."""
    delivery_dir = config.delivery_dir
    delivery_dir.mkdir(parents=True, exist_ok=True)
    write_task_manifest(config.task_manifest_path, tasks)

    for task in tasks:
        for model_name, session, score in [
            ("qwen", qwen_session, qwen_score),
            ("claude", claude_session, claude_score),
        ]:
            traj_path = delivery_dir / "trajectories" / model_name / f"{model_stem(task.id, model_name)}.jsonl"
            write_trajectory(traj_path, session, model_name)
            score_path = delivery_dir / "scores" / model_name / f"{model_stem(task.id, model_name)}.quality.toml"
            write_score(score_path, score)

        meta_dir = delivery_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / f"{task.id}.md").write_text(f"# Task {task.id}\n", encoding="utf-8")

    _write_submission_csv(delivery_dir / "submission.csv", csv_rows)


# =========================================================================
# Regression: check() must use mapped submission IDs to find CSV rows.
# =========================================================================


class CheckCSVCrossValidationTest(unittest.TestCase):
    """check() must look up CSV rows by mapped submission ID, not task.id."""

    def test_session_id_mismatch_detected_with_formatted_ids(self) -> None:
        """CSV has formatted ID '56-608-bug-fix-01' with wrong session → must detect."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                csv_rows = [
                    _make_csv_row(sub_id, task.id, qwen_session="WRONG-SESSION")
                ]
                _setup_delivery(config, [task], csv_rows)
                result, output = _run_check(config)

        self.assertFalse(result, "check should fail on session_id mismatch")
        self.assertIn("session_id mismatch", output)

    def test_passrate_mismatch_detected_with_formatted_ids(self) -> None:
        """CSV has formatted ID with wrong passrate → must detect."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                csv_rows = [
                    _make_csv_row(sub_id, task.id, qwen_passrate="0.9999")
                ]
                _setup_delivery(config, [task], csv_rows)
                result, output = _run_check(config)

        self.assertFalse(result, "check should fail on passrate mismatch")
        self.assertIn("passrate mismatch", output)

    def test_empty_person_id_still_cross_validates(self) -> None:
        """person_id='' → submission ID is '-608-bug-fix-01'; cross-validation must still fire."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="", delivery_date="20260608")
                sub_id = "-608-bug-fix-01"
                csv_rows = [
                    _make_csv_row(sub_id, task.id, qwen_session="WRONG-SESSION")
                ]
                _setup_delivery(config, [task], csv_rows)
                result, output = _run_check(config)

        self.assertFalse(result, "check should fail even with empty person_id")
        self.assertIn("session_id mismatch", output)

    def test_no_issues_when_csv_matches_trajectory_and_score(self) -> None:
        """Sanity: no cross-validation issues when CSV data matches reality."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                csv_rows = [_make_csv_row(sub_id, task.id)]
                _setup_delivery(config, [task], csv_rows)
                result, output = _run_check(config)

        self.assertTrue(result, f"check should pass, but got:\n{output}")
        self.assertNotIn("mismatch", output)

    def test_multiple_tasks_with_different_types(self) -> None:
        """Two tasks of different types get sequential IDs; both cross-validated."""
        task1 = make_task("CT-0001", task_type="bug-fix")
        task2 = make_task("CT-0002", task_type="feature")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config(
                    [task1, task2], person_id="56", delivery_date="20260608"
                )
                csv_rows = [
                    _make_csv_row("56-608-bug-fix-01", task1.id),
                    _make_csv_row("56-608-feature-01", task2.id, claude_session="WRONG-CL"),
                ]
                _setup_delivery(config, [task1, task2], csv_rows)
                result, output = _run_check(config)

        self.assertFalse(result)
        self.assertIn("session_id mismatch", output)
        # Only CT-0002/claude should have the mismatch
        self.assertIn("CT-0002/claude", output)
        self.assertNotIn("CT-0001", output.split("session_id mismatch")[0].split("\n")[-1])


# =========================================================================
# Unit tests for _count_turns
# =========================================================================


class CountTurnsTest(unittest.TestCase):
    """Tests for _count_turns() trajectory parsing."""

    def test_normal_trajectory_with_multiple_turns(self) -> None:
        """Valid trajectory with 3 user turns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test.jsonl"
            write_trajectory(jsonl_path, "session-123", "qwen", user_turns=3)
            user_turns, models, session_id, issues = _count_turns(jsonl_path)

        self.assertEqual(user_turns, 3)
        self.assertEqual(len(models), 1)
        self.assertIn("qwen", models[0].lower())
        self.assertEqual(session_id, "session-123")
        self.assertEqual(len(issues), 0)

    def test_trajectory_too_short(self) -> None:
        """Trajectory with fewer than MIN_TRAJECTORY_LINES lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test.jsonl"
            # Write only 5 lines (less than MIN_TRAJECTORY_LINES=10)
            lines = [
                json.dumps({"sessionId": "sess-1", "type": "system"}),
                json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}),
                json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "qwen-v1", "content": "reply"}}),
                json.dumps({"type": "system", "info": "padding"}),
                json.dumps({"type": "system", "info": "padding"}),
            ]
            jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            user_turns, models, session_id, issues = _count_turns(jsonl_path)

        self.assertEqual(user_turns, 1)
        self.assertEqual(session_id, "sess-1")
        self.assertTrue(any("trajectory too short" in i for i in issues))

    def test_no_session_id(self) -> None:
        """Trajectory without sessionId field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test.jsonl"
            lines = []
            for i in range(12):
                lines.append(json.dumps({"type": "system", "info": f"line {i}"}))
            jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            user_turns, models, session_id, issues = _count_turns(jsonl_path)

        self.assertEqual(user_turns, 0)
        self.assertEqual(session_id, "")
        self.assertTrue(any("no session_id" in i for i in issues))

    def test_no_model_identifiers(self) -> None:
        """Trajectory without assistant messages containing model info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test.jsonl"
            lines = [json.dumps({"sessionId": "sess-1", "type": "system"})]
            for i in range(11):
                lines.append(json.dumps({"type": "user", "message": {"role": "user", "content": f"msg {i}"}}))
            jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            user_turns, models, session_id, issues = _count_turns(jsonl_path)

        self.assertGreater(user_turns, 0)
        self.assertEqual(len(models), 0)
        self.assertTrue(any("no model identifiers" in i for i in issues))

    def test_malformed_json_lines(self) -> None:
        """Trajectory with malformed JSON lines should skip them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test.jsonl"
            lines = [
                json.dumps({"sessionId": "sess-1", "type": "system"}),
                "not valid json {{{",
                json.dumps({"type": "user", "message": {"role": "user", "content": "msg"}}),
                "",
                json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "qwen-v1", "content": "reply"}}),
            ]
            # Add padding to reach MIN_TRAJECTORY_LINES
            while len(lines) < 12:
                lines.append(json.dumps({"type": "system", "info": "padding"}))
            jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            user_turns, models, session_id, issues = _count_turns(jsonl_path)

        self.assertEqual(user_turns, 1)
        self.assertEqual(len(models), 1)
        self.assertEqual(session_id, "sess-1")
        # Malformed lines should be skipped but still counted
        self.assertEqual(len(issues), 0)

    def test_multiple_session_ids(self) -> None:
        """Trajectory with multiple sessionId values should use the first one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test.jsonl"
            lines = [
                json.dumps({"sessionId": "first-session", "type": "system"}),
                json.dumps({"type": "user", "message": {"role": "user", "content": "msg"}}),
                json.dumps({"sessionId": "second-session", "type": "system"}),
            ]
            while len(lines) < 12:
                lines.append(json.dumps({"type": "system", "info": "padding"}))
            jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            user_turns, models, session_id, issues = _count_turns(jsonl_path)

        self.assertEqual(session_id, "first-session")

    def test_duplicate_models_deduplicated(self) -> None:
        """Same model appearing multiple times should be deduplicated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test.jsonl"
            lines = [json.dumps({"sessionId": "sess-1", "type": "system"})]
            for i in range(5):
                lines.append(json.dumps({"type": "user", "message": {"role": "user", "content": f"msg {i}"}}))
                lines.append(json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "qwen-same-model", "content": f"reply {i}"}}))
            while len(lines) < 12:
                lines.append(json.dumps({"type": "system", "info": "padding"}))
            jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            user_turns, models, session_id, issues = _count_turns(jsonl_path)

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0], "qwen-same-model")

    def test_empty_trajectory(self) -> None:
        """Empty trajectory file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test.jsonl"
            jsonl_path.write_text("", encoding="utf-8")
            user_turns, models, session_id, issues = _count_turns(jsonl_path)

        self.assertEqual(user_turns, 0)
        self.assertEqual(len(models), 0)
        self.assertEqual(session_id, "")
        self.assertTrue(any("trajectory too short" in i for i in issues))
        self.assertTrue(any("no session_id" in i for i in issues))
        self.assertTrue(any("no model identifiers" in i for i in issues))


# =========================================================================
# Unit tests for _check_model_identity
# =========================================================================


class CheckModelIdentityTest(unittest.TestCase):
    """Tests for _check_model_identity() model validation."""

    def test_qwen_found(self) -> None:
        """Qwen model detected correctly."""
        result = _check_model_identity(["qwen-model-v1"], "qwen")
        self.assertIsNone(result)

    def test_claude_found(self) -> None:
        """Claude model detected correctly."""
        result = _check_model_identity(["claude-3-opus"], "claude")
        self.assertIsNone(result)

    def test_claude_via_anthropic_keyword(self) -> None:
        """Claude model detected via 'anthropic' keyword."""
        result = _check_model_identity(["anthropic-claude-v1"], "claude")
        self.assertIsNone(result)

    def test_empty_models_list(self) -> None:
        """Empty models list returns error."""
        result = _check_model_identity([], "qwen")
        self.assertIsNotNone(result)
        self.assertIn("no models detected", result)

    def test_qwen_expected_but_claude_found(self) -> None:
        """Mismatch: expected qwen but found claude."""
        result = _check_model_identity(["claude-3-opus"], "qwen")
        self.assertIsNotNone(result)
        self.assertIn("expected qwen model", result)
        self.assertIn("claude-3-opus", result)

    def test_claude_expected_but_qwen_found(self) -> None:
        """Mismatch: expected claude but found qwen."""
        result = _check_model_identity(["qwen-model-v1"], "claude")
        self.assertIsNotNone(result)
        self.assertIn("expected claude model", result)
        self.assertIn("qwen-model-v1", result)

    def test_case_insensitive_matching(self) -> None:
        """Model matching should be case-insensitive."""
        result = _check_model_identity(["QWEN-Model-V1"], "qwen")
        self.assertIsNone(result)
        result = _check_model_identity(["Claude-3-Opus"], "claude")
        self.assertIsNone(result)

    def test_multiple_models_one_matches(self) -> None:
        """Multiple models detected, at least one matches expected."""
        result = _check_model_identity(["other-model", "qwen-v1", "another-model"], "qwen")
        self.assertIsNone(result)

    def test_multiple_models_none_match(self) -> None:
        """Multiple models detected, none match expected."""
        result = _check_model_identity(["gpt-4", "gemini-pro"], "qwen")
        self.assertIsNotNone(result)
        self.assertIn("expected qwen model", result)


# =========================================================================
# Unit tests for score file validation in check()
# =========================================================================


class ScoreValidationTest(unittest.TestCase):
    """Tests for score/rubric validation branches in check()."""

    def test_unscored_template(self) -> None:
        """All criteria with score=0 and empty rationale → unscored template."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                csv_rows = [_make_csv_row(sub_id, task.id)]
                _setup_delivery(config, [task], csv_rows)

                # Overwrite qwen score with unscored template (score=0, no rationale)
                names = REFERENCE_CRITERION_NAMES[:7]
                criteria = [
                    Criterion(
                        name, REFERENCE_CRITERION_DESCRIPTIONS[name], "likert", 5,
                        1.0, 0, ""
                    )
                    for name in names
                ]
                score_path = config.delivery_dir / "scores" / "qwen" / f"{model_stem(task.id, 'qwen')}.quality.toml"
                _write_score_custom(score_path, criteria)

                result, output = _run_check(config)

        self.assertFalse(result)
        self.assertIn("score file is still an unscored template", output)
        self.assertIn("CT-0001/qwen", output)

    def test_too_few_criteria(self) -> None:
        """5 criteria (below MIN_CRITERIA_COUNT=7) → wrong criteria count."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                csv_rows = [_make_csv_row(sub_id, task.id)]
                _setup_delivery(config, [task], csv_rows)

                names = REFERENCE_CRITERION_NAMES[:5]
                criteria = [
                    Criterion(
                        name, REFERENCE_CRITERION_DESCRIPTIONS[name], "likert", 5,
                        1.0, 3, "评分理由"
                    )
                    for name in names
                ]
                score_path = config.delivery_dir / "scores" / "qwen" / f"{model_stem(task.id, 'qwen')}.quality.toml"
                _write_score_custom(score_path, criteria)

                result, output = _run_check(config)

        self.assertFalse(result)
        self.assertIn("wrong criteria count: 5", output)

    def test_too_many_criteria(self) -> None:
        """11 criteria (above MAX_CRITERIA_COUNT=10) → wrong criteria count."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                csv_rows = [_make_csv_row(sub_id, task.id)]
                _setup_delivery(config, [task], csv_rows)

                names = REFERENCE_CRITERION_NAMES[:11]
                criteria = [
                    Criterion(
                        name, REFERENCE_CRITERION_DESCRIPTIONS[name], "likert", 5,
                        1.0, 3, "评分理由"
                    )
                    for name in names
                ]
                score_path = config.delivery_dir / "scores" / "qwen" / f"{model_stem(task.id, 'qwen')}.quality.toml"
                _write_score_custom(score_path, criteria)

                result, output = _run_check(config)

        self.assertFalse(result)
        self.assertIn("wrong criteria count: 11", output)

    def test_score_out_of_range_high(self) -> None:
        """One criterion with score=6 (above max 5) → out of range."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                # passrate = (6+3*6)/(5*7) = 24/35 ≈ 0.6857
                csv_rows = [_make_csv_row(sub_id, task.id, qwen_passrate="0.6857")]
                _setup_delivery(config, [task], csv_rows)

                names = REFERENCE_CRITERION_NAMES[:7]
                criteria = [
                    Criterion(
                        names[0], REFERENCE_CRITERION_DESCRIPTIONS[names[0]], "likert", 5,
                        1.0, 6, "评分理由"
                    )
                ] + [
                    Criterion(
                        name, REFERENCE_CRITERION_DESCRIPTIONS[name], "likert", 5,
                        1.0, 3, "评分理由"
                    )
                    for name in names[1:]
                ]
                score_path = config.delivery_dir / "scores" / "qwen" / f"{model_stem(task.id, 'qwen')}.quality.toml"
                _write_score_custom(score_path, criteria)

                result, output = _run_check(config)

        self.assertFalse(result)
        self.assertIn("score 6 out of range 1-5", output)

    def test_score_out_of_range_low(self) -> None:
        """One criterion with score=0 but has rationale → out of range (not unscored)."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                # passrate = (0+3*6)/(5*7) = 18/35 ≈ 0.5143
                csv_rows = [_make_csv_row(sub_id, task.id, qwen_passrate="0.5143")]
                _setup_delivery(config, [task], csv_rows)

                names = REFERENCE_CRITERION_NAMES[:7]
                criteria = [
                    Criterion(
                        names[0], REFERENCE_CRITERION_DESCRIPTIONS[names[0]], "likert", 5,
                        1.0, 0, "待改进"
                    )
                ] + [
                    Criterion(
                        name, REFERENCE_CRITERION_DESCRIPTIONS[name], "likert", 5,
                        1.0, 3, "评分理由"
                    )
                    for name in names[1:]
                ]
                score_path = config.delivery_dir / "scores" / "qwen" / f"{model_stem(task.id, 'qwen')}.quality.toml"
                _write_score_custom(score_path, criteria)

                result, output = _run_check(config)

        self.assertFalse(result)
        self.assertIn("score 0 out of range 1-5", output)
        # Should NOT be flagged as unscored template (one criterion has rationale)
        self.assertNotIn("unscored template", output)

    def test_missing_rationale(self) -> None:
        """One criterion with score but no rationale → missing rationale + incomplete."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                csv_rows = [_make_csv_row(sub_id, task.id)]
                _setup_delivery(config, [task], csv_rows)

                names = REFERENCE_CRITERION_NAMES[:7]
                criteria = [
                    Criterion(
                        names[0], REFERENCE_CRITERION_DESCRIPTIONS[names[0]], "likert", 5,
                        1.0, 3, ""
                    )
                ] + [
                    Criterion(
                        name, REFERENCE_CRITERION_DESCRIPTIONS[name], "likert", 5,
                        1.0, 3, "评分理由"
                    )
                    for name in names[1:]
                ]
                score_path = config.delivery_dir / "scores" / "qwen" / f"{model_stem(task.id, 'qwen')}.quality.toml"
                _write_score_custom(score_path, criteria)

                result, output = _run_check(config)

        self.assertFalse(result)
        self.assertIn("missing rationale", output)
        self.assertIn("incomplete scoring: 6/7", output)

    def test_invalid_criterion_name(self) -> None:
        """All criteria with names that are not valid snake_case → invalid name."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                # passrate = (3*7)/(5*7) = 0.6000
                csv_rows = [_make_csv_row(sub_id, task.id, qwen_passrate="0.6000")]
                _setup_delivery(config, [task], csv_rows)

                criteria = [
                    Criterion(
                        f"InvalidName{i}", "自定义描述", "likert", 5,
                        1.0, 3, "评分理由"
                    )
                    for i in range(7)
                ]
                score_path = config.delivery_dir / "scores" / "qwen" / f"{model_stem(task.id, 'qwen')}.quality.toml"
                _write_score_custom(score_path, criteria)

                result, output = _run_check(config)

        self.assertFalse(result)
        self.assertIn("invalid name", output)


# =========================================================================
# Unit tests for cross-model criterion consistency check
# =========================================================================


class CrossModelCriterionConsistencyTest(unittest.TestCase):
    """Tests for the warning when qwen and claude use different scoring dimensions."""

    def test_criterion_mismatch_warning(self) -> None:
        """Qwen and claude use different valid criteria → warning issued."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                # Qwen: score=3 → passrate=0.6000, Claude: score=5 → passrate=1.0000
                csv_rows = [_make_csv_row(
                    sub_id, task.id,
                    qwen_passrate="0.6000",
                    claude_passrate="1.0000"
                )]
                _setup_delivery(config, [task], csv_rows)

                # Qwen uses REFERENCE_CRITERION_NAMES[0:7] with score=3
                names_qwen = REFERENCE_CRITERION_NAMES[0:7]
                criteria_qwen = [
                    Criterion(
                        name, REFERENCE_CRITERION_DESCRIPTIONS[name], "likert", 5,
                        1.0, 3, "评分理由"
                    )
                    for name in names_qwen
                ]
                qwen_score_path = config.delivery_dir / "scores" / "qwen" / f"{model_stem(task.id, 'qwen')}.quality.toml"
                _write_score_custom(qwen_score_path, criteria_qwen)

                # Claude uses REFERENCE_CRITERION_NAMES[1:8] (shifted by one) with score=5
                names_claude = REFERENCE_CRITERION_NAMES[1:8]
                criteria_claude = [
                    Criterion(
                        name, REFERENCE_CRITERION_DESCRIPTIONS[name], "likert", 5,
                        1.0, 5, "评分理由"
                    )
                    for name in names_claude
                ]
                claude_score_path = config.delivery_dir / "scores" / "claude" / f"{model_stem(task.id, 'claude')}.quality.toml"
                _write_score_custom(claude_score_path, criteria_claude)

                result, output = _run_check(config)

        # Should fail (criterion mismatch is now an error, not a warning)
        self.assertFalse(result, f"check should fail on criterion mismatch, but it passed:\n{output}")
        # Should mention the criterion name mismatch
        self.assertIn("criterion name mismatch between qwen/claude", output)
        # Should mention criteria only in qwen and only in claude
        only_in_qwen = set(names_qwen) - set(names_claude)
        only_in_claude = set(names_claude) - set(names_qwen)
        for name in only_in_qwen:
            self.assertIn(name, output)
        for name in only_in_claude:
            self.assertIn(name, output)

    def test_criterion_match_no_warning(self) -> None:
        """Qwen and claude use identical criteria → no warning."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                csv_rows = [_make_csv_row(sub_id, task.id)]
                # Both use default write_score which uses REFERENCE_CRITERION_NAMES[:7]
                _setup_delivery(config, [task], csv_rows)
                result, output = _run_check(config)

        self.assertTrue(result, f"check should pass, but got:\n{output}")
        self.assertNotIn("criterion mismatch", output)


    def test_no_warning_when_one_side_score_missing(self) -> None:
        """When one model's score file is missing, no criterion mismatch warning."""
        task = make_task("CT-0001")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config([task], person_id="56", delivery_date="20260608")
                sub_id = "56-608-bug-fix-01"
                csv_rows = [_make_csv_row(sub_id, task.id)]
                _setup_delivery(config, [task], csv_rows)

                # Delete claude score file so qwen_criteria is set but claude_criteria is empty
                claude_score_path = config.delivery_dir / "scores" / "claude" / f"{model_stem(task.id, 'claude')}.quality.toml"
                claude_score_path.unlink()

                result, output = _run_check(config)

        self.assertFalse(result, "check should fail due to missing score file")
        self.assertIn("score file missing", output)
        # No criterion mismatch warning since one side has no criteria data
        self.assertNotIn("criterion mismatch", output)


if __name__ == "__main__":
    unittest.main()
