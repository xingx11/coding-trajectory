from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from ctpipe.config import (
    BatchConfig,
    ModelConfig,
    TaskConfig,
    select_delivery_tasks,
    write_task_manifest,
)
from ctpipe.distribution import TaskSlot
from ctpipe.prepare import _create_submission_csv
from ctpipe.state import PipelineState
from ctpipe.trajectory import find_trajectory_for_run


def _build_config(tasks: list[TaskConfig] | None = None) -> BatchConfig:
    return BatchConfig(
        delivery_date="20990101",
        runs_root=Path("D:/runs"),
        max_parallel=2,
        tasks=tasks or [],
        qwen=ModelConfig(auth_token="", base_url="", model="qwen-test"),
        claude=ModelConfig(auth_token="", base_url="", model="claude-test"),
    )


def _make_task(task_id: str = "CT-0001") -> TaskConfig:
    return TaskConfig(
        id=task_id,
        project_path=Path("D:/projects/demo"),
        clone_method="git",
        task_type="bug-fix",
        domain="web_frontend",
        language="ts",
        prompt_qwen="qwen prompt",
        prompt_claude="claude prompt",
        followups_qwen=["f1"],
        followups_claude=["f1", "f2"],
    )


class PipelineOptimizationTests(unittest.TestCase):
    def test_batch_mode_expand_slot_produces_distinct_types(self) -> None:
        """expand_slot_for_batch produces per_project slots with distinct task_types."""
        from ctpipe.distribution import expand_slot_for_batch

        slot = TaskSlot("bug-fix", "ai_ml", "python", 7.6)
        result = expand_slot_for_batch(slot, 3)

        self.assertEqual(len(result), 3)
        # All share same domain and language
        for s in result:
            self.assertEqual(s.domain, "ai_ml")
            self.assertEqual(s.language, "python")
        # First slot keeps original task_type
        self.assertEqual(result[0].task_type, "bug-fix")
        # All task_types are distinct
        types = [s.task_type for s in result]
        self.assertEqual(len(set(types)), 3)

    def test_batch_mode_samples_num_projects_not_count(self) -> None:
        """With --count 6 --per-project 3, only 2 slots are sampled (not 6)."""
        from ctpipe.distribution import sample_slots

        count = 6
        per_project = 3
        num_projects = -(-count // per_project)
        sampled = sample_slots(num_projects)

        self.assertEqual(len(sampled), 2)

    def test_prepare_submission_csv_copies_header_only(self) -> None:
        config = _build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "submission.csv"
            _create_submission_csv(config, csv_path)
            rows = csv_path.read_text(encoding="utf-8-sig").splitlines()

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].startswith("id,qwen 本地trajectory"))

    def test_select_delivery_tasks_prefers_manifest_snapshot(self) -> None:
        manifest_task = TaskConfig(
            id="CT-0007",
            project_path=Path("D:/projects/demo"),
            clone_method="git",
            task_type="bug-fix",
            domain="web_frontend",
            language="ts",
            prompt_qwen="qwen prompt",
            prompt_claude="claude prompt",
            followups_qwen=["f1"],
            followups_claude=["f1", "f2"],
        )
        config = _build_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                write_task_manifest(config.task_manifest_path, [manifest_task])
                selected = select_delivery_tasks(config, ["CT-0007"])

        self.assertEqual([task.id for task in selected], ["CT-0007"])
        self.assertEqual(selected[0].prompt_claude, "claude prompt")


class RunAllErrorsTest(unittest.TestCase):
    """When every turn exits with nonzero, status should be 'failed', not 'done'."""

    def test_all_turns_errored_marks_failed(self) -> None:
        from ctpipe.run import run_single

        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            state = PipelineState(Path(tmpdir) / "state.json")
            model_config = ModelConfig(auth_token="tok", base_url="http://x", model="test")

            import asyncio

            async def fake_run_claude_p(prompt, env, cwd, model=None, resume_session=None, timeout=600):
                from ctpipe.run import TurnResult
                return TurnResult(
                    turn=0,
                    exit_code=1,
                    stdout='{"session_id": "sess-123"}',
                    stderr="error",
                    duration_s=1.0,
                    session_id="sess-123",
                )

            with patch("ctpipe.run._run_claude_p", side_effect=fake_run_claude_p):
                summary = asyncio.run(run_single(
                    task, "qwen", model_config, run_dir,
                    prompt="do something",
                    followups=["followup1", "followup2"],
                    state=state,
                    turn_timeout=60, total_timeout=300,
                ))

            self.assertEqual(summary["status"], "failed")
            self.assertTrue(summary["had_errors"])
            self.assertEqual(summary["turns"], 3)

    def test_some_turns_errored_marks_done(self) -> None:
        from ctpipe.run import run_single, TurnResult

        task = _make_task()
        call_count = 0

        async def fake_run_claude_p(prompt, env, cwd, model=None, resume_session=None, timeout=600):
            nonlocal call_count
            call_count += 1
            exit_code = 1 if call_count == 2 else 0
            return TurnResult(
                turn=0,
                exit_code=exit_code,
                stdout='{"session_id": "sess-456"}',
                stderr="",
                duration_s=1.0,
                session_id="sess-456",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            state = PipelineState(Path(tmpdir) / "state.json")
            model_config = ModelConfig(auth_token="tok", base_url="http://x", model="test")

            import asyncio
            with patch("ctpipe.run._run_claude_p", side_effect=fake_run_claude_p):
                summary = asyncio.run(run_single(
                    task, "qwen", model_config, run_dir,
                    prompt="do something",
                    followups=["followup1"],
                    state=state,
                    turn_timeout=60, total_timeout=300,
                ))

            self.assertEqual(summary["status"], "done")
            self.assertTrue(summary["had_errors"])


class CollectPartialRunTest(unittest.TestCase):
    """Partial run status should be accepted by collect_single."""

    def test_partial_run_is_collected(self) -> None:
        from ctpipe.collect import collect_single
        from ctpipe.trajectory import TrajectoryInfo

        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                (delivery_dir / "trajectories" / "qwen").mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "run", model="qwen", status="partial", session_id="s1", start_time=0)
                state.set(task.id, "prepare", qwen_dir=tmpdir)

                fake_jsonl = Path(tmpdir) / "fake.jsonl"
                fake_jsonl.write_text('{"sessionId":"s1"}\n', encoding="utf-8")
                fake_info = TrajectoryInfo(file_path=fake_jsonl, session_id="s1", line_count=50, models={"qwen-test"})

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=fake_jsonl), \
                     patch("ctpipe.collect.parse_trajectory", return_value=fake_info):
                    result = collect_single(task, "qwen", config, state)

                self.assertTrue(result)
                collect_info = state.get(task.id, "collect", "qwen")
                self.assertEqual(collect_info["status"], "done")

    def test_failed_run_is_skipped_by_collect(self) -> None:
        from ctpipe.collect import collect_single

        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "run", model="qwen", status="failed", error="no session")
                state.set(task.id, "prepare", qwen_dir=tmpdir)

                result = collect_single(task, "qwen", config, state)

                self.assertFalse(result)
                collect_info = state.get(task.id, "collect", "qwen")
                self.assertNotEqual(collect_info.get("status"), "done")


class ScoreExceptionWritesFailedTest(unittest.TestCase):
    """When score_single raises an exception caught by gather, state should be 'failed'."""

    def test_exception_writes_failed_state(self) -> None:
        import asyncio
        from ctpipe.score import score_all

        config = _build_config([_make_task()])

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)

            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                (delivery_dir / "metadata").mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [_make_task()])

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set("CT-0001", "collect", model="qwen", status="done",
                          jsonl_path="trajectories/qwen/CT-0001.jsonl")
                state.save()

                def fake_build_scoring_env(cfg):
                    return {"PATH": ""}

                async def fake_score_single(task, model_name, cfg, st, env):
                    raise RuntimeError("test explosion")

                with patch("ctpipe.score._build_scoring_env", side_effect=fake_build_scoring_env), \
                     patch("ctpipe.score.score_single", side_effect=fake_score_single):
                    asyncio.run(score_all(config, models=["qwen"]))

                state2 = PipelineState(delivery_dir / "pipeline_state.json")
                info = state2.get("CT-0001", "score", "qwen")
                self.assertEqual(info.get("status"), "failed")


class TrajectorySessionIdScanTest(unittest.TestCase):
    """sessionId placed beyond line 5 should still be found."""

    def test_session_id_on_line_10(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = Path(tmpdir) / "projects" / "hash123"
            proj_dir.mkdir(parents=True)

            target = proj_dir / "correct.jsonl"
            lines = []
            for i in range(9):
                lines.append(json.dumps({"type": "system", "message": f"line {i}"}))
            lines.append(json.dumps({"sessionId": "target-session", "type": "user"}))
            target.write_text("\n".join(lines), encoding="utf-8")

            wrong = proj_dir / "newer.jsonl"
            wrong.write_text(json.dumps({"sessionId": "wrong-session"}) + "\n", encoding="utf-8")

            import os
            import time
            old_time = time.time() - 100
            os.utime(target, (old_time + 10, old_time + 10))
            os.utime(wrong, (old_time + 50, old_time + 50))

            with patch("ctpipe.trajectory.project_hash_dir", return_value=proj_dir):
                result = find_trajectory_for_run(
                    Path(tmpdir),
                    start_time=old_time,
                    expected_session_id="target-session",
                )

            self.assertIsNotNone(result)
            self.assertEqual(result.name, "correct.jsonl")

    def test_session_id_on_line_1_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = Path(tmpdir) / "projects" / "hash456"
            proj_dir.mkdir(parents=True)

            target = proj_dir / "found.jsonl"
            target.write_text(json.dumps({"sessionId": "s1"}) + "\n", encoding="utf-8")

            import os
            import time
            old_time = time.time() - 100
            os.utime(target, (old_time + 10, old_time + 10))

            with patch("ctpipe.trajectory.project_hash_dir", return_value=proj_dir):
                result = find_trajectory_for_run(
                    Path(tmpdir),
                    start_time=old_time,
                    expected_session_id="s1",
                )

            self.assertIsNotNone(result)
            self.assertEqual(result.name, "found.jsonl")


class CheckStageMissingStatusTest(unittest.TestCase):
    """_check_stage logic: missing/draft statuses must be counted as problems."""

    def _count_problems(self, state: PipelineState, task_ids: list[str], stage: str, models: list[str]) -> dict:
        failed = 0
        partial = 0
        missing = 0
        for task_id in task_ids:
            if stage in ("run", "collect", "score"):
                for m in models:
                    info = state.get(task_id, stage, m)
                    status = info.get("status", "")
                    if status == "failed":
                        failed += 1
                    elif status == "partial":
                        partial += 1
                    elif status in ("", "draft"):
                        missing += 1
            else:
                info = state.get(task_id, stage)
                status = info.get("status", "")
                if status == "failed":
                    failed += 1
                elif status == "partial":
                    partial += 1
                elif status in ("", "draft"):
                    missing += 1
        return {"failed": failed, "partial": partial, "missing": missing}

    def test_empty_status_counted_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "score", model="qwen")
            counts = self._count_problems(state, ["CT-0001"], "score", ["qwen"])
            self.assertEqual(counts["missing"], 1)
            self.assertEqual(counts["failed"], 0)

    def test_draft_status_counted_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "score", model="qwen", status="draft")
            counts = self._count_problems(state, ["CT-0001"], "score", ["qwen"])
            self.assertEqual(counts["missing"], 1)

    def test_done_status_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "score", model="qwen", status="done")
            counts = self._count_problems(state, ["CT-0001"], "score", ["qwen"])
            self.assertEqual(counts["missing"], 0)
            self.assertEqual(counts["failed"], 0)
            self.assertEqual(counts["partial"], 0)

    def test_multiple_models_counted_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "score", model="qwen", status="failed")
            state.set("CT-0001", "score", model="claude", status="draft")
            counts = self._count_problems(state, ["CT-0001"], "score", ["qwen", "claude"])
            self.assertEqual(counts["failed"], 1)
            self.assertEqual(counts["missing"], 1)


class UnscoredTemplateDetectionTest(unittest.TestCase):
    """Unscored templates (all score=0, no rationale) must not produce passrates."""

    def test_is_unscored_template_true(self) -> None:
        from ctpipe.toml_utils import Criterion, is_unscored_template
        criteria = [
            Criterion("c1", "desc", "likert", 5, 1.0, 0, ""),
            Criterion("c2", "desc", "likert", 5, 1.0, 0, ""),
        ]
        self.assertTrue(is_unscored_template(criteria))

    def test_is_unscored_template_false_when_scored(self) -> None:
        from ctpipe.toml_utils import Criterion, is_unscored_template
        criteria = [
            Criterion("c1", "desc", "likert", 5, 1.0, 3, "good work"),
            Criterion("c2", "desc", "likert", 5, 1.0, 0, ""),
        ]
        self.assertFalse(is_unscored_template(criteria))

    def test_is_unscored_template_false_when_rationale_only(self) -> None:
        from ctpipe.toml_utils import Criterion, is_unscored_template
        criteria = [
            Criterion("c1", "desc", "likert", 5, 1.0, 0, "attempted but failed"),
        ]
        self.assertFalse(is_unscored_template(criteria))


class TrajectoryNonDictJsonTest(unittest.TestCase):
    """parse_trajectory must not crash on valid JSON that isn't a dict (e.g. arrays, scalars)."""

    def test_parse_trajectory_skips_non_dict_lines(self) -> None:
        from ctpipe.trajectory import parse_trajectory

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl = Path(tmpdir) / "mixed.jsonl"
            lines = [
                '[1, 2, 3]',
                '"just a string"',
                '42',
                'true',
                'null',
                json.dumps({"sessionId": "s1", "type": "user", "timestamp": "2026-01-01T00:00:00Z"}),
                json.dumps({"message": {"role": "assistant", "model": "test-model", "content": "hi"}}),
            ]
            jsonl.write_text("\n".join(lines), encoding="utf-8")

            info = parse_trajectory(jsonl)

        self.assertEqual(info.session_id, "s1")
        self.assertIn("test-model", info.models)
        self.assertEqual(info.line_count, 7)

    def test_extract_for_scoring_skips_non_dict_lines(self) -> None:
        from ctpipe.trajectory import extract_for_scoring

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl = Path(tmpdir) / "mixed2.jsonl"
            lines = [
                '[1, 2]',
                json.dumps({"message": {"role": "user", "content": "hello"}}),
                '999',
                json.dumps({"message": {"role": "assistant", "content": "world"}}),
            ]
            jsonl.write_text("\n".join(lines), encoding="utf-8")

            result = extract_for_scoring(jsonl)

        self.assertIn("hello", result)
        self.assertIn("world", result)


class FinalizeTrajectoryIssueStatusTest(unittest.TestCase):
    """finalize must set status='failed' when trajectory has no valid content or parse errors."""

    def test_no_valid_content_marks_failed(self) -> None:
        from ctpipe.trajectory import TrajectoryInfo

        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                traj_dir = delivery_dir / "trajectories" / "qwen"
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj_file = traj_dir / f"{task.id}.jsonl"
                traj_file.write_text("{}\n", encoding="utf-8")

                score_dir = delivery_dir / "scores" / "qwen"
                score_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "collect", model="qwen", status="done",
                          jsonl_path=f"trajectories/qwen/{task.id}.jsonl", session_id="s1")
                state.save()

                empty_info = TrajectoryInfo(
                    file_path=traj_file, session_id="", models=set(), line_count=1,
                )

                with patch("ctpipe.finalize.find_delivery_trajectory", return_value=traj_file), \
                     patch("ctpipe.finalize.parse_trajectory", return_value=empty_info):
                    from ctpipe.finalize import finalize
                    finalize(config, models=["qwen"])

                state2 = PipelineState(delivery_dir / "pipeline_state.json")
                info = state2.get(task.id, "finalize")
                self.assertEqual(info.get("status"), "failed")

    def test_parse_error_marks_failed(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                traj_dir = delivery_dir / "trajectories" / "qwen"
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj_file = traj_dir / f"{task.id}.jsonl"
                traj_file.write_text("{}\n", encoding="utf-8")

                score_dir = delivery_dir / "scores" / "qwen"
                score_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "collect", model="qwen", status="done",
                          jsonl_path=f"trajectories/qwen/{task.id}.jsonl", session_id="s1")
                state.save()

                with patch("ctpipe.finalize.find_delivery_trajectory", return_value=traj_file), \
                     patch("ctpipe.finalize.parse_trajectory", side_effect=Exception("parse error: bad encoding")):
                    from ctpipe.finalize import finalize
                    finalize(config, models=["qwen"])

                state2 = PipelineState(delivery_dir / "pipeline_state.json")
                info = state2.get(task.id, "finalize")
                self.assertEqual(info.get("status"), "failed")


class ValidateNonDictJsonTest(unittest.TestCase):
    """validate must not crash when trajectory JSONL contains non-dict JSON values."""

    def test_validate_handles_array_json_lines(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                (delivery_dir / "metadata").mkdir(parents=True, exist_ok=True)
                (delivery_dir / "metadata" / f"{task.id}.md").write_text("# task", encoding="utf-8")

                traj_dir = delivery_dir / "trajectories" / "qwen"
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj_file = traj_dir / f"{task.id}.jsonl"
                lines = [
                    '[1, 2, 3]',
                    json.dumps({"sessionId": "s1", "message": {"role": "assistant", "model": "qwen-x", "content": "hi"}}),
                ]
                traj_file.write_text("\n".join(lines), encoding="utf-8")

                from ctpipe.validate import validate
                result = validate(config, models=["qwen"])

        self.assertFalse(result)


class FinalizeSingleModelThresholdTest(unittest.TestCase):
    """Single-model finalize must evaluate thresholds only against requested models."""

    def test_single_model_qwen_done_when_threshold_met(self) -> None:
        from ctpipe.finalize import finalize
        from ctpipe.toml_utils import Criterion

        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                traj_dir = delivery_dir / "trajectories" / "qwen"
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj_file = traj_dir / f"{task.id}.jsonl"
                traj_file.write_text(
                    json.dumps({"sessionId": "s1", "message": {"role": "assistant", "model": "qwen-x", "content": "hi"}}) + "\n",
                    encoding="utf-8",
                )

                score_dir = delivery_dir / "scores" / "qwen"
                score_dir.mkdir(parents=True, exist_ok=True)
                score_file = score_dir / f"{task.id}.quality.toml"
                score_file.write_text("dummy", encoding="utf-8")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "collect", model="qwen", status="done",
                          jsonl_path=f"trajectories/qwen/{task.id}.jsonl", session_id="s1")
                state.save()

                criteria = [
                    Criterion(f"c{i}", "desc", "likert", 5, 1.0, 3, "ok") for i in range(7)
                ]

                with patch("ctpipe.finalize.find_delivery_trajectory", return_value=traj_file), \
                     patch("ctpipe.finalize.read_quality_toml", return_value=criteria):
                    finalize(config, models=["qwen"])

                state2 = PipelineState(delivery_dir / "pipeline_state.json")
                info = state2.get(task.id, "finalize")
                self.assertEqual(info.get("status"), "done")
                self.assertTrue(info.get("threshold_ok"))


class FindTrajectoryNonDictJsonTest(unittest.TestCase):
    """find_trajectory_for_run must not crash on non-dict JSON when scanning sessionId."""

    def test_skips_array_lines_in_session_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = Path(tmpdir) / "projects" / "hashABC"
            proj_dir.mkdir(parents=True)

            target = proj_dir / "target.jsonl"
            lines = [
                '[1, 2, 3]',
                '"a bare string"',
                json.dumps({"sessionId": "wanted", "type": "user"}),
            ]
            target.write_text("\n".join(lines), encoding="utf-8")

            import os
            import time
            old_time = time.time() - 100
            os.utime(target, (old_time + 10, old_time + 10))

            with patch("ctpipe.trajectory.project_hash_dir", return_value=proj_dir):
                result = find_trajectory_for_run(
                    Path(tmpdir),
                    start_time=old_time,
                    expected_session_id="wanted",
                )

            self.assertIsNotNone(result)
            self.assertEqual(result.name, "target.jsonl")


class ValidateBadTomlTest(unittest.TestCase):
    """validate must not crash on malformed TOML score files."""

    def test_bad_toml_recorded_as_issue(self) -> None:
        from ctpipe.validate import validate

        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                (delivery_dir / "metadata").mkdir(parents=True, exist_ok=True)
                (delivery_dir / "metadata" / f"{task.id}.md").write_text("# task", encoding="utf-8")

                csv_path = delivery_dir / "submission.csv"
                csv_path.write_text("id\nCT-0001\n", encoding="utf-8-sig")

                traj_dir = delivery_dir / "trajectories" / "qwen"
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj_file = traj_dir / f"{task.id}.jsonl"
                traj_file.write_text(
                    json.dumps({"sessionId": "s1", "message": {"role": "assistant", "model": "qwen-x", "content": "hi"}}) + "\n",
                    encoding="utf-8",
                )

                score_dir = delivery_dir / "scores" / "qwen"
                score_dir.mkdir(parents=True, exist_ok=True)
                score_file = score_dir / f"{task.id}.quality.toml"
                score_file.write_text("this is not valid [[[ toml", encoding="utf-8")

                result = validate(config, models=["qwen"])

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
