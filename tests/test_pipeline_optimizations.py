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

from conftest import build_config, make_task


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
        config = build_config(person_id="")
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
        config = build_config(person_id="")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                write_task_manifest(config.task_manifest_path, [manifest_task])
                selected = select_delivery_tasks(config, ["CT-0007"])

        self.assertEqual([task.id for task in selected], ["CT-0007"])
        self.assertEqual(selected[0].prompt_claude, "claude prompt")


class BatchResetNoIntermediateSaveTest(unittest.TestCase):
    """reset() inside a batch() context must not trigger intermediate disk writes."""

    def test_batch_mixed_reset_and_set_saves_only_on_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            state = PipelineState(path)
            # Seed some data outside batch so reset() has something to delete.
            state.set("CT-0001", "run", model="qwen", status="done")
            state.set("CT-0001", "run", model="claude", status="done")
            state.set("CT-0002", "score", model="qwen", status="done")

            with patch.object(state, "save", wraps=state.save) as mock_save:
                with state.batch():
                    # reset a model-level entry
                    state.reset("CT-0001", "run", model="qwen")
                    # reset a stage-level entry
                    state.reset("CT-0002", "score")
                    # set a new value
                    state.set("CT-0001", "run", model="qwen", status="pending")
                    # set another value
                    state.set("CT-0003", "prepare", status="done")

                    # No save should have happened yet.
                    mock_save.assert_not_called()

                # Exiting the batch triggers exactly one save.
                self.assertEqual(mock_save.call_count, 1)

            # Verify the final on-disk state reflects all mutations.
            reloaded = PipelineState(path)
            self.assertEqual(reloaded.get("CT-0001", "run", "qwen").get("status"), "pending")
            self.assertEqual(reloaded.get("CT-0001", "run", "claude").get("status"), "done")
            self.assertEqual(reloaded.get("CT-0002", "score"), {})
            self.assertEqual(reloaded.get("CT-0003", "prepare").get("status"), "done")

    def test_reset_outside_batch_saves_immediately(self) -> None:
        """reset() called outside batch() must write to disk right away."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            state = PipelineState(path)
            state.set("CT-0001", "run", model="qwen", status="done")
            state.set("CT-0001", "run", model="claude", status="done")
            state.set("CT-0002", "score", status="done")

            snapshot_before = path.read_text(encoding="utf-8")

            # model-level reset
            state.reset("CT-0001", "run", model="qwen")
            self.assertNotEqual(path.read_text(encoding="utf-8"), snapshot_before)
            reloaded = PipelineState(path)
            self.assertEqual(reloaded.get("CT-0001", "run", "qwen"), {})
            self.assertEqual(reloaded.get("CT-0001", "run", "claude").get("status"), "done")

            snapshot_mid = path.read_text(encoding="utf-8")

            # stage-level reset
            state.reset("CT-0002", "score")
            self.assertNotEqual(path.read_text(encoding="utf-8"), snapshot_mid)
            reloaded = PipelineState(path)
            self.assertEqual(reloaded.get("CT-0002", "score"), {})

    def test_batch_no_file_written_mid_batch(self) -> None:
        """Even without patching, the JSON file on disk must not change mid-batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            state = PipelineState(path)
            state.set("CT-0001", "run", model="qwen", status="done")

            # Snapshot the file content after the initial set.
            snapshot = path.read_text(encoding="utf-8")

            with state.batch():
                state.reset("CT-0001", "run", model="qwen")
                state.set("CT-0002", "prepare", status="done")
                # File should still match the pre-batch snapshot.
                self.assertEqual(path.read_text(encoding="utf-8"), snapshot)

            # After exiting, the file must differ (changes persisted).
            self.assertNotEqual(path.read_text(encoding="utf-8"), snapshot)


class RunAllErrorsTest(unittest.TestCase):
    """When every turn exits with nonzero, status should be 'failed', not 'done'."""

    def test_all_turns_errored_marks_failed(self) -> None:
        from ctpipe.run import run_single

        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
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

    def test_some_turns_errored_marks_partial(self) -> None:
        """When some (but not all) turns fail, status should be 'partial'."""
        from ctpipe.run import run_single, TurnResult

        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
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

            self.assertEqual(summary["status"], "partial")
            self.assertTrue(summary["had_errors"])


class CollectPartialRunTest(unittest.TestCase):
    """Partial run status should be accepted by collect_single."""

    def test_partial_run_is_collected(self) -> None:
        from ctpipe.collect import collect_single
        from ctpipe.trajectory import TrajectoryInfo

        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = build_config([task], person_id="")
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

        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = build_config([task], person_id="")
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

        config = build_config([make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])], person_id="")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)

            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                (delivery_dir / "metadata").mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])])

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set("CT-0001", "collect", model="qwen", status="done",
                          jsonl_path="trajectories/qwen/CT-0001.jsonl")
                state.save()

                def fake_build_scoring_env(cfg):
                    return {"PATH": ""}

                async def fake_score_single(task, model_name, cfg, st, env):
                    raise RuntimeError("test explosion")

                with patch("ctpipe.score.build_validated_env", side_effect=fake_build_scoring_env), \
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

        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = build_config([task], person_id="")
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
        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = build_config([task], person_id="")
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
        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = build_config([task], person_id="")
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

        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = build_config([task], person_id="")
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

        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = build_config([task], person_id="")
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


class StatsTests(unittest.TestCase):
    """Tests for the stats subcommand (ctpipe.stats.show_stats)."""

    def _setup_delivery(self, tmpdir: str, tasks: list[TaskConfig]) -> tuple[BatchConfig, PipelineState]:
        temp_base = Path(tmpdir)
        # We need to patch base_dir so delivery_dir resolves under tmpdir.
        config = build_config(tasks, person_id="")
        delivery_dir = temp_base / f"delivery_{config.delivery_date}"
        delivery_dir.mkdir(parents=True, exist_ok=True)
        (delivery_dir / "metadata").mkdir(parents=True, exist_ok=True)
        write_task_manifest(config.task_manifest_path, tasks)
        state = PipelineState(delivery_dir / "pipeline_state.json")
        return config, state

    @patch.object(BatchConfig, "base_dir", new_callable=PropertyMock)
    def test_all_done_returns_true(self, mock_base: PropertyMock) -> None:
        """When all stages are done for every task, show_stats returns True."""
        from io import StringIO
        from ctpipe.stats import show_stats

        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_base.return_value = Path(tmpdir)
            config, state = self._setup_delivery(tmpdir, [task])

            state.set(task.id, "prepare", status="done")
            state.set(task.id, "run", model="qwen", status="done")
            state.set(task.id, "run", model="claude", status="done")
            state.set(task.id, "collect", model="qwen", status="done")
            state.set(task.id, "collect", model="claude", status="done")
            state.set(task.id, "score", model="qwen", status="done")
            state.set(task.id, "score", model="claude", status="done")
            state.set(task.id, "finalize", status="done", qwen_passrate=0.5, claude_passrate=0.8)
            state.set(task.id, "validate", status="done")
            state.save()

            result = show_stats(config, models=["qwen", "claude"], fmt="table")

        self.assertTrue(result)

    @patch.object(BatchConfig, "base_dir", new_callable=PropertyMock)
    def test_failures_return_false(self, mock_base: PropertyMock) -> None:
        """When any stage has failures, show_stats returns False."""
        from ctpipe.stats import show_stats

        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_base.return_value = Path(tmpdir)
            config, state = self._setup_delivery(tmpdir, [task])

            state.set(task.id, "prepare", status="done")
            state.set(task.id, "run", model="qwen", status="failed", error="timeout")
            state.set(task.id, "run", model="claude", status="done")
            state.save()

            result = show_stats(config, models=["qwen", "claude"], fmt="table")

        self.assertFalse(result)

    @patch.object(BatchConfig, "base_dir", new_callable=PropertyMock)
    def test_pending_returns_false(self, mock_base: PropertyMock) -> None:
        """When tasks have empty (pending) status, show_stats returns False."""
        from ctpipe.stats import show_stats

        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_base.return_value = Path(tmpdir)
            config, state = self._setup_delivery(tmpdir, [task])
            # Write no state at all — everything is pending.
            state.save()

            result = show_stats(config, models=["qwen", "claude"], fmt="table")

        self.assertFalse(result)

    @patch.object(BatchConfig, "base_dir", new_callable=PropertyMock)
    def test_json_format_output(self, mock_base: PropertyMock) -> None:
        """--format json should produce valid JSON with expected structure."""
        import sys
        from io import StringIO
        from ctpipe.stats import show_stats

        task = make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_base.return_value = Path(tmpdir)
            config, state = self._setup_delivery(tmpdir, [task])

            state.set(task.id, "prepare", status="done")
            state.set(task.id, "run", model="qwen", status="done")
            state.set(task.id, "run", model="claude", status="partial")
            state.set(task.id, "finalize", status="done", qwen_passrate=0.6, claude_passrate=0.85)
            state.save()

            captured = StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                show_stats(config, models=["qwen", "claude"], fmt="json")
            finally:
                sys.stdout = old_stdout

            output = captured.getvalue()
            data = json.loads(output)

        # Top-level keys: summary + per_task
        self.assertIn("summary", data)
        self.assertIn("per_task", data)
        summary = data["summary"]
        self.assertIn("stages", summary)
        self.assertIn("passrates", summary)
        self.assertIn("bottleneck", summary)
        self.assertIn("passrate_diff", summary)
        # Bottleneck key should report failed count (not not_done).
        self.assertIn("failed", summary["bottleneck"])
        # Stages should include prepare and run/qwen at minimum.
        stage_names = [s["stage"] for s in summary["stages"]]
        self.assertIn("prepare", stage_names)
        self.assertIn("run/qwen", stage_names)
        self.assertIn("run/claude", stage_names)
        # Verify counts are integers.
        prepare_row = next(s for s in summary["stages"] if s["stage"] == "prepare")
        self.assertEqual(prepare_row["done"], 1)
        self.assertEqual(prepare_row["failed"], 0)
        # per_task should contain the task with its stage statuses.
        per_task = data["per_task"]
        self.assertIn("CT-0001", per_task)
        self.assertEqual(per_task["CT-0001"]["prepare"], "done")
        self.assertEqual(per_task["CT-0001"]["run/qwen"], "done")
        self.assertEqual(per_task["CT-0001"]["run/claude"], "partial")
        # Passrate values should appear in per_task when set.
        self.assertAlmostEqual(per_task["CT-0001"]["qwen_passrate"], 0.6)
        self.assertAlmostEqual(per_task["CT-0001"]["claude_passrate"], 0.85)

    @patch.object(BatchConfig, "base_dir", new_callable=PropertyMock)
    def test_bottleneck_identifies_worst_stage(self, mock_base: PropertyMock) -> None:
        """Bottleneck should point to the stage with the most failures."""
        from ctpipe.stats import _collect_stage_counts, _find_bottleneck

        tasks = [make_task(f"CT-{i:04d}", followups_qwen=["f1"], followups_claude=["f1", "f2"]) for i in range(1, 6)]
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_base.return_value = Path(tmpdir)
            config, state = self._setup_delivery(tmpdir, tasks)

            # prepare: all done
            for t in tasks:
                state.set(t.id, "prepare", status="done")
            # run/qwen: 3 failed, 2 done
            for t in tasks[:3]:
                state.set(t.id, "run", model="qwen", status="failed")
            for t in tasks[3:]:
                state.set(t.id, "run", model="qwen", status="done")
            # run/claude: 1 failed, 4 done
            state.set(tasks[0].id, "run", model="claude", status="failed")
            for t in tasks[1:]:
                state.set(t.id, "run", model="claude", status="done")
            # collect/score: no failures, just pending — should NOT be bottleneck
            for t in tasks:
                state.set(t.id, "collect", model="qwen", status="")
                state.set(t.id, "collect", model="claude", status="")
                state.set(t.id, "score", model="qwen", status="")
                state.set(t.id, "score", model="claude", status="")
                state.set(t.id, "finalize", status="")
                state.set(t.id, "validate", status="")
            state.save()

            tids = [t.id for t in tasks]
            rows = _collect_stage_counts(state, tids, ["qwen", "claude"])
            stage_name, count = _find_bottleneck(rows)

        # run/qwen has 3 failures, more than any other stage.
        self.assertEqual(stage_name, "run/qwen")
        self.assertEqual(count, 3)

    @patch.object(BatchConfig, "base_dir", new_callable=PropertyMock)
    def test_bottleneck_ignores_pending_only_stages(self, mock_base: PropertyMock) -> None:
        """A stage with 0 failures but many pending should not be the bottleneck."""
        from ctpipe.stats import _collect_stage_counts, _find_bottleneck

        tasks = [make_task(f"CT-{i:04d}", followups_qwen=["f1"], followups_claude=["f1", "f2"]) for i in range(1, 4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_base.return_value = Path(tmpdir)
            config, state = self._setup_delivery(tmpdir, tasks)

            # run/qwen: 1 failure
            state.set(tasks[0].id, "run", model="qwen", status="failed")
            for t in tasks[1:]:
                state.set(t.id, "run", model="qwen", status="done")
            # Everything else pending (0 failures) — bottleneck must still be run/qwen.
            state.save()

            tids = [t.id for t in tasks]
            rows = _collect_stage_counts(state, tids, ["qwen", "claude"])
            stage_name, count = _find_bottleneck(rows)

        self.assertEqual(stage_name, "run/qwen")
        self.assertEqual(count, 1)

    def test_passrate_diff_basic(self) -> None:
        """Passrate diff should compute mean/median/std of (model_b - model_a)."""
        from ctpipe.stats import _collect_passrate_diff

        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            # 3 tasks with paired passrates.
            # diff = claude - qwen: 0.3, 0.4, 0.2 → mean=0.3, median=0.3
            state.set("CT-0001", "finalize", status="done", qwen_passrate=0.5, claude_passrate=0.8)
            state.set("CT-0002", "finalize", status="done", qwen_passrate=0.5, claude_passrate=0.9)
            state.set("CT-0003", "finalize", status="done", qwen_passrate=0.6, claude_passrate=0.8)
            state.save()

            result = _collect_passrate_diff(state, ["CT-0001", "CT-0002", "CT-0003"], "qwen", "claude")

        self.assertIsNotNone(result)
        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["mean"], 0.3, places=4)
        self.assertAlmostEqual(result["median"], 0.3, places=4)
        self.assertEqual(result["positive"], 3)
        self.assertEqual(result["negative"], 0)
        self.assertGreater(result["std"], 0)

    def test_passrate_diff_statistics_exact_values(self) -> None:
        """Verify mean/median/std with known input values.

        Diffs (claude - qwen): [0.10, 0.20, 0.30, 0.40]
        mean   = 0.25
        median = (0.20 + 0.30) / 2 = 0.25
        std    = sqrt(((0.15^2)*4) / 3) ≈ 0.1291
        """
        from ctpipe.stats import _collect_passrate_diff

        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            # qwen=0.5 for all, claude varies so diff = claude - 0.5
            state.set("CT-0001", "finalize", status="done", qwen_passrate=0.50, claude_passrate=0.60)
            state.set("CT-0002", "finalize", status="done", qwen_passrate=0.50, claude_passrate=0.70)
            state.set("CT-0003", "finalize", status="done", qwen_passrate=0.50, claude_passrate=0.80)
            state.set("CT-0004", "finalize", status="done", qwen_passrate=0.50, claude_passrate=0.90)
            state.save()

            result = _collect_passrate_diff(
                state, ["CT-0001", "CT-0002", "CT-0003", "CT-0004"], "qwen", "claude"
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["mean"], 0.25, places=4)
        self.assertAlmostEqual(result["median"], 0.25, places=4)
        # std = stdev([0.1, 0.2, 0.3, 0.4]) ≈ 0.1291
        self.assertAlmostEqual(result["std"], 0.1291, places=3)
        # All diffs are positive (claude > qwen).
        self.assertEqual(result["positive"], 4)
        self.assertEqual(result["negative"], 0)

    def test_passrate_diff_mixed_signs(self) -> None:
        """Verify positive/negative counts when claude sometimes loses."""
        from ctpipe.stats import _collect_passrate_diff

        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            # diff = claude - qwen: +0.20, -0.10, +0.05
            state.set("CT-0001", "finalize", status="done", qwen_passrate=0.50, claude_passrate=0.70)
            state.set("CT-0002", "finalize", status="done", qwen_passrate=0.60, claude_passrate=0.50)
            state.set("CT-0003", "finalize", status="done", qwen_passrate=0.40, claude_passrate=0.45)
            state.save()

            result = _collect_passrate_diff(
                state, ["CT-0001", "CT-0002", "CT-0003"], "qwen", "claude"
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["positive"], 2)  # +0.20, +0.05
        self.assertEqual(result["negative"], 1)  # -0.10
        # mean = (0.20 + -0.10 + 0.05) / 3 = 0.05
        self.assertAlmostEqual(result["mean"], 0.05, places=4)
        # sorted: [-0.10, 0.05, 0.20] → median = 0.05
        self.assertAlmostEqual(result["median"], 0.05, places=4)

    def test_missing_delivery_dir_no_crash(self) -> None:
        """show_stats should print a friendly message and return False when delivery dir is missing."""
        import sys
        from io import StringIO
        from unittest.mock import patch
        from ctpipe.config import BatchConfig
        from ctpipe.stats import show_stats

        config = build_config([make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])], person_id="")

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            # Sandbox base_dir to a fresh tmp so delivery_dir is guaranteed absent
            # (sibling tests share the same delivery_date and may create it in the repo).
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.object(BatchConfig, "base_dir", property(lambda self: Path(tmpdir))):
                    result = show_stats(config, models=["qwen", "claude"], fmt="table")
        finally:
            sys.stdout = old_stdout

        self.assertFalse(result)
        output = captured.getvalue()
        self.assertIn("not found", output.lower())
        self.assertIn("prepare", output.lower())

    def test_missing_delivery_dir_json_format(self) -> None:
        """Missing delivery dir should produce valid JSON with an error field."""
        import sys
        from io import StringIO
        from unittest.mock import patch
        from ctpipe.config import BatchConfig
        from ctpipe.stats import show_stats

        config = build_config([make_task(followups_qwen=["f1"], followups_claude=["f1", "f2"])], person_id="")

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.object(BatchConfig, "base_dir", property(lambda self: Path(tmpdir))):
                    result = show_stats(config, models=["qwen", "claude"], fmt="json")
        finally:
            sys.stdout = old_stdout

        self.assertFalse(result)
        data = json.loads(captured.getvalue())
        self.assertIn("error", data)
        self.assertIn("not found", data["error"].lower())

    def test_passrate_diff_no_pairs_returns_none(self) -> None:
        """Passrate diff should return None when no tasks have both models."""
        from ctpipe.stats import _collect_passrate_diff

        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "finalize", status="done", qwen_passrate=0.5)
            state.save()

            result = _collect_passrate_diff(state, ["CT-0001"], "qwen", "claude")

        self.assertIsNone(result)

    def test_passrate_diff_in_json_output(self) -> None:
        """JSON output should include passrate_diff when both models have data."""
        import sys
        from io import StringIO
        from ctpipe.stats import show_stats

        tasks = [make_task(f"CT-{i:04d}", followups_qwen=["f1"], followups_claude=["f1", "f2"]) for i in range(1, 3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = build_config(tasks, person_id="")
                delivery_dir = temp_base / f"delivery_{config.delivery_date}"
                delivery_dir.mkdir(parents=True, exist_ok=True)
                (delivery_dir / "metadata").mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, tasks)
                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set("CT-0001", "finalize", status="done", qwen_passrate=0.5, claude_passrate=0.8)
                state.set("CT-0002", "finalize", status="done", qwen_passrate=0.6, claude_passrate=0.85)
                state.save()

                captured = StringIO()
                old_stdout = sys.stdout
                sys.stdout = captured
                try:
                    show_stats(config, models=["qwen", "claude"], fmt="json")
                finally:
                    sys.stdout = old_stdout

                data = json.loads(captured.getvalue())

        diff = data["summary"]["passrate_diff"]
        self.assertIsNotNone(diff)
        self.assertEqual(diff["count"], 2)
        self.assertEqual(diff["model_a"], "qwen")
        self.assertEqual(diff["model_b"], "claude")
        # diffs: 0.3, 0.25 → mean=0.275
        self.assertAlmostEqual(diff["mean"], 0.275, places=3)

    @patch.object(BatchConfig, "base_dir", new_callable=PropertyMock)
    def test_passrate_stats_computed(self, mock_base: PropertyMock) -> None:
        """Passrate min/max/mean should be computed from finalize state."""
        from ctpipe.stats import _collect_passrate_stats

        tasks = [make_task(f"CT-{i:04d}", followups_qwen=["f1"], followups_claude=["f1", "f2"]) for i in range(1, 4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_base.return_value = Path(tmpdir)
            config, state = self._setup_delivery(tmpdir, tasks)

            state.set("CT-0001", "finalize", status="done", qwen_passrate=0.4, claude_passrate=0.8)
            state.set("CT-0002", "finalize", status="done", qwen_passrate=0.6, claude_passrate=0.9)
            state.set("CT-0003", "finalize", status="done", qwen_passrate=0.5, claude_passrate=0.85)
            state.save()

            tids = [t.id for t in tasks]
            result = _collect_passrate_stats(state, tids, ["qwen", "claude"])

        self.assertIn("qwen", result)
        self.assertIn("claude", result)
        self.assertAlmostEqual(result["qwen"]["min"], 0.4)
        self.assertAlmostEqual(result["qwen"]["max"], 0.6)
        self.assertAlmostEqual(result["qwen"]["mean"], 0.5)
        self.assertEqual(result["qwen"]["count"], 3)
        self.assertAlmostEqual(result["claude"]["min"], 0.8)


if __name__ == "__main__":
    unittest.main()
