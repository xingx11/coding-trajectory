"""Tests for ctpipe.retry — auto-retry of failed/partial pipeline tasks."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, PropertyMock, patch

from ctpipe.config import BatchConfig, ModelConfig, TaskConfig
from ctpipe.config import MODEL_SPECIFIC_STAGES
from ctpipe.retry import (
    _DOWNSTREAM,
    _Entry,
    _MODEL_AGNOSTIC_STAGES,
    _RETRYABLE_STATUSES,
    _STAGE_ORDER,
    _expand_cascade,
    _find_failed_entries,
    _get_retry_count,
    _get_status,
    _group_by_stage,
    _increment_retry_count,
    _mark_permanently_failed,
    _models_from_entries,
    _print_results,
    _print_round_summary,
    _reset_entries,
    _task_ids_from_entries,
    retry,
)
from ctpipe.state import PipelineState


# =========================================================================
# Helpers
# =========================================================================


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
        bad_pattern="lazy_shortcut",
    )


MODELS = ["qwen", "claude"]


# =========================================================================
# _Entry Tests
# =========================================================================


class EntryLabelTest(unittest.TestCase):
    """Test _Entry.label property formatting."""

    def test_label_without_model(self) -> None:
        entry = _Entry(task_id="CT-0001", stage="prepare")
        self.assertEqual(entry.label, "CT-0001/prepare")

    def test_label_with_model(self) -> None:
        entry = _Entry(task_id="CT-0001", stage="run", model="qwen")
        self.assertEqual(entry.label, "CT-0001/run/qwen")

    def test_label_with_none_model(self) -> None:
        entry = _Entry(task_id="CT-0001", stage="finalize", model=None)
        self.assertEqual(entry.label, "CT-0001/finalize")


class EntryEqualityTest(unittest.TestCase):
    """Test _Entry equality and hashing."""

    def test_equal_entries(self) -> None:
        a = _Entry("CT-0001", "run", "qwen")
        b = _Entry("CT-0001", "run", "qwen")
        self.assertEqual(a, b)

    def test_different_task_id(self) -> None:
        a = _Entry("CT-0001", "run", "qwen")
        b = _Entry("CT-0002", "run", "qwen")
        self.assertNotEqual(a, b)

    def test_different_stage(self) -> None:
        a = _Entry("CT-0001", "run", "qwen")
        b = _Entry("CT-0001", "collect", "qwen")
        self.assertNotEqual(a, b)

    def test_different_model(self) -> None:
        a = _Entry("CT-0001", "run", "qwen")
        b = _Entry("CT-0001", "run", "claude")
        self.assertNotEqual(a, b)

    def test_eq_with_non_entry_returns_not_implemented(self) -> None:
        entry = _Entry("CT-0001", "run", "qwen")
        self.assertEqual(entry.__eq__("not an entry"), NotImplemented)

    def test_hash_equal_entries(self) -> None:
        a = _Entry("CT-0001", "run", "qwen")
        b = _Entry("CT-0001", "run", "qwen")
        self.assertEqual(hash(a), hash(b))

    def test_hash_different_entries(self) -> None:
        a = _Entry("CT-0001", "run", "qwen")
        b = _Entry("CT-0002", "run", "claude")
        # Hashes *could* collide, but extremely unlikely for distinct tuples.
        self.assertNotEqual(hash(a), hash(b))

    def test_entries_usable_in_set(self) -> None:
        entries = {
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "run", "qwen"),  # duplicate
            _Entry("CT-0001", "run", "claude"),
        }
        self.assertEqual(len(entries), 2)


# =========================================================================
# _get_status / _get_retry_count Tests
# =========================================================================


class GetStatusTest(unittest.TestCase):
    """Test _get_status reads state correctly."""

    def test_returns_status_from_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen", status="done")
            entry = _Entry("CT-0001", "run", "qwen")
            self.assertEqual(_get_status(state, entry), "done")

    def test_returns_empty_string_for_missing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            entry = _Entry("CT-0001", "run", "qwen")
            self.assertEqual(_get_status(state, entry), "")

    def test_model_agnostic_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "prepare", status="done")
            entry = _Entry("CT-0001", "prepare")
            self.assertEqual(_get_status(state, entry), "done")


class GetRetryCountTest(unittest.TestCase):
    """Test _get_retry_count reads correctly with defaults."""

    def test_returns_retry_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen",
                      status="failed", retry_count=3)
            entry = _Entry("CT-0001", "run", "qwen")
            self.assertEqual(_get_retry_count(state, entry), 3)

    def test_default_retry_count_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen", status="failed")
            entry = _Entry("CT-0001", "run", "qwen")
            self.assertEqual(_get_retry_count(state, entry), 0)

    def test_missing_entry_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            entry = _Entry("CT-9999", "run", "qwen")
            self.assertEqual(_get_retry_count(state, entry), 0)


# =========================================================================
# _find_failed_entries Tests
# =========================================================================


class FindFailedEntriesTest(unittest.TestCase):
    """Test scanning state for retryable entries."""

    def _setup_state_and_tasks(
        self, tmpdir: str, task_ids: list[str]
    ) -> tuple[PipelineState, list[TaskConfig]]:
        state = PipelineState(Path(tmpdir) / "state.json")
        tasks = [_make_task(tid) for tid in task_ids]
        return state, tasks

    def test_finds_failed_model_specific_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state, tasks = self._setup_state_and_tasks(tmpdir, ["CT-0001"])
            state.set("CT-0001", "run", model="qwen", status="failed")
            state.set("CT-0001", "run", model="claude", status="done")

            entries = _find_failed_entries(state, tasks, None, MODELS)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0], _Entry("CT-0001", "run", "qwen"))

    def test_finds_partial_and_draft_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state, tasks = self._setup_state_and_tasks(tmpdir, ["CT-0001"])
            state.set("CT-0001", "run", model="qwen", status="partial")
            state.set("CT-0001", "run", model="claude", status="draft")

            entries = _find_failed_entries(state, tasks, None, MODELS)
            labels = {e.label for e in entries}
            self.assertIn("CT-0001/run/qwen", labels)
            self.assertIn("CT-0001/run/claude", labels)

    def test_skips_done_and_pending_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state, tasks = self._setup_state_and_tasks(tmpdir, ["CT-0001"])
            state.set("CT-0001", "run", model="qwen", status="done")
            state.set("CT-0001", "run", model="claude", status="pending")

            entries = _find_failed_entries(state, tasks, None, MODELS)
            self.assertEqual(entries, [])

    def test_finds_model_agnostic_failed_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state, tasks = self._setup_state_and_tasks(tmpdir, ["CT-0001"])
            state.set("CT-0001", "prepare", status="failed")

            entries = _find_failed_entries(state, tasks, None, MODELS)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0], _Entry("CT-0001", "prepare"))
            self.assertIsNone(entries[0].model)

    def test_respects_stage_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state, tasks = self._setup_state_and_tasks(tmpdir, ["CT-0001"])
            state.set("CT-0001", "run", model="qwen", status="failed")
            state.set("CT-0001", "collect", model="qwen", status="failed")

            entries = _find_failed_entries(state, tasks, ["run"], MODELS)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].stage, "run")

    def test_results_ordered_by_stage_then_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state, tasks = self._setup_state_and_tasks(
                tmpdir, ["CT-0002", "CT-0001"]
            )
            state.set("CT-0001", "run", model="qwen", status="failed")
            state.set("CT-0002", "prepare", status="failed")
            state.set("CT-0001", "prepare", status="failed")

            entries = _find_failed_entries(state, tasks, None, MODELS)
            # prepare comes before run, and within prepare CT-0001 < CT-0002
            stages = [e.stage for e in entries]
            self.assertEqual(stages[0], "prepare")
            self.assertEqual(stages[1], "prepare")
            self.assertEqual(stages[2], "run")

    def test_multiple_tasks_all_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state, tasks = self._setup_state_and_tasks(
                tmpdir, ["CT-0001", "CT-0002"]
            )
            state.set("CT-0001", "run", model="qwen", status="failed")
            state.set("CT-0002", "run", model="qwen", status="failed")

            entries = _find_failed_entries(state, tasks, None, MODELS)
            self.assertEqual(len(entries), 2)

    def test_empty_when_no_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            entries = _find_failed_entries(state, [], None, MODELS)
            self.assertEqual(entries, [])


# =========================================================================
# _expand_cascade Tests
# =========================================================================


# -------------------------------------------------------------------------
# prepare → 扇出到所有模型（model=None 的分支）
# -------------------------------------------------------------------------


class ExpandCascadePrepareFanOutTest(unittest.TestCase):
    """prepare 失败时，下游 model-specific stage 展开到每一个 model。

    这是 _expand_cascade 最复杂的分支：entry.model 为 None，
    下游 model-specific stage 必须对 models 列表做 for 循环，
    然后 continue 跳过末尾的单条赋值。
    """

    def test_prepare_fanout_exact_count(self) -> None:
        """prepare → 1 prepare + 3 stages × 2 models + 1 finalize = 8."""
        entries = [_Entry("CT-0001", "prepare")]
        expanded = _expand_cascade(entries, MODELS)
        self.assertEqual(len(expanded), 8)

    def test_prepare_fanout_exact_labels(self) -> None:
        """验证展开后的 8 个 entry 每一条都精确存在。"""
        entries = [_Entry("CT-0001", "prepare")]
        expanded = _expand_cascade(entries, MODELS)
        labels = {e.label for e in expanded}
        expected = {
            "CT-0001/prepare",        # model-agnostic 自身
            "CT-0001/run/qwen",       # run 扇出
            "CT-0001/run/claude",
            "CT-0001/collect/qwen",   # collect 扇出
            "CT-0001/collect/claude",
            "CT-0001/score/qwen",     # score 扇出
            "CT-0001/score/claude",
            "CT-0001/finalize",       # model-agnostic 下游，不扇出
        }
        self.assertEqual(labels, expected)

    def test_prepare_finalize_not_fanned_out(self) -> None:
        """finalize 是 model-agnostic，不应按 model 展开。"""
        entries = [_Entry("CT-0001", "prepare")]
        expanded = _expand_cascade(entries, MODELS)
        finalize_entries = [e for e in expanded if e.stage == "finalize"]
        self.assertEqual(len(finalize_entries), 1)
        self.assertIsNone(finalize_entries[0].model)

    def test_prepare_fanout_three_models(self) -> None:
        """三个 model 时: 1 + 3×3 + 1 = 11。"""
        three_models = ["qwen", "claude", "gpt"]
        entries = [_Entry("CT-0001", "prepare")]
        expanded = _expand_cascade(entries, three_models)
        self.assertEqual(len(expanded), 11)
        for model in three_models:
            for stage in ("run", "collect", "score"):
                self.assertIn(
                    _Entry("CT-0001", stage, model), expanded,
                    f"missing CT-0001/{stage}/{model}",
                )

    def test_prepare_fanout_single_model(self) -> None:
        """单个 model: 1 + 3×1 + 1 = 5。"""
        entries = [_Entry("CT-0001", "prepare")]
        expanded = _expand_cascade(entries, ["qwen"])
        self.assertEqual(len(expanded), 5)
        labels = {e.label for e in expanded}
        self.assertEqual(labels, {
            "CT-0001/prepare",
            "CT-0001/run/qwen",
            "CT-0001/collect/qwen",
            "CT-0001/score/qwen",
            "CT-0001/finalize",
        })

    def test_prepare_fanout_each_model_specific_entry_has_correct_model(self) -> None:
        """扇出后每个 model-specific entry 的 .model 字段准确。"""
        entries = [_Entry("CT-0001", "prepare")]
        expanded = _expand_cascade(entries, MODELS)
        for e in expanded:
            if e.stage in MODEL_SPECIFIC_STAGES:
                self.assertIn(e.model, MODELS)
            else:
                self.assertIsNone(e.model)

    def test_prepare_multiple_tasks_fanout_independently(self) -> None:
        """两个 task 的 prepare 各自独立扇出，互不干扰。"""
        entries = [
            _Entry("CT-0001", "prepare"),
            _Entry("CT-0002", "prepare"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        # 每个 task 8 个 entry，总共 16 个
        self.assertEqual(len(expanded), 16)
        for tid in ("CT-0001", "CT-0002"):
            self.assertIn(_Entry(tid, "prepare"), expanded)
            for model in MODELS:
                for stage in ("run", "collect", "score"):
                    self.assertIn(_Entry(tid, stage, model), expanded)
            self.assertIn(_Entry(tid, "finalize"), expanded)

    def test_prepare_fanout_sorted_correctly(self) -> None:
        """prepare 扇出后排序: stage_rank → task_id → model。"""
        entries = [_Entry("CT-0001", "prepare")]
        expanded = _expand_cascade(entries, MODELS)
        labels = [e.label for e in expanded]
        self.assertEqual(labels, [
            "CT-0001/prepare",         # stage 0
            "CT-0001/run/claude",      # stage 1, claude < qwen
            "CT-0001/run/qwen",
            "CT-0001/collect/claude",  # stage 2
            "CT-0001/collect/qwen",
            "CT-0001/score/claude",    # stage 3
            "CT-0001/score/qwen",
            "CT-0001/finalize",        # stage 4
        ])


# -------------------------------------------------------------------------
# model-specific entry 的级联（entry.model 非 None，走携带分支）
# -------------------------------------------------------------------------


class ExpandCascadeModelCarryTest(unittest.TestCase):
    """model-specific entry 级联时只携带自身 model，不扇出。"""

    def test_run_qwen_cascades_only_qwen(self) -> None:
        """run/qwen → collect/qwen + score/qwen + finalize，不涉及 claude。"""
        entries = [_Entry("CT-0001", "run", "qwen")]
        expanded = _expand_cascade(entries, MODELS)
        labels = {e.label for e in expanded}
        self.assertEqual(labels, {
            "CT-0001/run/qwen",
            "CT-0001/collect/qwen",
            "CT-0001/score/qwen",
            "CT-0001/finalize",
        })
        # 不应出现 claude
        for e in expanded:
            self.assertNotEqual(e.model, "claude")

    def test_collect_claude_cascades_only_claude(self) -> None:
        """collect/claude → score/claude + finalize。"""
        entries = [_Entry("CT-0001", "collect", "claude")]
        expanded = _expand_cascade(entries, MODELS)
        labels = {e.label for e in expanded}
        self.assertEqual(labels, {
            "CT-0001/collect/claude",
            "CT-0001/score/claude",
            "CT-0001/finalize",
        })

    def test_score_cascades_only_to_finalize(self) -> None:
        """score/qwen → finalize，不回溯到 run/collect。"""
        entries = [_Entry("CT-0001", "score", "qwen")]
        expanded = _expand_cascade(entries, MODELS)
        labels = {e.label for e in expanded}
        self.assertEqual(labels, {
            "CT-0001/score/qwen",
            "CT-0001/finalize",
        })

    def test_run_claude_does_not_touch_qwen(self) -> None:
        """run/claude 级联不涉及 qwen 的下游。"""
        entries = [_Entry("CT-0001", "run", "claude")]
        expanded = _expand_cascade(entries, MODELS)
        qwen_entries = [e for e in expanded if e.model == "qwen"]
        self.assertEqual(qwen_entries, [])

    def test_both_models_run_independently(self) -> None:
        """两个 model 的 run 同时失败，各自级联互不干扰。"""
        entries = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "run", "claude"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        labels = {e.label for e in expanded}
        # qwen path
        self.assertIn("CT-0001/collect/qwen", labels)
        self.assertIn("CT-0001/score/qwen", labels)
        # claude path
        self.assertIn("CT-0001/collect/claude", labels)
        self.assertIn("CT-0001/score/claude", labels)
        # finalize 只出现一次
        finalize_entries = [e for e in expanded if e.stage == "finalize"]
        self.assertEqual(len(finalize_entries), 1)


# -------------------------------------------------------------------------
# 去重测试
# -------------------------------------------------------------------------


class ExpandCascadeDedupTest(unittest.TestCase):
    """_expand_cascade 使用 dict[label] 去重，覆盖多种场景。"""

    def test_identical_input_entries_deduped(self) -> None:
        """输入有完全相同的两条 entry → 结果只保留一条。"""
        entries = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "run", "qwen"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        labels = [e.label for e in expanded]
        self.assertEqual(labels.count("CT-0001/run/qwen"), 1)

    def test_chain_overlap_dedup(self) -> None:
        """run 和 collect 同时在输入中: collect 既是 run 的下游也是输入自身。"""
        entries = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "collect", "qwen"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        labels = [e.label for e in expanded]
        self.assertEqual(labels.count("CT-0001/collect/qwen"), 1)
        self.assertEqual(labels.count("CT-0001/score/qwen"), 1)
        self.assertEqual(labels.count("CT-0001/finalize"), 1)

    def test_chain_overlap_total_count(self) -> None:
        """run + collect 重叠后的精确条目数: run + collect + score + finalize = 4。"""
        entries = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "collect", "qwen"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        self.assertEqual(len(expanded), 4)
        labels = {e.label for e in expanded}
        self.assertEqual(labels, {
            "CT-0001/run/qwen",
            "CT-0001/collect/qwen",
            "CT-0001/score/qwen",
            "CT-0001/finalize",
        })

    def test_prepare_plus_run_same_task_dedup(self) -> None:
        """prepare 扇出到 run/qwen，同时 run/qwen 也在输入中 → run/qwen 不重复。"""
        entries = [
            _Entry("CT-0001", "prepare"),
            _Entry("CT-0001", "run", "qwen"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        labels = [e.label for e in expanded]
        self.assertEqual(labels.count("CT-0001/run/qwen"), 1)
        # 总条目 = prepare 扇出的 8 条（run/qwen 被覆盖但数量不变）
        self.assertEqual(len(expanded), 8)

    def test_prepare_plus_both_runs_dedup(self) -> None:
        """prepare + run/qwen + run/claude: 所有 run 条目去重。"""
        entries = [
            _Entry("CT-0001", "prepare"),
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "run", "claude"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        labels = [e.label for e in expanded]
        # run/collect/score 的 qwen 和 claude 各只出现一次
        for model in MODELS:
            for stage in ("run", "collect", "score"):
                self.assertEqual(
                    labels.count(f"CT-0001/{stage}/{model}"), 1,
                    f"CT-0001/{stage}/{model} appeared more than once",
                )
        # 总数和单独 prepare 一样: 8
        self.assertEqual(len(expanded), 8)

    def test_finalize_dedup_from_multiple_sources(self) -> None:
        """run 和 score 都级联到 finalize → finalize 只出现一次。"""
        entries = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "score", "claude"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        finalize_entries = [e for e in expanded if e.stage == "finalize"]
        self.assertEqual(len(finalize_entries), 1)

    def test_three_stage_chain_full_overlap(self) -> None:
        """run + collect + score 全部在输入中 → 完全去重。"""
        entries = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "collect", "qwen"),
            _Entry("CT-0001", "score", "qwen"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        self.assertEqual(len(expanded), 4)
        labels = {e.label for e in expanded}
        self.assertEqual(labels, {
            "CT-0001/run/qwen",
            "CT-0001/collect/qwen",
            "CT-0001/score/qwen",
            "CT-0001/finalize",
        })

    def test_prepare_plus_full_chain_dedup(self) -> None:
        """prepare + run/qwen + collect/qwen + score/qwen 全部重叠。"""
        entries = [
            _Entry("CT-0001", "prepare"),
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "collect", "qwen"),
            _Entry("CT-0001", "score", "qwen"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        labels = [e.label for e in expanded]
        # 所有条目去重: 结果和单独 prepare 完全一致
        self.assertEqual(len(expanded), 8)
        for label in labels:
            self.assertEqual(labels.count(label), 1, f"{label} duplicated")

    def test_cross_task_no_dedup(self) -> None:
        """不同 task 的同名 stage/model 不应被去重。"""
        entries = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0002", "run", "qwen"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        run_entries = [e for e in expanded if e.stage == "run"]
        self.assertEqual(len(run_entries), 2)
        task_ids = {e.task_id for e in run_entries}
        self.assertEqual(task_ids, {"CT-0001", "CT-0002"})

    def test_last_write_wins_on_duplicate(self) -> None:
        """相同 label 的 entry 在 dict 中 last-write-wins。

        当 prepare 和 run/qwen 都在输入中时，run/qwen 先由 prepare
        扇出生成，后被 run/qwen 输入条目覆盖。
        """
        original_run = _Entry("CT-0001", "run", "qwen")
        entries = [
            _Entry("CT-0001", "prepare"),
            original_run,
        ]
        expanded = _expand_cascade(entries, MODELS)
        run_entry = [e for e in expanded if e.label == "CT-0001/run/qwen"][0]
        # 输入的 run entry 覆盖 prepare 扇出的 run entry
        self.assertIs(run_entry, original_run)


# -------------------------------------------------------------------------
# 排序测试
# -------------------------------------------------------------------------


class ExpandCascadeSortTest(unittest.TestCase):
    """_expand_cascade 输出按 (stage_rank, task_id, model) 三级排序。"""

    def test_sort_by_stage_rank(self) -> None:
        """不同 stage 按 _STAGE_ORDER 排序。"""
        entries = [
            _Entry("CT-0001", "score", "qwen"),
            _Entry("CT-0001", "run", "qwen"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        stages = [e.stage for e in expanded]
        stage_rank = {s: i for i, s in enumerate(_STAGE_ORDER)}
        rank_values = [stage_rank[s] for s in stages]
        self.assertEqual(rank_values, sorted(rank_values))

    def test_sort_by_task_id_within_stage(self) -> None:
        """同 stage 内按 task_id 字典序排列。"""
        entries = [
            _Entry("CT-0003", "run", "qwen"),
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0002", "run", "qwen"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        run_entries = [e for e in expanded if e.stage == "run"]
        task_ids = [e.task_id for e in run_entries]
        self.assertEqual(task_ids, ["CT-0001", "CT-0002", "CT-0003"])

    def test_sort_by_model_within_stage_and_task(self) -> None:
        """同 stage 同 task_id 内按 model 字典序排列。"""
        entries = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "run", "claude"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        run_entries = [e for e in expanded if e.stage == "run"]
        models = [e.model for e in run_entries]
        self.assertEqual(models, ["claude", "qwen"])

    def test_sort_none_model_before_string_model(self) -> None:
        """model=None 排在任何 model 名称之前（None or '' → '' < 'claude'）。"""
        entries = [
            _Entry("CT-0001", "prepare"),
            _Entry("CT-0001", "run", "claude"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        # prepare (stage 0, model=None) 排在 run (stage 1) 之前
        self.assertEqual(expanded[0].stage, "prepare")
        self.assertIsNone(expanded[0].model)

    def test_sort_multi_task_multi_model(self) -> None:
        """多任务多模型的完整排序验证。"""
        entries = [
            _Entry("CT-0002", "run", "qwen"),
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0002", "run", "claude"),
            _Entry("CT-0001", "run", "claude"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        run_entries = [e for e in expanded if e.stage == "run"]
        keys = [(e.task_id, e.model) for e in run_entries]
        self.assertEqual(keys, [
            ("CT-0001", "claude"),
            ("CT-0001", "qwen"),
            ("CT-0002", "claude"),
            ("CT-0002", "qwen"),
        ])

    def test_input_order_does_not_affect_output(self) -> None:
        """无论输入顺序如何，输出排序一致。"""
        entries_a = [
            _Entry("CT-0001", "score", "qwen"),
            _Entry("CT-0001", "prepare"),
            _Entry("CT-0001", "run", "qwen"),
        ]
        entries_b = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "prepare"),
            _Entry("CT-0001", "score", "qwen"),
        ]
        labels_a = [e.label for e in _expand_cascade(entries_a, MODELS)]
        labels_b = [e.label for e in _expand_cascade(entries_b, MODELS)]
        self.assertEqual(labels_a, labels_b)


# -------------------------------------------------------------------------
# 边界情况
# -------------------------------------------------------------------------


class ExpandCascadeEdgeCaseTest(unittest.TestCase):
    """_expand_cascade 的边界和特殊情况。"""

    def test_empty_input(self) -> None:
        entries: list[_Entry] = []
        expanded = _expand_cascade(entries, MODELS)
        self.assertEqual(expanded, [])

    def test_finalize_no_downstream(self) -> None:
        """finalize 没有下游 stage，只返回自身。"""
        entries = [_Entry("CT-0001", "finalize")]
        expanded = _expand_cascade(entries, MODELS)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0], _Entry("CT-0001", "finalize"))

    def test_collect_and_score_from_different_models(self) -> None:
        """不同 model 的 collect 和 score 各自独立级联。"""
        entries = [
            _Entry("CT-0001", "collect", "qwen"),
            _Entry("CT-0001", "score", "claude"),
        ]
        expanded = _expand_cascade(entries, MODELS)
        labels = {e.label for e in expanded}
        # collect/qwen → score/qwen + finalize
        self.assertIn("CT-0001/collect/qwen", labels)
        self.assertIn("CT-0001/score/qwen", labels)
        # score/claude → finalize (score/claude 已在输入中)
        self.assertIn("CT-0001/score/claude", labels)
        self.assertIn("CT-0001/finalize", labels)
        # 不应出现 collect/claude 或 score/qwen 以外的交叉
        self.assertNotIn("CT-0001/collect/claude", labels)
        self.assertNotIn("CT-0001/run/qwen", labels)

    def test_prepare_fanout_does_not_produce_extra_finalize(self) -> None:
        """prepare 扇出到 model-specific 时走 continue 分支，
        不会在 model-agnostic 分支额外生成 finalize。"""
        entries = [_Entry("CT-0001", "prepare")]
        expanded = _expand_cascade(entries, MODELS)
        finalize_entries = [e for e in expanded if e.stage == "finalize"]
        # finalize 来自 _DOWNSTREAM["prepare"] 中的 model-agnostic 分支
        self.assertEqual(len(finalize_entries), 1)

    def test_models_list_controls_fanout(self) -> None:
        """扇出的 model 数量完全由 models 参数决定。"""
        five_models = ["a", "b", "c", "d", "e"]
        entries = [_Entry("CT-0001", "prepare")]
        expanded = _expand_cascade(entries, five_models)
        for stage in ("run", "collect", "score"):
            stage_entries = [e for e in expanded if e.stage == stage]
            self.assertEqual(len(stage_entries), 5)
            models_seen = {e.model for e in stage_entries}
            self.assertEqual(models_seen, set(five_models))


# =========================================================================
# _reset_entries Tests
# =========================================================================


class ResetEntriesTest(unittest.TestCase):
    """Test state reset for entries."""

    def test_reset_deletes_entry_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen", status="failed",
                      error="some error")

            entries = [_Entry("CT-0001", "run", "qwen")]
            count = _reset_entries(state, entries)

            self.assertEqual(count, 1)
            self.assertEqual(_get_status(state, _Entry("CT-0001", "run", "qwen")), "")

    def test_reset_model_agnostic_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "prepare", status="failed")

            entries = [_Entry("CT-0001", "prepare")]
            count = _reset_entries(state, entries)

            self.assertEqual(count, 1)
            self.assertEqual(_get_status(state, _Entry("CT-0001", "prepare")), "")

    def test_reset_nonexistent_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            entries = [_Entry("CT-0001", "run", "qwen")]
            count = _reset_entries(state, entries)
            self.assertEqual(count, 0)

    def test_reset_multiple_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen", status="failed")
            state.set("CT-0001", "run", model="claude", status="failed")
            state.set("CT-0001", "prepare", status="failed")

            entries = [
                _Entry("CT-0001", "run", "qwen"),
                _Entry("CT-0001", "run", "claude"),
                _Entry("CT-0001", "prepare"),
            ]
            count = _reset_entries(state, entries)
            self.assertEqual(count, 3)


# =========================================================================
# _group_by_stage Tests
# =========================================================================


class GroupByStageTest(unittest.TestCase):
    """Test grouping entries by stage."""

    def test_groups_correctly(self) -> None:
        entries = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "run", "claude"),
            _Entry("CT-0001", "collect", "qwen"),
            _Entry("CT-0001", "prepare"),
        ]
        groups = _group_by_stage(entries)
        self.assertEqual(len(groups["prepare"]), 1)
        self.assertEqual(len(groups["run"]), 2)
        self.assertEqual(len(groups["collect"]), 1)
        self.assertNotIn("score", groups)
        self.assertNotIn("finalize", groups)

    def test_preserves_stage_order(self) -> None:
        entries = [
            _Entry("CT-0001", "finalize"),
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "prepare"),
        ]
        groups = _group_by_stage(entries)
        keys = list(groups.keys())
        self.assertEqual(keys, ["prepare", "run", "finalize"])

    def test_empty_entries(self) -> None:
        groups = _group_by_stage([])
        self.assertEqual(groups, {})

    def test_empty_stages_excluded(self) -> None:
        entries = [_Entry("CT-0001", "run", "qwen")]
        groups = _group_by_stage(entries)
        self.assertEqual(list(groups.keys()), ["run"])


# =========================================================================
# _task_ids_from_entries / _models_from_entries Tests
# =========================================================================


class TaskIdsFromEntriesTest(unittest.TestCase):
    """Test unique task ID extraction."""

    def test_unique_task_ids(self) -> None:
        entries = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0002", "run", "qwen"),
            _Entry("CT-0001", "collect", "qwen"),
        ]
        ids = _task_ids_from_entries(entries)
        self.assertEqual(ids, ["CT-0001", "CT-0002"])

    def test_preserves_order(self) -> None:
        entries = [
            _Entry("CT-0003", "run", "qwen"),
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0002", "run", "qwen"),
        ]
        ids = _task_ids_from_entries(entries)
        self.assertEqual(ids, ["CT-0003", "CT-0001", "CT-0002"])

    def test_empty_entries(self) -> None:
        self.assertEqual(_task_ids_from_entries([]), [])


class ModelsFromEntriesTest(unittest.TestCase):
    """Test unique model extraction."""

    def test_extracts_models_from_entries(self) -> None:
        entries = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "collect", "claude"),
        ]
        models = _models_from_entries(entries, MODELS)
        self.assertIn("qwen", models)
        self.assertIn("claude", models)

    def test_preserves_all_models_order(self) -> None:
        entries = [
            _Entry("CT-0001", "run", "claude"),
            _Entry("CT-0001", "run", "qwen"),
        ]
        models = _models_from_entries(entries, MODELS)
        # Should preserve order from all_models, filtered to seen
        self.assertEqual(models, ["qwen", "claude"])

    def test_no_model_entries_returns_all_models(self) -> None:
        entries = [_Entry("CT-0001", "prepare")]
        models = _models_from_entries(entries, MODELS)
        self.assertEqual(models, MODELS)

    def test_empty_entries_returns_all_models(self) -> None:
        models = _models_from_entries([], MODELS)
        self.assertEqual(models, MODELS)

    def test_only_seen_models_returned(self) -> None:
        all_models = ["qwen", "claude", "gpt"]
        entries = [_Entry("CT-0001", "run", "qwen")]
        models = _models_from_entries(entries, all_models)
        self.assertEqual(models, ["qwen"])


# =========================================================================
# _mark_permanently_failed Tests
# =========================================================================


class MarkPermanentlyFailedTest(unittest.TestCase):
    """Test permanent failure marking."""

    def test_marks_failed_entry_as_permanently_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen",
                      status="failed", retry_count=3, error="boom")

            entries = [_Entry("CT-0001", "run", "qwen")]
            marked = _mark_permanently_failed(state, entries)

            self.assertEqual(len(marked), 1)
            info = state.get("CT-0001", "run", "qwen")
            self.assertEqual(info["status"], "permanently_failed")
            self.assertEqual(info["retry_count"], 3)
            self.assertEqual(info["error"], "boom")

    def test_skips_done_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen", status="done")

            entries = [_Entry("CT-0001", "run", "qwen")]
            marked = _mark_permanently_failed(state, entries)

            self.assertEqual(len(marked), 0)

    def test_default_error_message_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen",
                      status="failed", retry_count=2)

            entries = [_Entry("CT-0001", "run", "qwen")]
            _mark_permanently_failed(state, entries)

            info = state.get("CT-0001", "run", "qwen")
            self.assertEqual(info["error"], "exceeded max retries")

    def test_marks_partial_and_draft_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen", status="partial")
            state.set("CT-0001", "run", model="claude", status="draft")

            entries = [
                _Entry("CT-0001", "run", "qwen"),
                _Entry("CT-0001", "run", "claude"),
            ]
            marked = _mark_permanently_failed(state, entries)
            self.assertEqual(len(marked), 2)

    def test_model_agnostic_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "prepare", status="failed", retry_count=5)

            entries = [_Entry("CT-0001", "prepare")]
            marked = _mark_permanently_failed(state, entries)
            self.assertEqual(len(marked), 1)
            info = state.get("CT-0001", "prepare")
            self.assertEqual(info["status"], "permanently_failed")


# =========================================================================
# _increment_retry_count Tests
# =========================================================================


class IncrementRetryCountTest(unittest.TestCase):
    """Test retry count incrementing."""

    def test_increments_existing_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen",
                      status="pending", retry_count=2)

            entries = [_Entry("CT-0001", "run", "qwen")]
            _increment_retry_count(state, entries)

            info = state.get("CT-0001", "run", "qwen")
            self.assertEqual(info["retry_count"], 3)
            self.assertEqual(info["status"], "pending")

    def test_starts_from_zero_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            # No prior state for this entry

            entries = [_Entry("CT-0001", "run", "qwen")]
            _increment_retry_count(state, entries)

            info = state.get("CT-0001", "run", "qwen")
            self.assertEqual(info["retry_count"], 1)
            self.assertEqual(info["status"], "pending")

    def test_increments_multiple_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen",
                      status="pending", retry_count=0)
            state.set("CT-0001", "run", model="claude",
                      status="pending", retry_count=5)

            entries = [
                _Entry("CT-0001", "run", "qwen"),
                _Entry("CT-0001", "run", "claude"),
            ]
            _increment_retry_count(state, entries)

            self.assertEqual(
                state.get("CT-0001", "run", "qwen")["retry_count"], 1
            )
            self.assertEqual(
                state.get("CT-0001", "run", "claude")["retry_count"], 6
            )


# =========================================================================
# _print_round_summary / _print_results Tests
# =========================================================================


class PrintRoundSummaryTest(unittest.TestCase):
    """Test round summary output."""

    def test_prints_round_info(self) -> None:
        entries = [
            _Entry("CT-0001", "run", "qwen"),
            _Entry("CT-0001", "collect", "qwen"),
        ]
        stage_groups = _group_by_stage(entries)

        with patch("builtins.print") as mock_print:
            _print_round_summary(1, 3, entries, stage_groups)

            printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("RETRY ROUND 1/3", printed)
            self.assertIn("2 entries", printed)

    def test_truncates_long_group_display(self) -> None:
        entries = [
            _Entry(f"CT-{i:04d}", "run", "qwen") for i in range(10)
        ]
        stage_groups = _group_by_stage(entries)

        with patch("builtins.print") as mock_print:
            _print_round_summary(1, 2, entries, stage_groups)

            printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("+5 more", printed)


class PrintResultsTest(unittest.TestCase):
    """Test retry results summary output."""

    def test_shows_succeeded_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen", status="done")
            state.set("CT-0001", "run", model="claude", status="failed")

            entries = [
                _Entry("CT-0001", "run", "qwen"),
                _Entry("CT-0001", "run", "claude"),
            ]

            with patch("builtins.print") as mock_print:
                _print_results(state, entries, [], [])

                printed = " ".join(str(c) for c in mock_print.call_args_list)
                self.assertIn("Succeeded : 1/2", printed)
                self.assertIn("Still failed: 1", printed)

    def test_shows_permanent_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen",
                      status="permanently_failed", error="gave up")

            entries = [_Entry("CT-0001", "run", "qwen")]
            perm_failed = [_Entry("CT-0001", "run", "qwen")]

            with patch("builtins.print") as mock_print:
                _print_results(state, entries, perm_failed, [])

                printed = " ".join(str(c) for c in mock_print.call_args_list)
                self.assertIn("PERMANENTLY FAILED", printed)
                self.assertIn("gave up", printed)

    def test_shows_exceptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = PipelineState(Path(tmpdir) / "state.json")
            state.set("CT-0001", "run", model="qwen", status="failed")

            entries = [_Entry("CT-0001", "run", "qwen")]
            errors = [(_Entry("CT-0001", "run", "qwen"), "stage crashed")]

            with patch("builtins.print") as mock_print:
                _print_results(state, entries, [], errors)

                printed = " ".join(str(c) for c in mock_print.call_args_list)
                self.assertIn("EXCEPTIONS during retry", printed)
                self.assertIn("stage crashed", printed)


# =========================================================================
# retry() Integration Tests
# =========================================================================


class RetryNothingToRetryTest(unittest.TestCase):
    """Test retry() when there are no failed entries."""

    def test_all_done_returns_true(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", status="done")
                for model in MODELS:
                    state.set(task.id, "run", model=model, status="done")
                    state.set(task.id, "collect", model=model, status="done")
                    state.set(task.id, "score", model=model, status="done")
                state.set(task.id, "finalize", status="done")

                result = asyncio.run(retry(config, max_retries=2))
                self.assertTrue(result)

    def test_no_tasks_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                result = asyncio.run(retry(config))
                self.assertFalse(result)

    def test_delivery_dir_not_found_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([_make_task()])
                # Do NOT create delivery_dir
                result = asyncio.run(retry(config))
                self.assertFalse(result)


class RetryDryRunTest(unittest.TestCase):
    """Test retry() dry_run mode."""

    def test_dry_run_does_not_modify_state(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "run", model="qwen",
                          status="failed", error="test error")
                state.set(task.id, "run", model="claude", status="done")
                state.set(task.id, "prepare", status="done")
                for model in MODELS:
                    state.set(task.id, "collect", model=model, status="done")
                    state.set(task.id, "score", model=model, status="done")
                state.set(task.id, "finalize", status="done")

                result = asyncio.run(
                    retry(config, dry_run=True, max_retries=2)
                )

                # State should be unchanged for the failed entry
                state.reload()
                info = state.get(task.id, "run", "qwen")
                self.assertEqual(info["status"], "failed")
                self.assertEqual(info["error"], "test error")
                self.assertFalse(result)


class RetryMaxRetriesExhaustedTest(unittest.TestCase):
    """Test retry() when max retries are exceeded."""

    def test_marks_permanently_failed_after_max_retries(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "run", model="qwen",
                          status="failed", retry_count=3,
                          error="persistent error")
                state.set(task.id, "run", model="claude", status="done")
                state.set(task.id, "prepare", status="done")
                for model in MODELS:
                    state.set(task.id, "collect", model=model, status="done")
                    state.set(task.id, "score", model=model, status="done")
                state.set(task.id, "finalize", status="done")

                result = asyncio.run(retry(config, max_retries=2))

                state.reload()
                info = state.get(task.id, "run", "qwen")
                self.assertEqual(info["status"], "permanently_failed")
                self.assertFalse(result)


class RetrySuccessfulTest(unittest.TestCase):
    """Test retry() with successful re-execution."""

    def test_retry_succeeds_after_one_round(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "run", model="qwen",
                          status="failed", error="transient")
                state.set(task.id, "run", model="claude", status="done")
                state.set(task.id, "prepare", status="done")
                for model in MODELS:
                    state.set(task.id, "collect", model=model, status="done")
                    state.set(task.id, "score", model=model, status="done")
                state.set(task.id, "finalize", status="done")

                async def fake_execute_stage(
                    stage, cfg, tids, models, tt, tot, st,
                ):
                    # Simulate successful re-execution: set all to done
                    for tid in tids:
                        for m in models:
                            if stage in MODEL_SPECIFIC_STAGES:
                                st.set(tid, stage, model=m, status="done")
                            else:
                                st.set(tid, stage, status="done")
                    st.reload()

                with patch(
                    "ctpipe.retry._execute_stage",
                    side_effect=fake_execute_stage,
                ):
                    result = asyncio.run(
                        retry(config, max_retries=2, cascade=False)
                    )

                self.assertTrue(result)


class RetryStageCrashTest(unittest.TestCase):
    """Test retry() when a stage crashes entirely."""

    def test_stage_crash_marks_all_entries_failed(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "run", model="qwen", status="failed")
                state.set(task.id, "run", model="claude", status="failed")
                state.set(task.id, "prepare", status="done")
                for model in MODELS:
                    state.set(task.id, "collect", model=model, status="done")
                    state.set(task.id, "score", model=model, status="done")
                state.set(task.id, "finalize", status="done")

                async def crashing_stage(*args, **kwargs):
                    raise RuntimeError("stage exploded")

                with patch(
                    "ctpipe.retry._execute_stage",
                    side_effect=crashing_stage,
                ):
                    result = asyncio.run(
                        retry(config, max_retries=2, cascade=False)
                    )

                self.assertFalse(result)

                state.reload()
                info_q = state.get(task.id, "run", "qwen")
                self.assertEqual(info_q["status"], "failed")
                self.assertIn("exploded", info_q["error"])


class RetryCascadeTest(unittest.TestCase):
    """Test retry() cascade behavior."""

    def test_cascade_expands_downstream_stages(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "run", model="qwen", status="failed")
                state.set(task.id, "run", model="claude", status="done")
                state.set(task.id, "prepare", status="done")
                state.set(task.id, "collect", model="qwen", status="done")
                state.set(task.id, "collect", model="claude", status="done")
                state.set(task.id, "score", model="qwen", status="done")
                state.set(task.id, "score", model="claude", status="done")
                state.set(task.id, "finalize", status="done")

                executed_stages: list[str] = []

                async def fake_execute_stage(
                    stage, cfg, tids, models, tt, tot, st,
                ):
                    executed_stages.append(stage)
                    for tid in tids:
                        for m in models:
                            if stage in MODEL_SPECIFIC_STAGES:
                                st.set(tid, stage, model=m, status="done")
                            else:
                                st.set(tid, stage, status="done")
                    st.reload()

                with patch(
                    "ctpipe.retry._execute_stage",
                    side_effect=fake_execute_stage,
                ):
                    result = asyncio.run(
                        retry(config, max_retries=2, cascade=True)
                    )

                # With cascade, run/qwen failing should cascade to
                # collect, score, finalize
                self.assertIn("run", executed_stages)
                self.assertIn("collect", executed_stages)
                self.assertIn("score", executed_stages)
                self.assertIn("finalize", executed_stages)


class RetryMultipleRoundsTest(unittest.TestCase):
    """Test retry() with multiple rounds."""

    def test_stops_early_when_no_more_failures(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "run", model="qwen", status="failed")
                state.set(task.id, "run", model="claude", status="done")
                state.set(task.id, "prepare", status="done")
                for model in MODELS:
                    state.set(task.id, "collect", model=model, status="done")
                    state.set(task.id, "score", model=model, status="done")
                state.set(task.id, "finalize", status="done")

                call_count = 0

                async def fake_execute_stage(
                    stage, cfg, tids, models, tt, tot, st,
                ):
                    nonlocal call_count
                    call_count += 1
                    # First call: succeed, setting everything to done
                    for tid in tids:
                        for m in models:
                            if stage in MODEL_SPECIFIC_STAGES:
                                st.set(tid, stage, model=m, status="done")
                            else:
                                st.set(tid, stage, status="done")
                    st.reload()

                with patch(
                    "ctpipe.retry._execute_stage",
                    side_effect=fake_execute_stage,
                ):
                    result = asyncio.run(
                        retry(config, max_retries=5, cascade=False)
                    )

                self.assertTrue(result)

    def test_mixed_retryable_and_exhausted(self) -> None:
        """One entry exhausted, one still retryable."""
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                # qwen has been retried 3 times (exhausted at max_retries=2)
                state.set(task.id, "run", model="qwen",
                          status="failed", retry_count=3,
                          error="persistent")
                # claude has only been retried once (still retryable)
                state.set(task.id, "run", model="claude",
                          status="failed", retry_count=1)
                state.set(task.id, "prepare", status="done")
                for model in MODELS:
                    state.set(task.id, "collect", model=model, status="done")
                    state.set(task.id, "score", model=model, status="done")
                state.set(task.id, "finalize", status="done")

                async def fake_execute_stage(
                    stage, cfg, tids, models, tt, tot, st,
                ):
                    for tid in tids:
                        for m in models:
                            if stage in MODEL_SPECIFIC_STAGES:
                                st.set(tid, stage, model=m, status="done")
                            else:
                                st.set(tid, stage, status="done")
                    st.reload()

                with patch(
                    "ctpipe.retry._execute_stage",
                    side_effect=fake_execute_stage,
                ):
                    result = asyncio.run(
                        retry(config, max_retries=2, cascade=False)
                    )

                state.reload()
                # qwen should be permanently_failed
                info_q = state.get(task.id, "run", "qwen")
                self.assertEqual(info_q["status"], "permanently_failed")
                # Overall result should be False because of permanent failure
                self.assertFalse(result)


class RetryWithStageFilterTest(unittest.TestCase):
    """Test retry() with stage filter."""

    def test_only_retries_specified_stages(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                # Both run and collect are failed, but we only retry "run"
                state.set(task.id, "run", model="qwen", status="failed")
                state.set(task.id, "collect", model="qwen", status="failed")
                state.set(task.id, "run", model="claude", status="done")
                state.set(task.id, "collect", model="claude", status="done")
                state.set(task.id, "prepare", status="done")
                state.set(task.id, "score", model="qwen", status="done")
                state.set(task.id, "score", model="claude", status="done")
                state.set(task.id, "finalize", status="done")

                executed_stages: list[str] = []

                async def fake_execute_stage(
                    stage, cfg, tids, models, tt, tot, st,
                ):
                    executed_stages.append(stage)
                    for tid in tids:
                        for m in models:
                            st.set(tid, stage, model=m, status="done")
                    st.reload()

                with patch(
                    "ctpipe.retry._execute_stage",
                    side_effect=fake_execute_stage,
                ):
                    asyncio.run(
                        retry(
                            config,
                            stages=["run"],
                            max_retries=1,
                            cascade=False,
                        )
                    )

                self.assertIn("run", executed_stages)
                self.assertNotIn("collect", executed_stages)


class RetryIndividualEntryFailureTest(unittest.TestCase):
    """Test retry() when individual entries still fail after re-execution."""

    def test_individual_entry_still_failed_after_retry(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "run", model="qwen", status="failed")
                state.set(task.id, "run", model="claude", status="done")
                state.set(task.id, "prepare", status="done")
                for model in MODELS:
                    state.set(task.id, "collect", model=model, status="done")
                    state.set(task.id, "score", model=model, status="done")
                state.set(task.id, "finalize", status="done")

                async def partial_execute_stage(
                    stage, cfg, tids, models, tt, tot, st,
                ):
                    # qwen still fails, claude succeeds
                    for tid in tids:
                        for m in models:
                            if m == "qwen":
                                st.set(tid, stage, model=m,
                                       status="failed", error="still broken")
                            else:
                                st.set(tid, stage, model=m, status="done")
                    st.reload()

                with patch(
                    "ctpipe.retry._execute_stage",
                    side_effect=partial_execute_stage,
                ):
                    result = asyncio.run(
                        retry(config, max_retries=2, cascade=False)
                    )

                # qwen still failed → overall False
                self.assertFalse(result)


class RetryCustomModelsTest(unittest.TestCase):
    """Test retry() with custom model list."""

    def test_custom_models_list(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=tmp,
            ):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                custom_models = ["gpt", "gemini"]
                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "run", model="gpt", status="failed")
                state.set(task.id, "run", model="gemini", status="done")
                state.set(task.id, "prepare", status="done")
                for model in custom_models:
                    state.set(task.id, "collect", model=model, status="done")
                    state.set(task.id, "score", model=model, status="done")
                state.set(task.id, "finalize", status="done")

                captured_models: list[list[str]] = []

                async def fake_execute_stage(
                    stage, cfg, tids, models, tt, tot, st,
                ):
                    captured_models.append(list(models))
                    for tid in tids:
                        for m in models:
                            if stage in MODEL_SPECIFIC_STAGES:
                                st.set(tid, stage, model=m, status="done")
                            else:
                                st.set(tid, stage, status="done")
                    st.reload()

                with patch(
                    "ctpipe.retry._execute_stage",
                    side_effect=fake_execute_stage,
                ):
                    result = asyncio.run(
                        retry(
                            config,
                            models=custom_models,
                            max_retries=1,
                            cascade=False,
                        )
                    )

                self.assertTrue(result)
                # The executed models should include "gpt" (the failed one)
                self.assertTrue(
                    any("gpt" in m_list for m_list in captured_models)
                )


# =========================================================================
# Constants Sanity Tests
# =========================================================================


class ConstantsSanityTest(unittest.TestCase):
    """Verify module constants are consistent."""

    def test_stage_order_covers_all_stages(self) -> None:
        all_stages = set(_MODEL_AGNOSTIC_STAGES) | set(MODEL_SPECIFIC_STAGES)
        self.assertEqual(set(_STAGE_ORDER), all_stages)

    def test_downstream_keys_match_stage_order(self) -> None:
        self.assertEqual(set(_DOWNSTREAM.keys()), set(_STAGE_ORDER))

    def test_retryable_statuses(self) -> None:
        self.assertEqual(_RETRYABLE_STATUSES, {"failed", "partial", "draft"})

    def test_model_agnostic_stages_not_in_model_specific(self) -> None:
        self.assertEqual(
            set(_MODEL_AGNOSTIC_STAGES) & set(MODEL_SPECIFIC_STAGES),
            set(),
        )

    def test_downstream_finalize_is_empty(self) -> None:
        self.assertEqual(_DOWNSTREAM["finalize"], [])

    def test_downstream_prepare_includes_all_later_stages(self) -> None:
        self.assertEqual(
            set(_DOWNSTREAM["prepare"]),
            {"run", "collect", "score", "finalize"},
        )


# =========================================================================
# Entry Frozen Dataclass Test
# =========================================================================


class EntryFrozenTest(unittest.TestCase):
    """Test that _Entry is immutable."""

    def test_cannot_set_attribute(self) -> None:
        entry = _Entry("CT-0001", "run", "qwen")
        with self.assertRaises(AttributeError):
            entry.task_id = "CT-9999"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
