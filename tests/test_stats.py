from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ctpipe.state import PipelineState
from ctpipe.stats import (
    _collect_passrate_diff,
    _collect_timing_stats,
    _find_slowest,
    _fmt_duration,
)


class TestFmtDuration(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(_fmt_duration(45), "45s")

    def test_minutes(self):
        self.assertEqual(_fmt_duration(150), "2m30s")

    def test_hours(self):
        self.assertEqual(_fmt_duration(3720), "1h02m")

    def test_zero(self):
        self.assertEqual(_fmt_duration(0), "0s")


class TestCollectTimingStats(unittest.TestCase):
    def _make_state(self, tmp: Path) -> PipelineState:
        return PipelineState(tmp / "pipeline_state.json")

    def test_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            state.set("T1", "run", model="qwen", status="done", duration_s=100.0)
            state.set("T2", "run", model="qwen", status="done", duration_s=200.0)
            state.set("T1", "run", model="claude", status="done", duration_s=150.0)
            state.set("T2", "run", model="claude", status="done", duration_s=250.0)

            result = _collect_timing_stats(state, ["T1", "T2"], ["qwen", "claude"])

            self.assertIn("run/qwen", result)
            self.assertIn("run/claude", result)
            rq = result["run/qwen"]
            self.assertAlmostEqual(rq["min"], 100.0)
            self.assertAlmostEqual(rq["max"], 200.0)
            self.assertAlmostEqual(rq["mean"], 150.0)
            self.assertAlmostEqual(rq["total"], 300.0)
            self.assertEqual(rq["count"], 2)

    def test_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            state.set("T1", "run", model="qwen", status="done")
            state.set("T1", "run", model="claude", status="done")

            result = _collect_timing_stats(state, ["T1"], ["qwen", "claude"])

            self.assertEqual(result, {})

    def test_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            state.set("T1", "run", model="qwen", status="done", duration_s=120.0)
            state.set("T2", "run", model="qwen", status="done")
            state.set("T1", "run", model="claude", status="failed")

            result = _collect_timing_stats(state, ["T1", "T2"], ["qwen", "claude"])

            self.assertIn("run/qwen", result)
            self.assertNotIn("run/claude", result)
            rq = result["run/qwen"]
            self.assertEqual(rq["count"], 1)
            self.assertAlmostEqual(rq["min"], 120.0)
            self.assertAlmostEqual(rq["total"], 120.0)

    def test_score_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            state.set("T1", "score", model="qwen", status="done", duration_s=30.0)
            state.set("T2", "score", model="qwen", status="done", duration_s=50.0)

            result = _collect_timing_stats(state, ["T1", "T2"], ["qwen"])

            self.assertIn("score/qwen", result)
            sq = result["score/qwen"]
            self.assertAlmostEqual(sq["min"], 30.0)
            self.assertAlmostEqual(sq["max"], 50.0)
            self.assertEqual(sq["count"], 2)

    def test_zero_duration_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            state.set("T1", "run", model="qwen", status="done", duration_s=0)
            state.set("T2", "run", model="qwen", status="done", duration_s=80.0)

            result = _collect_timing_stats(state, ["T1", "T2"], ["qwen"])

            rq = result["run/qwen"]
            self.assertEqual(rq["count"], 1)


class TestFindSlowest(unittest.TestCase):
    def _make_state(self, tmp: Path) -> PipelineState:
        return PipelineState(tmp / "pipeline_state.json")

    def test_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            state.set("T1", "run", model="qwen", status="done", duration_s=100.0)
            state.set("T2", "run", model="claude", status="done", duration_s=300.0)
            state.set("T1", "run", model="claude", status="done", duration_s=150.0)

            result = _find_slowest(state, ["T1", "T2"], ["qwen", "claude"])

            self.assertIsNotNone(result)
            tid, model, dur = result
            self.assertEqual(tid, "T2")
            self.assertEqual(model, "claude")
            self.assertAlmostEqual(dur, 300.0)

    def test_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            state.set("T1", "run", model="qwen", status="pending")

            result = _find_slowest(state, ["T1"], ["qwen", "claude"])

            self.assertIsNone(result)

    def test_single_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            state.set("T1", "run", model="qwen", status="done", duration_s=42.5)

            result = _find_slowest(state, ["T1"], ["qwen"])

            self.assertIsNotNone(result)
            self.assertEqual(result, ("T1", "qwen", 42.5))


class TestCollectPassrateDiff(unittest.TestCase):
    def _make_state(self, tmp: Path) -> PipelineState:
        return PipelineState(tmp / "pipeline_state.json")

    def _set_pr(self, state, tid, qwen=None, claude=None):
        kwargs = {}
        if qwen is not None:
            kwargs["qwen_passrate"] = qwen
        if claude is not None:
            kwargs["claude_passrate"] = claude
        state.set(tid, "finalize", **kwargs)

    def test_three_buckets_partition(self):
        # claude>qwen (x2), tie, claude<qwen -> positive/negative/tie are
        # disjoint and sum to count. (A tie must NOT be folded into negative.)
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            self._set_pr(state, "T1", qwen=0.5, claude=0.8)  # claude > qwen
            self._set_pr(state, "T2", qwen=0.4, claude=0.9)  # claude > qwen
            self._set_pr(state, "T3", qwen=0.6, claude=0.6)  # tie
            self._set_pr(state, "T4", qwen=0.7, claude=0.5)  # claude < qwen

            res = _collect_passrate_diff(
                state, ["T1", "T2", "T3", "T4"], "qwen", "claude"
            )

            self.assertIsNotNone(res)
            self.assertEqual(res["count"], 4)
            self.assertEqual(res["positive"], 2)
            self.assertEqual(res["negative"], 1)
            self.assertEqual(res["tie"], 1)
            # The three buckets partition every paired task exactly once.
            self.assertEqual(
                res["positive"] + res["negative"] + res["tie"], res["count"]
            )

    def test_all_ties(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            self._set_pr(state, "T1", qwen=0.5, claude=0.5)
            self._set_pr(state, "T2", qwen=0.7, claude=0.7)

            res = _collect_passrate_diff(state, ["T1", "T2"], "qwen", "claude")

            self.assertEqual(res["positive"], 0)
            self.assertEqual(res["negative"], 0)
            self.assertEqual(res["tie"], 2)
            self.assertEqual(
                res["positive"] + res["negative"] + res["tie"], res["count"]
            )
            self.assertAlmostEqual(res["mean"], 0.0)

    def test_single_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            self._set_pr(state, "T1", qwen=0.6, claude=0.6)

            res = _collect_passrate_diff(state, ["T1"], "qwen", "claude")

            self.assertEqual(res["count"], 1)
            self.assertEqual(res["positive"], 0)
            self.assertEqual(res["negative"], 0)
            self.assertEqual(res["tie"], 1)
            self.assertEqual(
                res["positive"] + res["negative"] + res["tie"], res["count"]
            )
            self.assertAlmostEqual(res["mean"], 0.0)

    def test_unpaired_excluded(self):
        # Tasks missing one model's passrate are not paired -> 0 pairs -> None.
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            self._set_pr(state, "T1", qwen=0.5)            # claude missing
            self._set_pr(state, "T2", claude=0.8)          # qwen missing

            res = _collect_passrate_diff(state, ["T1", "T2"], "qwen", "claude")

            self.assertIsNone(res)

    def test_mean_direction(self):
        # diff is model_b - model_a == claude - qwen.
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            self._set_pr(state, "T1", qwen=0.5, claude=0.8)  # +0.3
            self._set_pr(state, "T2", qwen=0.4, claude=0.6)  # +0.2

            res = _collect_passrate_diff(state, ["T1", "T2"], "qwen", "claude")

            self.assertAlmostEqual(res["mean"], 0.25)
            self.assertEqual(res["positive"], 2)
            self.assertEqual(res["negative"], 0)
            self.assertEqual(res["tie"], 0)

    def test_all_positive(self):
        # Every task: claude > qwen -> all diffs positive, zero negative/tie.
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            self._set_pr(state, "T1", qwen=0.4, claude=0.7)  # +0.3
            self._set_pr(state, "T2", qwen=0.5, claude=0.9)  # +0.4
            self._set_pr(state, "T3", qwen=0.6, claude=0.8)  # +0.2

            res = _collect_passrate_diff(
                state, ["T1", "T2", "T3"], "qwen", "claude"
            )

            self.assertEqual(res["count"], 3)
            self.assertEqual(res["positive"], 3)
            self.assertEqual(res["negative"], 0)
            self.assertEqual(res["tie"], 0)
            self.assertEqual(
                res["positive"] + res["negative"] + res["tie"], res["count"]
            )
            self.assertGreater(res["mean"], 0)

    def test_all_negative(self):
        # Every task: claude < qwen -> all diffs negative, zero positive/tie.
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            self._set_pr(state, "T1", qwen=0.8, claude=0.5)  # -0.3
            self._set_pr(state, "T2", qwen=0.9, claude=0.6)  # -0.3
            self._set_pr(state, "T3", qwen=0.7, claude=0.4)  # -0.3

            res = _collect_passrate_diff(
                state, ["T1", "T2", "T3"], "qwen", "claude"
            )

            self.assertEqual(res["count"], 3)
            self.assertEqual(res["positive"], 0)
            self.assertEqual(res["negative"], 3)
            self.assertEqual(res["tie"], 0)
            self.assertEqual(
                res["positive"] + res["negative"] + res["tie"], res["count"]
            )
            self.assertLess(res["mean"], 0)

    def test_single_positive(self):
        # Single paired task where claude > qwen.
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            self._set_pr(state, "T1", qwen=0.4, claude=0.8)

            res = _collect_passrate_diff(state, ["T1"], "qwen", "claude")

            self.assertIsNotNone(res)
            self.assertEqual(res["count"], 1)
            self.assertEqual(res["positive"], 1)
            self.assertEqual(res["negative"], 0)
            self.assertEqual(res["tie"], 0)
            self.assertEqual(
                res["positive"] + res["negative"] + res["tie"], res["count"]
            )

    def test_single_negative(self):
        # Single paired task where claude < qwen.
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            self._set_pr(state, "T1", qwen=0.8, claude=0.4)

            res = _collect_passrate_diff(state, ["T1"], "qwen", "claude")

            self.assertIsNotNone(res)
            self.assertEqual(res["count"], 1)
            self.assertEqual(res["positive"], 0)
            self.assertEqual(res["negative"], 1)
            self.assertEqual(res["tie"], 0)
            self.assertEqual(
                res["positive"] + res["negative"] + res["tie"], res["count"]
            )

    def test_floating_point_tie(self):
        # Passrates that are equal at 4-decimal precision but could differ by
        # a floating-point artefact (e.g. 0.3333 vs 0.3333 stored as slightly
        # different IEEE 754 values). The round() inside _collect_passrate_diff
        # must normalise them so diff == 0.0 and the task lands in the tie
        # bucket rather than leaking into positive or negative.
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state(Path(tmp))
            # Simulate values that could arise from float("0.3333"):
            # one exact, one nudged by a ULP-scale epsilon.
            self._set_pr(state, "T1", qwen=0.3333, claude=0.3333 + 1e-15)
            self._set_pr(state, "T2", qwen=0.6667 + 1e-16, claude=0.6667)

            res = _collect_passrate_diff(state, ["T1", "T2"], "qwen", "claude")

            self.assertEqual(res["count"], 2)
            self.assertEqual(res["positive"], 0)
            self.assertEqual(res["negative"], 0)
            self.assertEqual(res["tie"], 2)
            self.assertEqual(
                res["positive"] + res["negative"] + res["tie"], res["count"]
            )


if __name__ == "__main__":
    unittest.main()
