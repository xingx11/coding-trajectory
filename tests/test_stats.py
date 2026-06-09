from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ctpipe.state import PipelineState
from ctpipe.stats import (
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


if __name__ == "__main__":
    unittest.main()
