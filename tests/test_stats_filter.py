"""Tests for --filter and --group-by support in stats subcommand.

Covers:
  1. _parse_filter_expr  – tokenizer + recursive-descent parser
  2. _apply_filter       – record-level filtering
  3. _group_by_fields    – per-field independent grouping + passrate aggregation
  4. Filter + group-by combined workflow
  5. Whitelist enforcement – illegal field names raise ValueError
  6. show_stats end-to-end – table and JSON output with filter/group-by
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import PropertyMock, patch

from ctpipe.config import BatchConfig, TaskConfig
from ctpipe.state import PipelineState
from ctpipe.stats import (
    _FILTER_FIELD_WHITELIST,
    _apply_filter,
    _build_task_records,
    _format_pr,
    _group_by_fields,
    _parse_filter_expr,
    show_stats,
)

from conftest import build_config, make_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODELS = ["qwen", "claude"]


def _records() -> list[dict]:
    """Five-record fixture spanning three task types, three domains, three languages."""
    return [
        {"id": "CT-0001", "task_type": "bug-fix", "domain": "cli",
         "language": "python", "bad_pattern": "lazy_shortcut",
         "qwen_passrate": 0.30, "claude_passrate": 0.80},
        {"id": "CT-0002", "task_type": "feature", "domain": "web",
         "language": "python", "bad_pattern": "",
         "qwen_passrate": 0.60, "claude_passrate": 0.90},
        {"id": "CT-0003", "task_type": "bug-fix", "domain": "web",
         "language": "java", "bad_pattern": "",
         "qwen_passrate": 0.40, "claude_passrate": 0.85},
        {"id": "CT-0004", "task_type": "feature", "domain": "cli",
         "language": "python", "bad_pattern": "",
         "qwen_passrate": 0.70, "claude_passrate": 0.75},
        {"id": "CT-0005", "task_type": "enhancement", "domain": "data",
         "language": "go", "bad_pattern": "",
         "qwen_passrate": 0.50, "claude_passrate": 0.95},
    ]


def _write_state(state_path: Path, finalize_entries: list[dict]) -> PipelineState:
    """Create a PipelineState with all stages set to done for each entry.

    Each entry is a dict with at least ``task_id`` plus optional
    ``qwen_passrate``, ``claude_passrate``, ``status``, ``threshold_ok``.

    All pipeline stages (prepare, run, collect, score, finalize, validate)
    are populated so that ``_collect_stage_counts`` reports no pending work
    and ``all_ok`` evaluates to True.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = PipelineState(state_path)
    for e in finalize_entries:
        tid = e["task_id"]
        # Model-agnostic stages
        state.set(tid, "prepare", status="done")
        state.set(
            tid, "finalize",
            status=e.get("status", "done"),
            qwen_passrate=e.get("qwen_passrate", 0.0),
            claude_passrate=e.get("claude_passrate", 0.0),
            threshold_ok=e.get("threshold_ok", True),
        )
        state.set(tid, "validate", status="done")
        # Per-model stages
        for model in ("qwen", "claude"):
            state.set(tid, "run", model=model, status="done")
            state.set(tid, "collect", model=model, status="done")
            state.set(tid, "score", model=model, status="done")
    return state


# ===================================================================
# 1. Pure --filter tests
# ===================================================================

class TestFilterParser(unittest.TestCase):
    """Validate the recursive-descent parser for filter expressions."""

    # -- comparison operators -----------------------------------------

    def test_eq_string(self):
        fn = _parse_filter_expr("task_type = 'bug-fix'")
        self.assertTrue(fn({"task_type": "bug-fix"}))
        self.assertFalse(fn({"task_type": "feature"}))

    def test_eq_unquoted_identifier(self):
        fn = _parse_filter_expr("domain = cli")
        self.assertTrue(fn({"domain": "cli"}))
        self.assertFalse(fn({"domain": "web"}))

    def test_neq(self):
        fn = _parse_filter_expr("task_type != 'bug-fix'")
        self.assertFalse(fn({"task_type": "bug-fix"}))
        self.assertTrue(fn({"task_type": "feature"}))

    def test_lt(self):
        fn = _parse_filter_expr("qwen_passrate < 0.5")
        self.assertTrue(fn({"qwen_passrate": 0.3}))
        self.assertFalse(fn({"qwen_passrate": 0.5}))
        self.assertFalse(fn({"qwen_passrate": 0.7}))

    def test_gt(self):
        fn = _parse_filter_expr("claude_passrate > 0.8")
        self.assertTrue(fn({"claude_passrate": 0.9}))
        self.assertFalse(fn({"claude_passrate": 0.8}))

    def test_le(self):
        fn = _parse_filter_expr("qwen_passrate <= 0.5")
        self.assertTrue(fn({"qwen_passrate": 0.5}))
        self.assertTrue(fn({"qwen_passrate": 0.3}))
        self.assertFalse(fn({"qwen_passrate": 0.6}))

    def test_ge(self):
        fn = _parse_filter_expr("claude_passrate >= 0.7")
        self.assertTrue(fn({"claude_passrate": 0.7}))
        self.assertTrue(fn({"claude_passrate": 0.9}))
        self.assertFalse(fn({"claude_passrate": 0.6}))

    # -- logical operators --------------------------------------------

    def test_and(self):
        fn = _parse_filter_expr("task_type = 'bug-fix' AND qwen_passrate < 0.5")
        self.assertTrue(fn({"task_type": "bug-fix", "qwen_passrate": 0.3}))
        self.assertFalse(fn({"task_type": "bug-fix", "qwen_passrate": 0.7}))
        self.assertFalse(fn({"task_type": "feature", "qwen_passrate": 0.3}))

    def test_or(self):
        fn = _parse_filter_expr("domain = 'cli' OR language = 'python'")
        self.assertTrue(fn({"domain": "cli", "language": "java"}))
        self.assertTrue(fn({"domain": "web", "language": "python"}))
        self.assertFalse(fn({"domain": "web", "language": "java"}))

    def test_and_or_precedence(self):
        # AND binds tighter than OR: (A AND B) OR C
        fn = _parse_filter_expr(
            "task_type = 'bug-fix' AND qwen_passrate < 0.5 OR claude_passrate > 0.9"
        )
        self.assertTrue(fn({"task_type": "bug-fix", "qwen_passrate": 0.3,
                            "claude_passrate": 0.5}))
        self.assertTrue(fn({"task_type": "feature", "qwen_passrate": 0.7,
                            "claude_passrate": 0.95}))
        self.assertFalse(fn({"task_type": "feature", "qwen_passrate": 0.7,
                             "claude_passrate": 0.5}))

    def test_parentheses_override_precedence(self):
        fn = _parse_filter_expr(
            "(task_type = 'bug-fix' OR task_type = 'feature') AND qwen_passrate < 0.5"
        )
        self.assertTrue(fn({"task_type": "bug-fix", "qwen_passrate": 0.3}))
        self.assertTrue(fn({"task_type": "feature", "qwen_passrate": 0.4}))
        self.assertFalse(fn({"task_type": "bug-fix", "qwen_passrate": 0.7}))
        self.assertFalse(fn({"task_type": "enhancement", "qwen_passrate": 0.3}))

    def test_nested_parentheses(self):
        fn = _parse_filter_expr("((task_type = 'bug-fix'))")
        self.assertTrue(fn({"task_type": "bug-fix"}))

    # -- value types --------------------------------------------------

    def test_double_quoted_string(self):
        fn = _parse_filter_expr('task_type = "feature"')
        self.assertTrue(fn({"task_type": "feature"}))

    def test_string_with_spaces(self):
        fn = _parse_filter_expr("bad_pattern = 'lazy shortcut'")
        self.assertTrue(fn({"bad_pattern": "lazy shortcut"}))

    def test_integer_value(self):
        fn = _parse_filter_expr("qwen_passrate > 0")
        self.assertTrue(fn({"qwen_passrate": 0.5}))

    def test_negative_number(self):
        fn = _parse_filter_expr("qwen_passrate > -0.5")
        self.assertTrue(fn({"qwen_passrate": 0.0}))
        self.assertFalse(fn({"qwen_passrate": -0.6}))

    # -- None / missing fields ----------------------------------------

    def test_none_field_returns_false(self):
        fn = _parse_filter_expr("qwen_passrate < 0.5")
        self.assertFalse(fn({"qwen_passrate": None}))
        self.assertFalse(fn({}))

    # -- syntax errors ------------------------------------------------

    def test_empty_expression(self):
        with self.assertRaises(ValueError):
            _parse_filter_expr("")

    def test_trailing_and(self):
        with self.assertRaises(ValueError):
            _parse_filter_expr("task_type = 'bug-fix' AND")

    def test_unclosed_paren(self):
        with self.assertRaises(ValueError):
            _parse_filter_expr("(task_type = 'bug-fix'")

    def test_missing_operator(self):
        with self.assertRaises(ValueError):
            _parse_filter_expr("task_type 'bug-fix'")

    def test_unexpected_character(self):
        with self.assertRaises(ValueError):
            _parse_filter_expr("task_type; DROP TABLE")

    def test_unterminated_string(self):
        with self.assertRaises(ValueError):
            _parse_filter_expr("task_type = 'bug-fix")


# ===================================================================
# 2. Pure --group-by tests
# ===================================================================

class TestGroupBy(unittest.TestCase):
    """Validate per-field independent grouping and passrate aggregation."""

    def test_single_field(self):
        groupings = _group_by_fields(_records(), ["task_type"], _MODELS)

        self.assertEqual(len(groupings), 1)
        g = groupings[0]
        self.assertEqual(g["field"], "task_type")
        self.assertEqual(g["count"], 5)
        self.assertEqual(len(g["groups"]), 3)

        bf = [x for x in g["groups"] if x["group_value"] == "bug-fix"][0]
        self.assertEqual(bf["count"], 2)
        self.assertAlmostEqual(bf["passrate_stats"]["qwen"]["mean"], 0.35)
        self.assertAlmostEqual(bf["passrate_stats"]["qwen"]["min"], 0.30)
        self.assertAlmostEqual(bf["passrate_stats"]["qwen"]["max"], 0.40)
        self.assertEqual(bf["passrate_stats"]["qwen"]["count"], 2)

    def test_multiple_fields_independent(self):
        groupings = _group_by_fields(
            _records(), ["task_type", "domain", "language"], _MODELS,
        )

        self.assertEqual(len(groupings), 3)
        fields = [g["field"] for g in groupings]
        self.assertEqual(fields, ["task_type", "domain", "language"])

        # domain has 3 unique values: cli, web, data
        domain_g = groupings[1]
        self.assertEqual(len(domain_g["groups"]), 3)

        # language has 3 unique values: python, java, go
        lang_g = groupings[2]
        self.assertEqual(len(lang_g["groups"]), 3)

    def test_passrate_stats_keys(self):
        groupings = _group_by_fields(_records(), ["task_type"], _MODELS)
        for grp in groupings[0]["groups"]:
            for model in _MODELS:
                stats = grp["passrate_stats"][model]
                self.assertIn("min", stats)
                self.assertIn("max", stats)
                self.assertIn("mean", stats)
                self.assertIn("median", stats)
                self.assertIn("std", stats)
                self.assertIn("count", stats)

    def test_none_passrate_excluded(self):
        records = [
            {"task_type": "bug-fix", "qwen_passrate": None, "claude_passrate": 0.8},
            {"task_type": "bug-fix", "qwen_passrate": 0.4, "claude_passrate": None},
        ]
        groupings = _group_by_fields(records, ["task_type"], _MODELS)
        grp = groupings[0]["groups"][0]

        self.assertEqual(grp["passrate_stats"]["qwen"]["count"], 1)
        self.assertAlmostEqual(grp["passrate_stats"]["qwen"]["mean"], 0.4)
        self.assertEqual(grp["passrate_stats"]["claude"]["count"], 1)
        self.assertAlmostEqual(grp["passrate_stats"]["claude"]["mean"], 0.8)

    def test_all_none_passrate_omits_model(self):
        records = [
            {"task_type": "bug-fix", "qwen_passrate": None, "claude_passrate": None},
        ]
        groupings = _group_by_fields(records, ["task_type"], _MODELS)
        grp = groupings[0]["groups"][0]

        self.assertNotIn("qwen", grp["passrate_stats"])
        self.assertNotIn("claude", grp["passrate_stats"])

    def test_single_record_group(self):
        groupings = _group_by_fields(_records()[:1], ["task_type"], _MODELS)
        grp = groupings[0]["groups"][0]
        self.assertEqual(grp["count"], 1)
        self.assertAlmostEqual(grp["passrate_stats"]["qwen"]["min"], 0.30)
        self.assertAlmostEqual(grp["passrate_stats"]["qwen"]["max"], 0.30)

    def test_single_sample_std_is_zero(self):
        """Single sample: std should be 0.0, median should equal the value."""
        records = [
            {"task_type": "bug-fix", "qwen_passrate": 0.75, "claude_passrate": 0.85},
        ]
        groupings = _group_by_fields(records, ["task_type"], _MODELS)
        grp = groupings[0]["groups"][0]

        # qwen: single sample
        qwen_stats = grp["passrate_stats"]["qwen"]
        self.assertEqual(qwen_stats["count"], 1)
        self.assertAlmostEqual(qwen_stats["median"], 0.75)
        self.assertAlmostEqual(qwen_stats["std"], 0.0)
        self.assertAlmostEqual(qwen_stats["mean"], 0.75)

        # claude: single sample
        claude_stats = grp["passrate_stats"]["claude"]
        self.assertEqual(claude_stats["count"], 1)
        self.assertAlmostEqual(claude_stats["median"], 0.85)
        self.assertAlmostEqual(claude_stats["std"], 0.0)

    def test_multiple_samples_std_computed(self):
        """Multiple samples: std should be computed correctly."""
        records = [
            {"task_type": "bug-fix", "qwen_passrate": 0.6, "claude_passrate": 0.8},
            {"task_type": "bug-fix", "qwen_passrate": 0.8, "claude_passrate": 0.9},
        ]
        groupings = _group_by_fields(records, ["task_type"], _MODELS)
        grp = groupings[0]["groups"][0]

        # qwen: [0.6, 0.8] -> mean=0.7, median=0.7, std=0.1414...
        qwen_stats = grp["passrate_stats"]["qwen"]
        self.assertEqual(qwen_stats["count"], 2)
        self.assertAlmostEqual(qwen_stats["mean"], 0.7)
        self.assertAlmostEqual(qwen_stats["median"], 0.7)
        self.assertAlmostEqual(qwen_stats["std"], 0.1414, places=4)

        # claude: [0.8, 0.9] -> mean=0.85, median=0.85, std=0.0707...
        claude_stats = grp["passrate_stats"]["claude"]
        self.assertEqual(claude_stats["count"], 2)
        self.assertAlmostEqual(claude_stats["mean"], 0.85)
        self.assertAlmostEqual(claude_stats["median"], 0.85)
        self.assertAlmostEqual(claude_stats["std"], 0.0707, places=4)

    def test_no_data_omits_model(self):
        """No data for a model: that model should be omitted from passrate_stats."""
        records = [
            {"task_type": "bug-fix", "qwen_passrate": None, "claude_passrate": None},
            {"task_type": "bug-fix", "qwen_passrate": None, "claude_passrate": None},
        ]
        groupings = _group_by_fields(records, ["task_type"], _MODELS)
        grp = groupings[0]["groups"][0]

        # Both models should be omitted since all values are None
        self.assertNotIn("qwen", grp["passrate_stats"])
        self.assertNotIn("claude", grp["passrate_stats"])
        self.assertEqual(grp["passrate_stats"], {})


# ===================================================================
# 3. Filter + group-by combined
# ===================================================================

class TestFilterAndGroupByCombined(unittest.TestCase):
    """Filter first, then group the surviving records."""

    def test_filter_then_group(self):
        records = _records()
        fn = _parse_filter_expr("task_type = 'bug-fix'")
        filtered = _apply_filter(records, fn)

        self.assertEqual(len(filtered), 2)

        groupings = _group_by_fields(filtered, ["domain"], _MODELS)
        g = groupings[0]
        self.assertEqual(g["field"], "domain")
        self.assertEqual(g["count"], 2)

        # bug-fix tasks span cli and web
        values = {grp["group_value"] for grp in g["groups"]}
        self.assertEqual(values, {"cli", "web"})

    def test_numeric_filter_then_group(self):
        records = _records()
        fn = _parse_filter_expr("qwen_passrate < 0.5")
        filtered = _apply_filter(records, fn)

        # CT-0001 (0.30) and CT-0003 (0.40)
        self.assertEqual(len(filtered), 2)

        groupings = _group_by_fields(filtered, ["language"], _MODELS)
        values = {grp["group_value"] for grp in groupings[0]["groups"]}
        self.assertEqual(values, {"python", "java"})

    def test_complex_filter_then_multi_group(self):
        records = _records()
        fn = _parse_filter_expr(
            "(task_type = 'bug-fix' OR task_type = 'feature') "
            "AND claude_passrate >= 0.8"
        )
        filtered = _apply_filter(records, fn)

        # CT-0001 (bug-fix, 0.80), CT-0002 (feature, 0.90), CT-0003 (bug-fix, 0.85)
        self.assertEqual(len(filtered), 3)

        groupings = _group_by_fields(filtered, ["task_type", "domain"], _MODELS)
        # task_type: bug-fix(2), feature(1)
        tt_groups = {grp["group_value"]: grp["count"]
                     for grp in groupings[0]["groups"]}
        self.assertEqual(tt_groups, {"bug-fix": 2, "feature": 1})

        # domain: cli(1), web(2)
        d_groups = {grp["group_value"]: grp["count"]
                    for grp in groupings[1]["groups"]}
        self.assertEqual(d_groups, {"cli": 1, "web": 2})

    def test_filter_removes_all_records(self):
        records = _records()
        fn = _parse_filter_expr("qwen_passrate > 0.99")
        filtered = _apply_filter(records, fn)
        self.assertEqual(filtered, [])

        groupings = _group_by_fields(filtered, ["task_type"], _MODELS)
        self.assertEqual(groupings[0]["groups"], [])


# ===================================================================
# 4. Illegal field name rejection (whitelist enforcement)
# ===================================================================

class TestFilterWhitelist(unittest.TestCase):
    """Only fields in _FILTER_FIELD_WHITELIST are accepted."""

    def test_whitelist_contents(self):
        expected = {
            "task_type", "domain", "language",
            "bad_pattern", "qwen_passrate", "claude_passrate",
            "relative_gain", "threshold_ok", "finalize_status",
            "qwen_over_threshold", "claude_under_threshold",
            "claude_not_better", "gain_below_threshold",
        }
        self.assertEqual(set(_FILTER_FIELD_WHITELIST), expected)

    def test_all_whitelisted_fields_parse(self):
        for field in _FILTER_FIELD_WHITELIST:
            fn = _parse_filter_expr(f"{field} = 'x'")
            self.assertTrue(callable(fn), f"{field} should produce a callable")

    def test_internal_record_fields_rejected(self):
        """Fields stored in records but not whitelisted must be blocked."""
        for field in ("id", "task_title"):
            with self.assertRaises(ValueError, msg=f"{field} should be rejected"):
                _parse_filter_expr(f"{field} = 'x'")

    def test_arbitrary_names_rejected(self):
        for field in ("status", "passrate", "admin", "__import__", "eval"):
            with self.assertRaises(ValueError, msg=f"{field} should be rejected"):
                _parse_filter_expr(f"{field} = 'x'")

    def test_injection_via_field_name(self):
        with self.assertRaises(ValueError):
            _parse_filter_expr("x; DROP TABLE meta.import = 'y'")

    def test_injection_via_value_is_safe(self):
        """SQL-style injection in the VALUE position is harmless:
        the value is treated as a literal string, never executed."""
        fn = _parse_filter_expr("task_type = \"'; DROP TABLE users; --\"")
        self.assertTrue(fn({"task_type": "'; DROP TABLE users; --"}))
        self.assertFalse(fn({"task_type": "bug-fix"}))

    def test_error_lists_allowed_fields(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_filter_expr("badfield = 'x'")
        for field in _FILTER_FIELD_WHITELIST:
            self.assertIn(field, str(ctx.exception))

    def test_rejection_inside_parentheses(self):
        with self.assertRaises(ValueError):
            _parse_filter_expr("(secret_field = 'x' OR task_type = 'bug-fix')")

    def test_rejection_in_and_chain(self):
        with self.assertRaises(ValueError):
            _parse_filter_expr("task_type = 'bug-fix' AND secret > 0")


# ===================================================================
# 5. New filterable fields: relative_gain, threshold_ok, finalize_status
# ===================================================================

class TestRelativeGainFilter(unittest.TestCase):
    """Filter on the derived relative_gain metric."""

    def test_relative_gain_lt(self):
        fn = _parse_filter_expr("relative_gain < 0.25")
        self.assertTrue(fn({"relative_gain": 0.10}))
        self.assertFalse(fn({"relative_gain": 0.25}))
        self.assertFalse(fn({"relative_gain": 0.50}))

    def test_relative_gain_ge(self):
        fn = _parse_filter_expr("relative_gain >= 0.25")
        self.assertTrue(fn({"relative_gain": 0.25}))
        self.assertTrue(fn({"relative_gain": 1.0}))
        self.assertFalse(fn({"relative_gain": 0.20}))

    def test_relative_gain_none_returns_false(self):
        """qwen == 0 or missing passrate → None → any comparison is False."""
        fn = _parse_filter_expr("relative_gain < 0.25")
        self.assertFalse(fn({"relative_gain": None}))
        self.assertFalse(fn({}))

    def test_relative_gain_combined_with_passrate(self):
        fn = _parse_filter_expr(
            "relative_gain < 0.25 AND qwen_passrate > 0.3"
        )
        self.assertTrue(fn({"relative_gain": 0.10, "qwen_passrate": 0.5}))
        self.assertFalse(fn({"relative_gain": 0.10, "qwen_passrate": 0.2}))
        self.assertFalse(fn({"relative_gain": 0.50, "qwen_passrate": 0.5}))

    def test_relative_gain_negative(self):
        """claude < qwen → negative gain, should match < 0."""
        fn = _parse_filter_expr("relative_gain < 0")
        self.assertTrue(fn({"relative_gain": -0.3}))
        self.assertFalse(fn({"relative_gain": 0.0}))
        self.assertFalse(fn({"relative_gain": 0.5}))

    def test_relative_gain_eq_boundary(self):
        """Exact boundary: = 0.25 matches 0.25, not 0.2499."""
        fn = _parse_filter_expr("relative_gain = 0.25")
        self.assertTrue(fn({"relative_gain": 0.25}))
        self.assertFalse(fn({"relative_gain": 0.2499}))
        self.assertFalse(fn({"relative_gain": 0.2501}))

    def test_relative_gain_in_build_task_records(self):
        """_build_task_records computes relative_gain from passrates."""
        import tempfile
        from conftest import build_config, make_task
        tasks = [
            make_task("CT-0001", "bug-fix", "cli", "python"),
            make_task("CT-0002", "feature", "web", "python"),
            make_task("CT-0003", "enhancement", "data", "go"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=Path(tmp)):
                config = build_config(tasks=tasks)
                _write_state(config.state_path, [
                    {"task_id": "CT-0001", "qwen_passrate": 0.4,
                     "claude_passrate": 0.6, "status": "done", "threshold_ok": True},
                    {"task_id": "CT-0002", "qwen_passrate": 0.0,
                     "claude_passrate": 0.5, "status": "partial", "threshold_ok": False},
                    {"task_id": "CT-0003", "claude_passrate": 0.8,
                     "status": "failed", "threshold_ok": False},
                ])
                from ctpipe.state import PipelineState as PS
                state = PS(config.state_path)
                records = _build_task_records(config, tasks, state, ["qwen", "claude"])
                # (0.6 - 0.4) / 0.4 = 0.5
                self.assertAlmostEqual(records[0]["relative_gain"], 0.5)
                # qwen == 0 → None (consistent with red-line validation)
                self.assertIsNone(records[1]["relative_gain"])
                # missing qwen passrate → None
                self.assertIsNone(records[2]["relative_gain"])

    def test_redline_fields_in_build_task_records(self):
        """_build_task_records computes the 4 redline booleans correctly."""
        import tempfile
        from conftest import build_config, make_task
        tasks = [
            make_task("CT-0001", "bug-fix", "cli", "python"),
            make_task("CT-0002", "feature", "web", "python"),
            make_task("CT-0003", "enhancement", "data", "go"),
            make_task("CT-0004", "feature", "cli", "python"),
            make_task("CT-0005", "bug-fix", "web", "java"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=Path(tmp)):
                config = build_config(tasks=tasks)
                _write_state(config.state_path, [
                    # CT-0001: all clear — qwen 0.30, claude 0.80, gain 1.67
                    {"task_id": "CT-0001", "qwen_passrate": 0.30,
                     "claude_passrate": 0.80, "status": "done", "threshold_ok": True},
                    # CT-0002: qwen over (0.75 >= 0.7)
                    {"task_id": "CT-0002", "qwen_passrate": 0.75,
                     "claude_passrate": 0.95, "status": "partial", "threshold_ok": False},
                    # CT-0003: claude under (0.60 <= 0.7) + claude_not_better (0.60 <= 0.80)
                    {"task_id": "CT-0003", "qwen_passrate": 0.80,
                     "claude_passrate": 0.60, "status": "partial", "threshold_ok": False},
                    # CT-0004: gain below (gain = (0.76-0.50)/0.50 = 0.52 > 0.25 → clear)
                    # Actually let's make it violate: qwen 0.50, claude 0.55 → gain = 0.10
                    {"task_id": "CT-0004", "qwen_passrate": 0.50,
                     "claude_passrate": 0.55, "status": "partial", "threshold_ok": False},
                    # CT-0005: missing qwen → no redline booleans apply
                    {"task_id": "CT-0005", "claude_passrate": 0.80,
                     "status": "failed", "threshold_ok": False},
                ])
                from ctpipe.state import PipelineState as PS
                state = PS(config.state_path)
                records = _build_task_records(config, tasks, state, ["qwen", "claude"])

                # CT-0001: all redlines clear
                r = records[0]
                self.assertFalse(r["qwen_over_threshold"])
                self.assertFalse(r["claude_under_threshold"])
                self.assertFalse(r["claude_not_better"])
                self.assertFalse(r["gain_below_threshold"])

                # CT-0002: qwen_over_threshold only
                r = records[1]
                self.assertTrue(r["qwen_over_threshold"])
                self.assertFalse(r["claude_under_threshold"])
                self.assertFalse(r["claude_not_better"])
                self.assertFalse(r["gain_below_threshold"])

                # CT-0003: claude_under + claude_not_better + gain_below
                r = records[2]
                self.assertTrue(r["qwen_over_threshold"])  # 0.80 >= 0.7
                self.assertTrue(r["claude_under_threshold"])  # 0.60 <= 0.7
                self.assertTrue(r["claude_not_better"])  # 0.60 <= 0.80
                self.assertTrue(r["gain_below_threshold"])  # (0.60-0.80)/0.80 < 0

                # CT-0004: claude_under + gain_below
                r = records[3]
                self.assertFalse(r["qwen_over_threshold"])  # 0.50 < 0.7
                self.assertTrue(r["claude_under_threshold"])  # 0.55 <= 0.7
                self.assertFalse(r["claude_not_better"])  # 0.55 > 0.50
                self.assertTrue(r["gain_below_threshold"])  # gain=0.10 <= 0.25

                # CT-0005: missing qwen → redlines that need qwen are False
                r = records[4]
                self.assertFalse(r["qwen_over_threshold"])
                self.assertFalse(r["claude_under_threshold"])  # 0.80 > 0.7
                self.assertFalse(r["claude_not_better"])
                self.assertFalse(r["gain_below_threshold"])

    def test_redline_boundary_qwen_exactly_at_threshold(self):
        """qwen = 0.7 exactly should trigger qwen_over_threshold (>= 0.7)."""
        import tempfile
        from conftest import build_config, make_task
        tasks = [make_task("CT-0001", "bug-fix", "cli", "python")]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=Path(tmp)):
                config = build_config(tasks=tasks)
                _write_state(config.state_path, [
                    {"task_id": "CT-0001", "qwen_passrate": 0.7,
                     "claude_passrate": 0.95, "status": "done", "threshold_ok": False},
                ])
                from ctpipe.state import PipelineState as PS
                state = PS(config.state_path)
                records = _build_task_records(config, tasks, state, ["qwen", "claude"])
                r = records[0]
                # 0.7 >= THRESHOLD_QWEN_MAX (0.7) → True
                self.assertTrue(r["qwen_over_threshold"])
                # claude 0.95 > 0.7 → no issue
                self.assertFalse(r["claude_under_threshold"])
                # 0.95 > 0.70 → claude is better
                self.assertFalse(r["claude_not_better"])
                # gain = (0.95-0.7)/0.7 ≈ 0.357 > 0.25
                self.assertFalse(r["gain_below_threshold"])

    def test_redline_boundary_qwen_zero_gain_below(self):
        """qwen=0 uses claude absolute check for gain_below_threshold.

        When qwen=0 and claude < THRESHOLD_RELATIVE_GAIN_MIN (0.25),
        gain_below_threshold should be True.
        """
        import tempfile
        from conftest import build_config, make_task
        tasks = [make_task("CT-0001", "bug-fix", "cli", "python")]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=Path(tmp)):
                config = build_config(tasks=tasks)
                _write_state(config.state_path, [
                    {"task_id": "CT-0001", "qwen_passrate": 0.0,
                     "claude_passrate": 0.20, "status": "partial", "threshold_ok": False},
                ])
                from ctpipe.state import PipelineState as PS
                state = PS(config.state_path)
                records = _build_task_records(config, tasks, state, ["qwen", "claude"])
                r = records[0]
                self.assertFalse(r["qwen_over_threshold"])  # 0.0 < 0.7
                self.assertTrue(r["claude_under_threshold"])  # 0.20 <= 0.7
                self.assertFalse(r["claude_not_better"])  # 0.20 > 0.0
                # qwen=0 branch: claude 0.20 < 0.25 → gain_below True
                self.assertTrue(r["gain_below_threshold"])

    def test_redline_boundary_qwen_zero_gain_clear(self):
        """qwen=0 but claude high enough → gain_below_threshold is False."""
        import tempfile
        from conftest import build_config, make_task
        tasks = [make_task("CT-0001", "bug-fix", "cli", "python")]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=Path(tmp)):
                config = build_config(tasks=tasks)
                _write_state(config.state_path, [
                    {"task_id": "CT-0001", "qwen_passrate": 0.0,
                     "claude_passrate": 0.80, "status": "partial", "threshold_ok": False},
                ])
                from ctpipe.state import PipelineState as PS
                state = PS(config.state_path)
                records = _build_task_records(config, tasks, state, ["qwen", "claude"])
                r = records[0]
                # qwen=0 branch: claude 0.80 >= 0.25 → gain_below False
                self.assertFalse(r["gain_below_threshold"])

    def test_redline_boundary_claude_exactly_at_threshold(self):
        """claude = 0.7 exactly should trigger claude_under_threshold (<= 0.7)."""
        import tempfile
        from conftest import build_config, make_task
        tasks = [make_task("CT-0001", "bug-fix", "cli", "python")]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=Path(tmp)):
                config = build_config(tasks=tasks)
                _write_state(config.state_path, [
                    {"task_id": "CT-0001", "qwen_passrate": 0.50,
                     "claude_passrate": 0.7, "status": "partial", "threshold_ok": False},
                ])
                from ctpipe.state import PipelineState as PS
                state = PS(config.state_path)
                records = _build_task_records(config, tasks, state, ["qwen", "claude"])
                r = records[0]
                self.assertFalse(r["qwen_over_threshold"])  # 0.50 < 0.7
                # 0.7 <= THRESHOLD_CLAUDE_MIN (0.7) → True
                self.assertTrue(r["claude_under_threshold"])
                # 0.70 > 0.50 → claude is better
                self.assertFalse(r["claude_not_better"])
                # gain = (0.7-0.5)/0.5 = 0.40 > 0.25
                self.assertFalse(r["gain_below_threshold"])


class TestThresholdOkFilter(unittest.TestCase):
    """Filter on the boolean threshold_ok field."""

    def test_threshold_ok_true(self):
        fn = _parse_filter_expr("threshold_ok = True")
        self.assertTrue(fn({"threshold_ok": True}))
        self.assertFalse(fn({"threshold_ok": False}))

    def test_threshold_ok_false(self):
        fn = _parse_filter_expr("threshold_ok = False")
        self.assertTrue(fn({"threshold_ok": False}))
        self.assertFalse(fn({"threshold_ok": True}))

    def test_threshold_ok_neq(self):
        fn = _parse_filter_expr("threshold_ok != True")
        self.assertTrue(fn({"threshold_ok": False}))
        self.assertFalse(fn({"threshold_ok": True}))

    def test_threshold_ok_in_or(self):
        """threshold_ok in OR: at least one branch True → match."""
        fn = _parse_filter_expr(
            "threshold_ok = True OR finalize_status = 'partial'"
        )
        self.assertTrue(fn({"threshold_ok": True, "finalize_status": "done"}))
        self.assertTrue(fn({"threshold_ok": False, "finalize_status": "partial"}))
        self.assertFalse(fn({"threshold_ok": False, "finalize_status": "done"}))


class TestFinalizeStatusFilter(unittest.TestCase):
    """Filter on the finalize_status string field."""

    def test_finalize_status_eq(self):
        fn = _parse_filter_expr("finalize_status = 'done'")
        self.assertTrue(fn({"finalize_status": "done"}))
        self.assertFalse(fn({"finalize_status": "partial"}))

    def test_finalize_status_neq(self):
        fn = _parse_filter_expr("finalize_status != 'failed'")
        self.assertTrue(fn({"finalize_status": "done"}))
        self.assertTrue(fn({"finalize_status": "partial"}))
        self.assertFalse(fn({"finalize_status": "failed"}))

    def test_finalize_status_combined(self):
        fn = _parse_filter_expr(
            "finalize_status = 'partial' AND threshold_ok = False"
        )
        self.assertTrue(fn({"finalize_status": "partial", "threshold_ok": False}))
        self.assertFalse(fn({"finalize_status": "done", "threshold_ok": True}))
        self.assertFalse(fn({"finalize_status": "partial", "threshold_ok": True}))

    def test_finalize_status_failed(self):
        """'failed' status participates in = and != normally."""
        fn_eq = _parse_filter_expr("finalize_status = 'failed'")
        self.assertTrue(fn_eq({"finalize_status": "failed"}))
        self.assertFalse(fn_eq({"finalize_status": "done"}))

        fn_neq = _parse_filter_expr("finalize_status != 'failed'")
        self.assertTrue(fn_neq({"finalize_status": "done"}))
        self.assertTrue(fn_neq({"finalize_status": "partial"}))
        self.assertFalse(fn_neq({"finalize_status": "failed"}))

    def test_all_three_new_fields_in_one_filter(self):
        """AND of all 3 new fields: every clause must hold."""
        fn = _parse_filter_expr(
            "finalize_status = 'done' AND threshold_ok = True "
            "AND relative_gain >= 0.25"
        )
        # All conditions met
        self.assertTrue(fn({
            "finalize_status": "done", "threshold_ok": True,
            "relative_gain": 0.50,
        }))
        # relative_gain too low
        self.assertFalse(fn({
            "finalize_status": "done", "threshold_ok": True,
            "relative_gain": 0.10,
        }))
        # threshold not ok
        self.assertFalse(fn({
            "finalize_status": "done", "threshold_ok": False,
            "relative_gain": 0.50,
        }))
        # wrong status
        self.assertFalse(fn({
            "finalize_status": "partial", "threshold_ok": True,
            "relative_gain": 0.50,
        }))
        # relative_gain is None (qwen=0) → excluded
        self.assertFalse(fn({
            "finalize_status": "done", "threshold_ok": True,
            "relative_gain": None,
        }))


# ===================================================================
# 5a. 交付红线指标 (redline boolean fields)
# ===================================================================

class TestRedlineFilters(unittest.TestCase):
    """Filter on individual delivery-redline boolean indicators.

    Each field mirrors one check in config.check_passrate_thresholds:
      qwen_over_threshold   – qwen passrate >= 0.7
      claude_under_threshold – claude passrate <= 0.7
      claude_not_better     – claude passrate <= qwen passrate
      gain_below_threshold  – relative gain <= 0.25
    """

    # -- qwen_over_threshold --

    def test_qwen_over_threshold_true(self):
        fn = _parse_filter_expr("qwen_over_threshold = True")
        self.assertTrue(fn({"qwen_over_threshold": True}))
        self.assertFalse(fn({"qwen_over_threshold": False}))

    def test_qwen_over_threshold_false(self):
        fn = _parse_filter_expr("qwen_over_threshold = False")
        self.assertTrue(fn({"qwen_over_threshold": False}))
        self.assertFalse(fn({"qwen_over_threshold": True}))

    # -- claude_under_threshold --

    def test_claude_under_threshold_true(self):
        fn = _parse_filter_expr("claude_under_threshold = True")
        self.assertTrue(fn({"claude_under_threshold": True}))
        self.assertFalse(fn({"claude_under_threshold": False}))

    def test_claude_under_threshold_false(self):
        fn = _parse_filter_expr("claude_under_threshold = False")
        self.assertTrue(fn({"claude_under_threshold": False}))
        self.assertFalse(fn({"claude_under_threshold": True}))

    # -- claude_not_better --

    def test_claude_not_better_true(self):
        fn = _parse_filter_expr("claude_not_better = True")
        self.assertTrue(fn({"claude_not_better": True}))
        self.assertFalse(fn({"claude_not_better": False}))

    def test_claude_not_better_false(self):
        fn = _parse_filter_expr("claude_not_better = False")
        self.assertTrue(fn({"claude_not_better": False}))
        self.assertFalse(fn({"claude_not_better": True}))

    # -- gain_below_threshold --

    def test_gain_below_threshold_true(self):
        fn = _parse_filter_expr("gain_below_threshold = True")
        self.assertTrue(fn({"gain_below_threshold": True}))
        self.assertFalse(fn({"gain_below_threshold": False}))

    def test_gain_below_threshold_false(self):
        fn = _parse_filter_expr("gain_below_threshold = False")
        self.assertTrue(fn({"gain_below_threshold": False}))
        self.assertFalse(fn({"gain_below_threshold": True}))

    # -- combined: any redline violated --

    def test_any_redline_violated_or(self):
        """OR over all 4 redline fields: match if any is True."""
        fn = _parse_filter_expr(
            "qwen_over_threshold = True OR claude_under_threshold = True "
            "OR claude_not_better = True OR gain_below_threshold = True"
        )
        # all clear
        self.assertFalse(fn({
            "qwen_over_threshold": False, "claude_under_threshold": False,
            "claude_not_better": False, "gain_below_threshold": False,
        }))
        # one violated
        self.assertTrue(fn({
            "qwen_over_threshold": True, "claude_under_threshold": False,
            "claude_not_better": False, "gain_below_threshold": False,
        }))
        self.assertTrue(fn({
            "qwen_over_threshold": False, "claude_under_threshold": False,
            "claude_not_better": False, "gain_below_threshold": True,
        }))

    # -- combined: all redlines clear --

    def test_all_redlines_clear(self):
        """AND of all = False: every redline must be clear."""
        fn = _parse_filter_expr(
            "qwen_over_threshold = False AND claude_under_threshold = False "
            "AND claude_not_better = False AND gain_below_threshold = False"
        )
        self.assertTrue(fn({
            "qwen_over_threshold": False, "claude_under_threshold": False,
            "claude_not_better": False, "gain_below_threshold": False,
        }))
        self.assertFalse(fn({
            "qwen_over_threshold": False, "claude_under_threshold": True,
            "claude_not_better": False, "gain_below_threshold": False,
        }))

    # -- combined with existing fields --

    def test_redline_with_passrate(self):
        """Mix a redline field with a numeric passrate filter."""
        fn = _parse_filter_expr(
            "claude_under_threshold = True AND claude_passrate < 0.5"
        )
        self.assertTrue(fn({
            "claude_under_threshold": True, "claude_passrate": 0.40,
        }))
        self.assertFalse(fn({
            "claude_under_threshold": True, "claude_passrate": 0.80,
        }))
        self.assertFalse(fn({
            "claude_under_threshold": False, "claude_passrate": 0.40,
        }))

    def test_redline_with_finalize_status(self):
        """Redline + finalize_status compound filter."""
        fn = _parse_filter_expr(
            "finalize_status = 'partial' AND gain_below_threshold = True"
        )
        self.assertTrue(fn({
            "finalize_status": "partial", "gain_below_threshold": True,
        }))
        self.assertFalse(fn({
            "finalize_status": "done", "gain_below_threshold": True,
        }))
        self.assertFalse(fn({
            "finalize_status": "partial", "gain_below_threshold": False,
        }))

    def test_redline_neq_false(self):
        """!= False is equivalent to = True for boolean redline fields."""
        fn = _parse_filter_expr("qwen_over_threshold != False")
        self.assertTrue(fn({"qwen_over_threshold": True}))
        self.assertFalse(fn({"qwen_over_threshold": False}))

    def test_multiple_redlines_and(self):
        """AND of two correlated redlines: both must hold."""
        fn = _parse_filter_expr(
            "claude_under_threshold = True AND claude_not_better = True"
        )
        # claude 0.60, qwen 0.80 → both True
        self.assertTrue(fn({
            "claude_under_threshold": True, "claude_not_better": True,
        }))
        # claude 0.80, qwen 0.50 → under=False
        self.assertFalse(fn({
            "claude_under_threshold": False, "claude_not_better": False,
        }))
        # claude 0.65, qwen 0.60 → under=True but better=True
        self.assertFalse(fn({
            "claude_under_threshold": True, "claude_not_better": False,
        }))

    def test_redline_with_threshold_ok(self):
        """threshold_ok=False should correlate with at least one redline True."""
        fn = _parse_filter_expr(
            "threshold_ok = False AND qwen_over_threshold = True"
        )
        self.assertTrue(fn({
            "threshold_ok": False, "qwen_over_threshold": True,
        }))
        # threshold ok but qwen still flagged — unusual but logically valid
        self.assertFalse(fn({
            "threshold_ok": True, "qwen_over_threshold": True,
        }))
        # threshold not ok for other reasons, not qwen
        self.assertFalse(fn({
            "threshold_ok": False, "qwen_over_threshold": False,
        }))

    def test_redline_parenthesized(self):
        """Parentheses group OR of redlines, AND'd with task_type."""
        fn = _parse_filter_expr(
            "task_type = 'bug-fix' AND "
            "(qwen_over_threshold = True OR gain_below_threshold = True)"
        )
        self.assertTrue(fn({
            "task_type": "bug-fix",
            "qwen_over_threshold": True, "gain_below_threshold": False,
        }))
        self.assertTrue(fn({
            "task_type": "bug-fix",
            "qwen_over_threshold": False, "gain_below_threshold": True,
        }))
        # wrong task_type
        self.assertFalse(fn({
            "task_type": "feature",
            "qwen_over_threshold": True, "gain_below_threshold": False,
        }))
        # right task_type but no redline violated
        self.assertFalse(fn({
            "task_type": "bug-fix",
            "qwen_over_threshold": False, "gain_below_threshold": False,
        }))

    def test_redline_with_relative_gain(self):
        """Combine numeric relative_gain with boolean redline field."""
        fn = _parse_filter_expr(
            "relative_gain < 0.25 AND gain_below_threshold = True"
        )
        self.assertTrue(fn({
            "relative_gain": 0.10, "gain_below_threshold": True,
        }))
        # gain ok
        self.assertFalse(fn({
            "relative_gain": 0.50, "gain_below_threshold": False,
        }))
        # relative_gain=None (qwen=0) → numeric side returns False
        self.assertFalse(fn({
            "relative_gain": None, "gain_below_threshold": True,
        }))

    def test_redline_in_apply_filter_mixed_records(self):
        """_apply_filter with a mix of clear/violated/missing redlines."""
        records = [
            {"id": "CT-0001", "qwen_over_threshold": False,
             "claude_under_threshold": False, "claude_not_better": False,
             "gain_below_threshold": False},
            {"id": "CT-0002", "qwen_over_threshold": True,
             "claude_under_threshold": False, "claude_not_better": False,
             "gain_below_threshold": False},
            {"id": "CT-0003", "qwen_over_threshold": False,
             "claude_under_threshold": True, "claude_not_better": True,
             "gain_below_threshold": True},
            {"id": "CT-0004"},  # fields missing entirely
        ]
        fn = _parse_filter_expr("qwen_over_threshold = True")
        result = _apply_filter(records, fn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "CT-0002")

        fn_any = _parse_filter_expr(
            "qwen_over_threshold = True OR claude_under_threshold = True "
            "OR claude_not_better = True OR gain_below_threshold = True"
        )
        result_any = _apply_filter(records, fn_any)
        self.assertEqual(len(result_any), 2)
        ids = {r["id"] for r in result_any}
        self.assertEqual(ids, {"CT-0002", "CT-0003"})


# ===================================================================
# 5b. Boundary: empty / None values must never match
# ===================================================================

class TestEmptyValueBoundary(unittest.TestCase):
    """None and empty-string field values are treated as 'no data':
    they must not match ANY comparison operator, never raise, and
    never be accidentally selected."""

    # -- empty string (finalize_status not yet set) ------------------

    def test_empty_string_eq_no_match(self):
        fn = _parse_filter_expr("finalize_status = 'done'")
        self.assertFalse(fn({"finalize_status": ""}))

    def test_empty_string_neq_no_match(self):
        """Even != must NOT select an empty value."""
        fn = _parse_filter_expr("finalize_status != 'done'")
        self.assertFalse(fn({"finalize_status": ""}))

    def test_empty_string_lt_no_match(self):
        fn = _parse_filter_expr("finalize_status < 'z'")
        self.assertFalse(fn({"finalize_status": ""}))

    def test_empty_string_eq_empty_literal(self):
        """Explicitly comparing to '' must still return False (no data)."""
        fn = _parse_filter_expr("finalize_status = ''")
        self.assertFalse(fn({"finalize_status": ""}))

    # -- None values (relative_gain, passrates, missing fields) ------

    def test_none_eq_no_match(self):
        fn = _parse_filter_expr("relative_gain = 0.5")
        self.assertFalse(fn({"relative_gain": None}))

    def test_none_neq_no_match(self):
        fn = _parse_filter_expr("relative_gain != 0.5")
        self.assertFalse(fn({"relative_gain": None}))

    def test_none_gt_no_match(self):
        fn = _parse_filter_expr("relative_gain > 0")
        self.assertFalse(fn({"relative_gain": None}))

    def test_missing_field_no_match(self):
        fn = _parse_filter_expr("relative_gain >= 0")
        self.assertFalse(fn({}))

    # -- False is a real value, NOT empty ----------------------------

    def test_false_is_not_empty(self):
        """threshold_ok=False is a valid value and must participate."""
        fn = _parse_filter_expr("threshold_ok = False")
        self.assertTrue(fn({"threshold_ok": False}))

    def test_false_neq_true(self):
        fn = _parse_filter_expr("threshold_ok != True")
        self.assertTrue(fn({"threshold_ok": False}))

    # -- mixed: valid record + empty record --------------------------

    def test_apply_filter_skips_empty_values(self):
        """_apply_filter on a mix of valid and empty records."""
        records = [
            {"finalize_status": "done", "threshold_ok": True,
             "relative_gain": 0.5},
            {"finalize_status": "", "threshold_ok": False,
             "relative_gain": None},       # not yet finalized
            {"finalize_status": "partial", "threshold_ok": False,
             "relative_gain": 0.1},
        ]
        fn = _parse_filter_expr("finalize_status != 'done'")
        result = _apply_filter(records, fn)
        # Only the 'partial' record matches; the '' record is excluded.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["finalize_status"], "partial")

    def test_apply_filter_relative_gain_skips_none(self):
        records = [
            {"relative_gain": 0.5, "qwen_passrate": 0.4,
             "claude_passrate": 0.6},
            {"relative_gain": None, "qwen_passrate": 0.0,
             "claude_passrate": 0.5},      # qwen=0 → no gain
            {"relative_gain": 0.1, "qwen_passrate": 0.5,
             "claude_passrate": 0.55},
        ]
        fn = _parse_filter_expr("relative_gain < 0.25")
        result = _apply_filter(records, fn)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["relative_gain"], 0.1)


# ===================================================================
# 6. show_stats end-to-end
# ===================================================================

class TestShowStatsEndToEnd(unittest.TestCase):
    """Integration tests: build real state, call show_stats, check output."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patcher = patch.object(
            BatchConfig, "base_dir",
            new_callable=PropertyMock, return_value=Path(self._tmpdir.name),
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def _make_tasks_and_state(self):
        tasks = [
            make_task("CT-0001", "bug-fix", "cli", "python"),
            make_task("CT-0002", "feature", "web", "python"),
            make_task("CT-0003", "bug-fix", "web", "java"),
        ]
        config = build_config(tasks=tasks)
        _write_state(config.state_path, [
            {"task_id": "CT-0001", "qwen_passrate": 0.3,
             "claude_passrate": 0.8, "status": "done"},
            {"task_id": "CT-0002", "qwen_passrate": 0.6,
             "claude_passrate": 0.9, "status": "done"},
            {"task_id": "CT-0003", "qwen_passrate": 0.4,
             "claude_passrate": 0.85, "status": "done"},
        ])
        return config

    def test_filter_only(self):
        config = self._make_tasks_and_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = show_stats(
                config, fmt="json",
                filter_expr="task_type = 'bug-fix'",
            )
        output = json.loads(buf.getvalue())

        # Filtered to 2 bug-fix tasks; standard stats JSON has per_task
        self.assertIn("per_task", output)
        self.assertEqual(len(output["per_task"]), 2)
        self.assertIn("CT-0001", output["per_task"])
        self.assertIn("CT-0003", output["per_task"])
        self.assertNotIn("CT-0002", output["per_task"])
        self.assertTrue(result)

    def test_group_by_only_table(self):
        config = self._make_tasks_and_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = show_stats(config, group_by="task_type")
        output = buf.getvalue()

        self.assertIn("PASSRATE BY TASK_TYPE", output)
        self.assertIn("bug-fix (2)", output)
        self.assertIn("feature (1)", output)
        self.assertIn("Total: 3 task(s)", output)
        self.assertIn("All stages OK", output)
        self.assertTrue(result)

    def test_group_by_only_json(self):
        config = self._make_tasks_and_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = show_stats(config, fmt="json", group_by="task_type")
        output = json.loads(buf.getvalue())

        self.assertIn("groupings", output)
        self.assertEqual(len(output["groupings"]), 1)
        g = output["groupings"][0]
        self.assertEqual(g["group_key"], "task_type")
        self.assertEqual(g["count"], 3)
        self.assertEqual(len(g["groups"]), 2)

        # Verify structure of each group element
        for grp in g["groups"]:
            self.assertIn("group_value", grp)
            self.assertIn("count", grp)
            self.assertIn("passrate_stats", grp)
            for model_stats in grp["passrate_stats"].values():
                self.assertIn("min", model_stats)
                self.assertIn("max", model_stats)
                self.assertIn("mean", model_stats)
                self.assertIn("median", model_stats)
                self.assertIn("std", model_stats)
                self.assertIn("count", model_stats)
        self.assertTrue(result)

    def test_filter_and_group_by_combined(self):
        config = self._make_tasks_and_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = show_stats(
                config, fmt="json",
                filter_expr="task_type = 'bug-fix'",
                group_by="domain",
            )
        output = json.loads(buf.getvalue())

        self.assertIn("groupings", output)
        self.assertEqual(output["filter"], "task_type = 'bug-fix'")
        self.assertEqual(output["total_before_filter"], 3)
        self.assertEqual(output["total_after_filter"], 2)

        g = output["groupings"][0]
        self.assertEqual(g["group_key"], "domain")
        values = {grp["group_value"] for grp in g["groups"]}
        self.assertEqual(values, {"cli", "web"})
        self.assertTrue(result)

    def test_group_by_multiple_fields_json(self):
        config = self._make_tasks_and_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = show_stats(
                config, fmt="json",
                group_by="task_type,domain",
            )
        output = json.loads(buf.getvalue())

        groupings = output["groupings"]
        self.assertEqual(len(groupings), 2)
        self.assertEqual(groupings[0]["group_key"], "task_type")
        self.assertEqual(groupings[1]["group_key"], "domain")
        self.assertTrue(result)

    def test_filter_no_match(self):
        config = self._make_tasks_and_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = show_stats(
                config, fmt="json",
                filter_expr="qwen_passrate > 0.99",
            )
        output = json.loads(buf.getvalue())

        self.assertEqual(output["total_after_filter"], 0)
        self.assertIn("No tasks match", output["message"])
        self.assertTrue(result)

    def test_illegal_field_in_show_stats(self):
        config = self._make_tasks_and_state()
        with self.assertRaises(ValueError):
            show_stats(config, filter_expr="secret_field = 'x'")

    def test_no_filter_no_group_by_unchanged(self):
        """Without filter/group-by the original stats output is preserved."""
        config = self._make_tasks_and_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = show_stats(config, fmt="json")
        output = json.loads(buf.getvalue())

        # Original format: summary + per_task, no groupings
        self.assertIn("summary", output)
        self.assertIn("per_task", output)
        self.assertNotIn("groupings", output)
        self.assertTrue(result)


class TestBoundaryEndToEnd(unittest.TestCase):
    """End-to-end: empty/None values are excluded from filter in both outputs."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patcher = patch.object(
            BatchConfig, "base_dir",
            new_callable=PropertyMock, return_value=Path(self._tmpdir.name),
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def _make_mixed_state(self):
        """Three tasks spanning edge cases:
          CT-0001: normal, relative_gain computable, threshold_ok=True
          CT-0002: qwen=0 → relative_gain=None, threshold_ok=False
          CT-0003: finalize not yet run (status=''), all derived fields empty
        """
        tasks = [
            make_task("CT-0001", "bug-fix", "cli", "python"),
            make_task("CT-0002", "feature", "web", "python"),
            make_task("CT-0003", "bug-fix", "web", "java"),
        ]
        config = build_config(tasks=tasks)
        state = PipelineState(config.state_path)
        # CT-0001: normal
        state.set("CT-0001", "prepare", status="done")
        state.set("CT-0001", "finalize", status="done",
                  qwen_passrate=0.4, claude_passrate=0.6, threshold_ok=True)
        state.set("CT-0001", "validate", status="done")
        for m in ("qwen", "claude"):
            state.set("CT-0001", "run", model=m, status="done")
            state.set("CT-0001", "collect", model=m, status="done")
            state.set("CT-0001", "score", model=m, status="done")
        # CT-0002: qwen=0 → relative_gain=None
        state.set("CT-0002", "prepare", status="done")
        state.set("CT-0002", "finalize", status="partial",
                  qwen_passrate=0.0, claude_passrate=0.5, threshold_ok=False)
        state.set("CT-0002", "validate", status="done")
        for m in ("qwen", "claude"):
            state.set("CT-0002", "run", model=m, status="done")
            state.set("CT-0002", "collect", model=m, status="done")
            state.set("CT-0002", "score", model=m, status="done")
        # CT-0003: finalize not yet run — status is empty
        state.set("CT-0003", "prepare", status="done")
        # finalize state intentionally NOT set
        state.set("CT-0003", "validate", status="done")
        for m in ("qwen", "claude"):
            state.set("CT-0003", "run", model=m, status="done")
            state.set("CT-0003", "collect", model=m, status="done")
            state.set("CT-0003", "score", model=m, status="done")
        return config

    # -- JSON output ------------------------------------------------

    def test_relative_gain_filter_json_excludes_none(self):
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(config, fmt="json",
                       filter_expr="relative_gain >= 0.25")
        output = json.loads(buf.getvalue())
        # CT-0001: gain=0.5 ✓ | CT-0002: None ✗ | CT-0003: None ✗
        per_task = output["per_task"]
        self.assertEqual(len(per_task), 1)
        self.assertIn("CT-0001", per_task)
        self.assertNotIn("CT-0002", per_task)
        self.assertNotIn("CT-0003", per_task)

    def test_finalize_status_filter_json_excludes_empty(self):
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(config, fmt="json",
                       filter_expr="finalize_status != 'done'")
        output = json.loads(buf.getvalue())
        # CT-0001: 'done' → excluded | CT-0002: 'partial' ✓
        # CT-0003: '' (no data) → excluded, NOT mistakenly selected
        per_task = output["per_task"]
        self.assertEqual(len(per_task), 1)
        self.assertIn("CT-0002", per_task)
        self.assertNotIn("CT-0001", per_task)
        self.assertNotIn("CT-0003", per_task)

    def test_threshold_ok_filter_json(self):
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(config, fmt="json",
                       filter_expr="threshold_ok = True")
        output = json.loads(buf.getvalue())
        # CT-0001: True ✓ | CT-0002: False ✗ | CT-0003: False ✗
        per_task = output["per_task"]
        self.assertEqual(len(per_task), 1)
        self.assertIn("CT-0001", per_task)

    # -- Table output -----------------------------------------------

    def test_relative_gain_filter_table_excludes_none(self):
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(config, fmt="table",
                       filter_expr="relative_gain >= 0.25",
                       group_by="task_type")
        output = buf.getvalue()
        # Only CT-0001 survives: bug-fix (1)
        self.assertIn("bug-fix (1)", output)
        self.assertNotIn("feature", output)
        self.assertIn("Total: 1 task(s)", output)

    def test_finalize_status_filter_table_excludes_empty(self):
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(config, fmt="table",
                       filter_expr="finalize_status = 'partial'",
                       group_by="task_type")
        output = buf.getvalue()
        # Only CT-0002 (feature, partial) survives
        self.assertIn("feature (1)", output)
        self.assertIn("Total: 1 task(s)", output)

    def test_combined_filter_table_no_false_positives(self):
        """AND filter: empty relative_gain must not leak through."""
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(config, fmt="table",
                       filter_expr=(
                           "threshold_ok = False AND relative_gain < 0.5"
                       ),
                       group_by="domain")
        output = buf.getvalue()
        # CT-0002: threshold_ok=False ✓, relative_gain=None → excluded
        # CT-0003: threshold_ok=False ✓, relative_gain=None → excluded
        # Result: 0 tasks
        self.assertIn("No tasks match", output)

    def test_group_by_threshold_ok_json(self):
        """group-by on threshold_ok: splits tasks into True/False buckets."""
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(config, fmt="json", group_by="threshold_ok")
        output = json.loads(buf.getvalue())
        g = output["groupings"][0]
        self.assertEqual(g["group_key"], "threshold_ok")
        values = {grp["group_value"] for grp in g["groups"]}
        self.assertEqual(values, {True, False})

    def test_all_three_fields_filter_json(self):
        """AND of all 3 new fields in a single show_stats call."""
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(
                config, fmt="json",
                filter_expr=(
                    "finalize_status = 'done' AND threshold_ok = True "
                    "AND relative_gain >= 0.25"
                ),
            )
        output = json.loads(buf.getvalue())
        per_task = output["per_task"]
        # Only CT-0001 satisfies all three conditions.
        self.assertEqual(len(per_task), 1)
        self.assertIn("CT-0001", per_task)

    def test_or_filter_with_new_field_json(self):
        """OR with threshold_ok: picks up either-ok or partial tasks."""
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(
                config, fmt="json",
                filter_expr=(
                    "threshold_ok = True OR finalize_status = 'partial'"
                ),
            )
        output = json.loads(buf.getvalue())
        per_task = output["per_task"]
        # CT-0001: threshold_ok=True ✓
        # CT-0002: finalize_status='partial' ✓
        # CT-0003: both False/'' → excluded
        self.assertEqual(len(per_task), 2)
        self.assertIn("CT-0001", per_task)
        self.assertIn("CT-0002", per_task)
        self.assertNotIn("CT-0003", per_task)


    def test_redline_filter_json_no_match(self):
        """Filter on qwen_over_threshold=True when no task has qwen >= 0.7.

        _make_mixed_state: CT-0001 qwen=0.4, CT-0002 qwen=0.0, CT-0003 no data.
        All have qwen_over_threshold=False → filter returns empty.
        show_stats emits a 'No tasks match' JSON envelope instead of per_task.
        """
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(
                config, fmt="json",
                filter_expr="qwen_over_threshold = True",
            )
        output = json.loads(buf.getvalue())
        self.assertEqual(output["total_after_filter"], 0)
        self.assertIn("No tasks match", output["message"])

    def test_redline_claude_under_filter_json(self):
        """Filter on claude_under_threshold = True.

        CT-0001: claude=0.6 <= 0.7 → selected
        CT-0002: claude=0.5 <= 0.7 → selected
        CT-0003: no finalize data → excluded
        """
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(
                config, fmt="json",
                filter_expr="claude_under_threshold = True",
            )
        output = json.loads(buf.getvalue())
        per_task = output["per_task"]
        ids = set(per_task.keys())
        self.assertIn("CT-0001", ids)  # claude 0.6 <= 0.7
        self.assertIn("CT-0002", ids)  # claude 0.5 <= 0.7
        self.assertNotIn("CT-0003", ids)  # no data → excluded

    def test_redline_gain_below_filter_table(self):
        """Filter on gain_below_threshold=True in table output.

        CT-0001: gain=(0.6-0.4)/0.4=0.50 > 0.25 → not flagged
        CT-0002: qwen=0, claude=0.50 >= 0.25 → not flagged
        Neither has gain_below_threshold=True → empty result.
        """
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(
                config, fmt="table",
                filter_expr="gain_below_threshold = True",
            )
        table = buf.getvalue()
        self.assertIn("No tasks match", table)
        self.assertNotIn("CT-0001", table)
        self.assertNotIn("CT-0002", table)
        self.assertNotIn("CT-0003", table)

    def test_redline_all_clear_filter_json(self):
        """AND of all four redlines = False.

        CT-0001: claude=0.6 <= 0.7 → claude_under_threshold=True → excluded
        CT-0002: claude=0.5 <= 0.7 → claude_under_threshold=True → excluded
        CT-0003: no finalize data → all redlines False → selected
        """
        config = self._make_mixed_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            show_stats(
                config, fmt="json",
                filter_expr=(
                    "qwen_over_threshold = False AND "
                    "claude_under_threshold = False AND "
                    "claude_not_better = False AND "
                    "gain_below_threshold = False"
                ),
            )
        output = json.loads(buf.getvalue())
        per_task = output["per_task"]
        # CT-0001 has claude_under_threshold=True (0.6 <= 0.7) → excluded
        self.assertNotIn("CT-0001", per_task)
        # CT-0002 has claude_under_threshold=True (0.5 <= 0.7) → excluded
        self.assertNotIn("CT-0002", per_task)
        # CT-0003 has no data → all redlines default to False → matches
        self.assertIn("CT-0003", per_task)


if __name__ == "__main__":
    unittest.main()
