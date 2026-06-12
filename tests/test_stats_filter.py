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

from ctpipe.config import TaskConfig
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
        }
        self.assertEqual(set(_FILTER_FIELD_WHITELIST), expected)

    def test_all_whitelisted_fields_parse(self):
        for field in _FILTER_FIELD_WHITELIST:
            fn = _parse_filter_expr(f"{field} = 'x'")
            self.assertTrue(callable(fn), f"{field} should produce a callable")

    def test_internal_record_fields_rejected(self):
        """Fields stored in records but not whitelisted must be blocked."""
        for field in ("id", "finalize_status", "threshold_ok", "task_title"):
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
# 5. show_stats end-to-end
# ===================================================================

class TestShowStatsEndToEnd(unittest.TestCase):
    """Integration tests: build real state, call show_stats, check output."""

    def _make_tasks_and_state(self, tmp: Path):
        tasks = [
            make_task("CT-0001", "bug-fix", "cli", "python"),
            make_task("CT-0002", "feature", "web", "python"),
            make_task("CT-0003", "bug-fix", "web", "java"),
        ]
        config = build_config(tasks=tasks)
        # Point delivery_dir at a unique temp-scoped directory so no real
        # tasks.json manifest is loaded.  Use tmp.name as delivery_date
        # (validation is skipped when setting the attribute directly).
        config.delivery_date = tmp.name
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
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_tasks_and_state(Path(tmp))
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
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_tasks_and_state(Path(tmp))
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
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_tasks_and_state(Path(tmp))
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
                    self.assertIn("count", model_stats)
            self.assertTrue(result)

    def test_filter_and_group_by_combined(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_tasks_and_state(Path(tmp))
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
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_tasks_and_state(Path(tmp))
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
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_tasks_and_state(Path(tmp))
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
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_tasks_and_state(Path(tmp))
            with self.assertRaises(ValueError):
                show_stats(config, filter_expr="secret_field = 'x'")

    def test_no_filter_no_group_by_unchanged(self):
        """Without filter/group-by the original stats output is preserved."""
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_tasks_and_state(Path(tmp))
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = show_stats(config, fmt="json")
            output = json.loads(buf.getvalue())

            # Original format: summary + per_task, no groupings
            self.assertIn("summary", output)
            self.assertIn("per_task", output)
            self.assertNotIn("groupings", output)
            self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
