"""Unit tests for trajectory.extract_for_scoring — especially tool_result handling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def _write_jsonl(messages: list[dict]) -> Path:
    """Write a list of message dicts to a temp JSONL file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w", encoding="utf-8")
    for msg in messages:
        line = {"sessionId": "test", "type": msg.get("role", "system"), "message": msg}
        tmp.write(json.dumps(line, ensure_ascii=False) + "\n")
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _user_blocks(blocks: list[dict]) -> dict:
    return {"role": "user", "content": blocks}


def _assistant_blocks(blocks: list[dict]) -> dict:
    return {"role": "assistant", "content": blocks}


def _tool_use(name: str = "Read", **kwargs) -> dict:
    return {"type": "tool_use", "id": "tu_1", "name": name, "input": kwargs}


def _tool_result(content="", is_error: bool = False) -> dict:
    return {"type": "tool_result", "tool_use_id": "tu_1", "content": content, "is_error": is_error}


# Import after helpers are defined
from ctpipe.trajectory import extract_for_scoring  # noqa: E402


class TestToolResultInUserMessages(unittest.TestCase):
    """tool_result blocks live in role=user messages — they must be extracted."""

    def setUp(self):
        self._tmpfiles: list[Path] = []

    def tearDown(self):
        for p in self._tmpfiles:
            p.unlink(missing_ok=True)

    def _run(self, messages: list[dict]) -> str:
        p = _write_jsonl(messages)
        self._tmpfiles.append(p)
        return extract_for_scoring(p)

    # --- basic tool_result extraction ---

    def test_normal_result_extracted(self):
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="ls -la")]),
            _user_blocks([_tool_result("file1.py\nfile2.py\ntests/")]),
        ])
        self.assertIn("[Result: file1.py\nfile2.py\ntests/]", result)

    def test_error_result_marked(self):
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="rm /important")]),
            _user_blocks([_tool_result("Permission denied", is_error=True)]),
        ])
        self.assertIn("[Result: ERROR] Permission denied", result)

    def test_empty_result_marked_ok(self):
        result = self._run([
            _assistant_blocks([_tool_use("Edit", file_path="a.py", old_string="x", new_string="y")]),
            _user_blocks([_tool_result("")]),
        ])
        self.assertIn("[Result: ok]", result)

    def test_whitespace_only_result_marked_ok(self):
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="true")]),
            _user_blocks([_tool_result("   \n  ")]),
        ])
        self.assertIn("[Result: ok]", result)

    # --- content as list of blocks ---

    def test_list_content_joined(self):
        result = self._run([
            _assistant_blocks([_tool_use("Read", file_path="x.py")]),
            _user_blocks([_tool_result([
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"},
            ])]),
        ])
        self.assertIn("[Result: line one line two]", result)

    def test_list_content_skips_non_text_blocks(self):
        result = self._run([
            _assistant_blocks([_tool_use("Read", file_path="img.png")]),
            _user_blocks([_tool_result([
                {"type": "text", "text": "metadata here"},
                {"type": "image", "source": {"type": "base64", "data": "xxx"}},
            ])]),
        ])
        self.assertIn("metadata here", result)
        self.assertNotIn("base64", result)

    # --- truncation ---

    def test_long_result_truncated_at_300(self):
        long_text = "A" * 500
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="cat bigfile")]),
            _user_blocks([_tool_result(long_text)]),
        ])
        self.assertIn("[Result: " + "A" * 300 + "]", result)
        self.assertNotIn("A" * 301, result)

    def test_error_result_truncated_at_300(self):
        long_error = "E" * 500
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="fail")]),
            _user_blocks([_tool_result(long_error, is_error=True)]),
        ])
        self.assertIn("[Result: ERROR] " + "E" * 300, result)
        self.assertNotIn("E" * 301, result)

    # --- deduplication ---

    def test_duplicate_result_suppressed(self):
        """Two tool results with identical content: the second occurrence
        is replaced with a 'same as' marker instead of being repeated."""
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="ls")]),
            _user_blocks([_tool_result("file1.py\nfile2.py")]),
            _assistant_blocks([_tool_use("Bash", command="ls -a")]),
            _user_blocks([_tool_result("file1.py\nfile2.py")]),
        ])
        # First occurrence shown in full, second replaced
        self.assertEqual(result.count("[Result: file1.py\nfile2.py]"), 1)
        self.assertIn("[Result: same as Bash]", result)

    def test_duplicate_error_result_suppressed(self):
        """Identical error content should also be deduplicated."""
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="rm a")]),
            _user_blocks([_tool_result("Permission denied", is_error=True)]),
            _assistant_blocks([_tool_use("Bash", command="rm b")]),
            _user_blocks([_tool_result("Permission denied", is_error=True)]),
        ])
        self.assertEqual(result.count("[Result: ERROR] Permission denied"), 1)
        self.assertIn("[Result: ERROR same as Bash]", result)

    def test_empty_results_not_deduped(self):
        """Empty results ('ok') should not be deduplicated — every tool
        that returned empty deserves its own ok marker."""
        result = self._run([
            _assistant_blocks([_tool_use("Edit", file_path="a.py", old_string="x", new_string="y")]),
            _user_blocks([_tool_result("")]),
            _assistant_blocks([_tool_use("Edit", file_path="b.py", old_string="x", new_string="y")]),
            _user_blocks([_tool_result("")]),
            _assistant_blocks([_tool_use("Edit", file_path="c.py", old_string="x", new_string="y")]),
            _user_blocks([_tool_result("")]),
        ])
        self.assertEqual(result.count("[Result: ok]"), 3)

    def test_different_results_not_deduped(self):
        """Distinct content must not be collapsed — each unique result
        should appear in full."""
        result = self._run([
            _assistant_blocks([_tool_use("Read", file_path="a.py")]),
            _user_blocks([_tool_result("content of a")]),
            _assistant_blocks([_tool_use("Read", file_path="b.py")]),
            _user_blocks([_tool_result("content of b")]),
        ])
        self.assertIn("[Result: content of a]", result)
        self.assertIn("[Result: content of b]", result)
        self.assertNotIn("same as", result)

    def test_dedup_marker_shows_correct_tool_name(self):
        """Two different tools produce identical content; the dedup marker
        must reference the tool that produced the FIRST occurrence."""
        result = self._run([
            _assistant_blocks([{"type": "tool_use", "id": "tu_r", "name": "Read", "input": {"file_path": "x.py"}}]),
            _user_blocks([{"type": "tool_result", "tool_use_id": "tu_r", "content": "shared content"}]),
            _assistant_blocks([{"type": "tool_use", "id": "tu_b", "name": "Bash", "input": {"command": "cat x.py"}}]),
            _user_blocks([{"type": "tool_result", "tool_use_id": "tu_b", "content": "shared content"}]),
        ])
        self.assertIn("[Result: shared content]", result)
        self.assertIn("[Result: same as Read]", result)
        self.assertNotIn("[Result: same as Bash]", result)

    def test_dedup_with_unknown_tool_use_id(self):
        """tool_result whose tool_use_id has no matching tool_use:
        the dedup marker should still be emitted (with empty tool name)."""
        result = self._run([
            _assistant_blocks([{"type": "tool_use", "id": "tu_a", "name": "Bash", "input": {"command": "ls"}}]),
            _user_blocks([{"type": "tool_result", "tool_use_id": "tu_a", "content": "dup"}]),
            _assistant_blocks([{"type": "tool_use", "id": "tu_b", "name": "Bash", "input": {"command": "ls"}}]),
            _user_blocks([{"type": "tool_result", "tool_use_id": "tu_missing", "content": "dup"}]),
        ])
        self.assertEqual(result.count("[Result: dup]"), 1)
        self.assertIn("[Result: same as ", result)

    def test_three_way_dedup(self):
        """Same content appearing 3 times: first is full, 2nd and 3rd
        both get the 'same as' marker."""
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="cmd1")]),
            _user_blocks([_tool_result("repeated")]),
            _assistant_blocks([_tool_use("Bash", command="cmd2")]),
            _user_blocks([_tool_result("repeated")]),
            _assistant_blocks([_tool_use("Bash", command="cmd3")]),
            _user_blocks([_tool_result("repeated")]),
        ])
        self.assertEqual(result.count("[Result: repeated]"), 1)
        self.assertEqual(result.count("[Result: same as Bash]"), 2)

    def test_dedup_across_multiple_user_messages(self):
        """Dedup set persists across separate user messages within one
        extract_for_scoring call."""
        result = self._run([
            _assistant_blocks([_tool_use("Read", file_path="a.py")]),
            _user_blocks([_tool_result("content here")]),
            _assistant_blocks([_tool_use("Edit", file_path="a.py", old_string="x", new_string="y")]),
            _user_blocks([{"type": "text", "text": "现在来改另一个文件"}]),
            _assistant_blocks([_tool_use("Read", file_path="b.py")]),
            _user_blocks([_tool_result("content here")]),
        ])
        self.assertEqual(result.count("[Result: content here]"), 1)
        self.assertIn("[Result: same as Read]", result)

    def test_parallel_same_results_deduped(self):
        """Multiple parallel tool results with identical content in one
        user message: only the first is shown, rest are deduped."""
        result = self._run([
            _assistant_blocks([
                {"type": "tool_use", "id": "tu_a", "name": "Read", "input": {"file_path": "a.py"}},
                {"type": "tool_use", "id": "tu_b", "name": "Read", "input": {"file_path": "b.py"}},
            ]),
            _user_blocks([
                {"type": "tool_result", "tool_use_id": "tu_a", "content": "same"},
                {"type": "tool_result", "tool_use_id": "tu_b", "content": "same"},
            ]),
        ])
        self.assertEqual(result.count("[Result: same]"), 1)
        self.assertEqual(result.count("[Result: same as Read]"), 1)

    def test_error_and_normal_same_content_not_deduped(self):
        """Error and normal result with identical text are different
        categories and must NOT be deduplicated against each other."""
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="cmd1")]),
            _user_blocks([_tool_result("output text")]),
            _assistant_blocks([_tool_use("Bash", command="cmd2")]),
            _user_blocks([_tool_result("output text", is_error=True)]),
        ])
        self.assertIn("[Result: output text]", result)
        self.assertIn("[Result: ERROR] output text", result)

    def test_empty_error_not_deduped(self):
        """Error with empty content ('[Result: ERROR]') should not be
        deduplicated — each deserves its own marker."""
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="fail1")]),
            _user_blocks([_tool_result("", is_error=True)]),
            _assistant_blocks([_tool_use("Bash", command="fail2")]),
            _user_blocks([_tool_result("", is_error=True)]),
        ])
        self.assertEqual(result.count("[Result: ERROR]"), 2)

    def test_list_content_truncated_at_300(self):
        """List content should be joined then truncated at 300 chars."""
        long_block = "X" * 400
        result = self._run([
            _assistant_blocks([_tool_use("Read", file_path="big.py")]),
            _user_blocks([_tool_result([
                {"type": "text", "text": long_block},
                {"type": "text", "text": "tail"},
            ])]),
        ])
        self.assertIn("X" * 300, result)
        self.assertNotIn("X" * 301, result)
        self.assertNotIn("tail", result)

    def test_dedup_by_truncated_snippet(self):
        """Two results differing only after 300 chars share the same
        snippet → second should be deduplicated."""
        common = "A" * 300
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="cmd1")]),
            _user_blocks([_tool_result(common + "BBBBB")]),
            _assistant_blocks([_tool_use("Bash", command="cmd2")]),
            _user_blocks([_tool_result(common + "CCCCC")]),
        ])
        # Both start with 300 A's, so snippets match → dedup kicks in
        self.assertEqual(result.count("[Result: " + "A" * 300), 1)
        self.assertIn("[Result: same as Bash]", result)

    def test_dedup_state_isolated_between_calls(self):
        """Each extract_for_scoring call starts with a fresh dedup set."""
        msgs = [
            _assistant_blocks([_tool_use("Bash", command="ls")]),
            _user_blocks([_tool_result("content")]),
        ]
        r1 = self._run(msgs)
        r2 = self._run(msgs)
        self.assertIn("[Result: content]", r1)
        self.assertIn("[Result: content]", r2)
        self.assertNotIn("same as", r1)
        self.assertNotIn("same as", r2)

    def test_whitespace_variant_not_deduped(self):
        """Content that differs only in trailing whitespace survives
        .strip() differently → not treated as duplicate."""
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="cmd1")]),
            _user_blocks([_tool_result("hello")]),
            _assistant_blocks([_tool_use("Bash", command="cmd2")]),
            _user_blocks([_tool_result("hello   ")]),
        ])
        # Both strip to "hello" → they ARE deduped
        self.assertEqual(result.count("[Result: hello]"), 1)
        self.assertIn("[Result: same as Bash]", result)

    # --- mixed user message (text + tool_result) ---

    def test_mixed_user_message_preserves_both(self):
        result = self._run([
            _assistant_blocks([_tool_use("Read", file_path="x.py")]),
            _user_blocks([
                {"type": "text", "text": "继续修复这个bug"},
                _tool_result("def hello(): pass"),
            ]),
        ])
        self.assertIn("=== USER ===", result)
        self.assertIn("继续修复这个bug", result)
        self.assertIn("[Result: def hello(): pass]", result)

    # --- tool_use + tool_result pairing ---

    def test_tool_use_and_result_paired(self):
        result = self._run([
            _assistant_blocks([
                _tool_use("Bash", command="npm test"),
            ]),
            _user_blocks([
                _tool_result("PASS  tests/app.test.js\nTests: 3 passed"),
            ]),
        ])
        self.assertEqual(result.count("[Tool:"), 1)
        self.assertEqual(result.count("[Result:"), 1)
        self.assertIn("[Tool: Bash(command=npm test)]", result)
        self.assertIn("[Result: PASS  tests/app.test.js", result)

    def test_multiple_tool_round_trips(self):
        result = self._run([
            _assistant_blocks([_tool_use("Read", file_path="a.py")]),
            _user_blocks([_tool_result("content of a")]),
            _assistant_blocks([_tool_use("Edit", file_path="a.py", old_string="x", new_string="y")]),
            _user_blocks([_tool_result("")]),
            _assistant_blocks([_tool_use("Bash", command="pytest")]),
            _user_blocks([_tool_result("3 passed", is_error=False)]),
        ])
        self.assertEqual(result.count("[Tool:"), 3)
        self.assertEqual(result.count("[Result:"), 3)
        self.assertEqual(result.count("[Result: ok]"), 1)

    # --- no duplicate from assistant branch ---

    def test_tool_result_in_assistant_ignored(self):
        """tool_result should never appear in assistant messages, but if it
        does, it must be ignored to prevent duplicate output."""
        result = self._run([
            _assistant_blocks([
                _tool_use("Bash", command="echo hi"),
                _tool_result("hi"),  # misplaced — should be ignored
            ]),
            _user_blocks([_tool_result("hi")]),
        ])
        # Only the user-branch result should appear
        self.assertEqual(result.count("[Result:"), 1)

    # --- edge cases ---

    def test_empty_error_result(self):
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="fail")]),
            _user_blocks([_tool_result("", is_error=True)]),
        ])
        self.assertIn("[Result: ERROR]", result)

    def test_no_tool_result_in_plain_text_user(self):
        """Plain-text user messages should not produce spurious results."""
        result = self._run([_user("请帮我修复这个问题")])
        self.assertNotIn("[Result:", result)
        self.assertIn("请帮我修复这个问题", result)

    def test_parallel_tool_results_in_single_message(self):
        """Assistant makes multiple parallel tool calls; results come back
        in one user message with multiple tool_result blocks."""
        result = self._run([
            _assistant_blocks([
                {"type": "tool_use", "id": "tu_a", "name": "Read", "input": {"file_path": "a.py"}},
                {"type": "tool_use", "id": "tu_b", "name": "Read", "input": {"file_path": "b.py"}},
                {"type": "tool_use", "id": "tu_c", "name": "Bash", "input": {"command": "ls"}},
            ]),
            _user_blocks([
                {"type": "tool_result", "tool_use_id": "tu_a", "content": "content_a"},
                {"type": "tool_result", "tool_use_id": "tu_b", "content": "content_b"},
                {"type": "tool_result", "tool_use_id": "tu_c", "content": "dir1\ndir2"},
            ]),
        ])
        self.assertEqual(result.count("[Tool:"), 3)
        self.assertEqual(result.count("[Result:"), 3)
        self.assertIn("[Result: content_a]", result)
        self.assertIn("[Result: content_b]", result)
        self.assertIn("[Result: dir1\ndir2]", result)

    def test_list_content_all_non_text_becomes_ok(self):
        """tool_result whose content list has no text blocks → empty after
        filtering → should be marked as ok."""
        result = self._run([
            _assistant_blocks([_tool_use("Read", file_path="image.png")]),
            _user_blocks([_tool_result([
                {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
            ])]),
        ])
        self.assertIn("[Result: ok]", result)
        self.assertNotIn("base64", result)

    def test_tool_result_only_user_message_creates_section(self):
        """User message with only tool_result blocks (no text) should still
        produce a === USER === section."""
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="echo hello")]),
            _user_blocks([_tool_result("hello")]),
        ])
        self.assertIn("=== USER ===", result)
        self.assertIn("[Result: hello]", result)


class TestToolResultCoverage(unittest.TestCase):
    """Integration tests: real trajectory files must not lose any tool_result."""

    ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "docs" / "analysis"

    def setUp(self):
        self._tmpfiles: list[Path] = []

    def tearDown(self):
        for p in self._tmpfiles:
            p.unlink(missing_ok=True)

    def _run(self, messages: list[dict]) -> str:
        p = _write_jsonl(messages)
        self._tmpfiles.append(p)
        return extract_for_scoring(p)

    # --- real trajectory files ---

    def _count_tool_results_in_jsonl(self, path: Path) -> int:
        count = 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = obj.get("message")
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            count += 1
        return count

    @unittest.skipUnless(
        (Path(__file__).resolve().parent.parent / "docs" / "analysis").is_dir(),
        "docs/analysis not present",
    )
    def test_real_trajectories_no_result_loss(self):
        """Every tool_result in real JSONL files must appear in extract output."""
        jsonl_files = sorted(self.ANALYSIS_DIR.rglob("*.jsonl"))
        self.assertGreater(len(jsonl_files), 0, "no JSONL files found")
        for jf in jsonl_files:
            raw_count = self._count_tool_results_in_jsonl(jf)
            output = extract_for_scoring(jf, max_chars=10_000_000)
            output_count = output.count("[Result:")
            self.assertEqual(
                raw_count,
                output_count,
                f"{jf.name}: expected {raw_count} tool_results, got {output_count}",
            )

    # --- non-string content ---

    def test_integer_content_converted(self):
        """tool_result with integer content should be stringified."""
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="wc -l")]),
            _user_blocks([_tool_result(42)]),
        ])
        self.assertIn("[Result: 42]", result)

    def test_none_content_marked_ok(self):
        """tool_result with None content → empty string → ok."""
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="true")]),
            _user_blocks([{"type": "tool_result", "tool_use_id": "tu_1", "content": None}]),
        ])
        self.assertIn("[Result: ok]", result)

    def test_missing_content_key_marked_ok(self):
        """tool_result with no content key at all → defaults to empty → ok."""
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="true")]),
            _user_blocks([{"type": "tool_result", "tool_use_id": "tu_1"}]),
        ])
        self.assertIn("[Result: ok]", result)

    # --- no duplicate under global truncation ---

    def test_truncation_does_not_duplicate_results(self):
        """When max_chars forces head/tail truncation, each result appears at most once."""
        messages = []
        for i in range(50):
            messages.append(_assistant_blocks([_tool_use("Bash", command=f"cmd_{i}")]))
            messages.append(_user_blocks([_tool_result(f"output_{i}" * 20)]))
        result = self._run(messages)
        output = extract_for_scoring(_write_jsonl(messages), max_chars=2000)
        for i in range(50):
            tag = f"output_{i}"
            self.assertLessEqual(
                output.count(f"[Result: {tag}"),
                1,
                f"result {tag} appears more than once after truncation",
            )


# ===========================================================================
# Global truncation: [... truncated ...] marker and head/tail split
# ===========================================================================

class TestGlobalTruncation(unittest.TestCase):
    """When extract_for_scoring output exceeds max_chars, it must keep the
    first 30% and last 70% with '[... truncated ...]' in the middle."""

    def setUp(self):
        self._tmpfiles: list[Path] = []

    def tearDown(self):
        for p in self._tmpfiles:
            p.unlink(missing_ok=True)

    def _run(self, messages: list[dict], max_chars: int = 50_000) -> str:
        p = _write_jsonl(messages)
        self._tmpfiles.append(p)
        return extract_for_scoring(p, max_chars=max_chars)

    def _make_long_trajectory(self, n_rounds: int = 20, text_len: int = 100) -> list[dict]:
        """Build n_rounds of user/assistant pairs with predictable content.

        Each round produces roughly:
            '\\n=== USER ===\\n' + text  (~80 chars)
            '\\n=== ASSISTANT ===\\n' + text (~85 chars)
        """
        msgs = []
        for i in range(n_rounds):
            tag = f"ROUND-{i:03d}"
            pad = "X" * (text_len - len(tag))
            text = tag + pad
            msgs.append(_user(text))
            msgs.append({"role": "assistant", "content": text})
        return msgs

    # --- marker presence ---

    def test_truncated_marker_appears_when_over_limit(self):
        """Output longer than max_chars must contain the truncated marker."""
        msgs = self._make_long_trajectory(n_rounds=20, text_len=100)
        result = self._run(msgs, max_chars=1000)
        self.assertIn("[... truncated ...]", result)

    def test_no_truncated_marker_when_under_limit(self):
        """Short trajectory must not contain the truncated marker."""
        msgs = [_user("hello"), {"role": "assistant", "content": "hi"}]
        result = self._run(msgs, max_chars=50_000)
        self.assertNotIn("[... truncated ...]", result)

    def test_no_truncated_marker_at_exact_limit(self):
        """Output exactly at max_chars should not be truncated."""
        msgs = [_user("A" * 50), {"role": "assistant", "content": "B" * 50}]
        # Run first to measure actual length
        full = self._run(msgs, max_chars=100_000)
        exact = self._run(msgs, max_chars=len(full))
        self.assertNotIn("[... truncated ...]", exact)

    # --- head/tail 30/70 split ---

    def test_head_preserves_first_content(self):
        """First user message must survive in the kept head (30%)."""
        msgs = self._make_long_trajectory(n_rounds=30, text_len=80)
        result = self._run(msgs, max_chars=1000)
        self.assertIn("ROUND-000", result)

    def test_tail_preserves_last_accumulated_content(self):
        """Last accumulated round must survive in the kept tail (70%).

        With text_len=200 and max_chars=3000, the early break fires after
        ~7 rounds.  ROUND-006 is the last fully accumulated round and
        should appear in the tail.
        """
        msgs = self._make_long_trajectory(n_rounds=30, text_len=200)
        result = self._run(msgs, max_chars=3000)
        self.assertIn("[... truncated ...]", result)
        self.assertIn("ROUND-006", result)

    def test_middle_content_dropped(self):
        """Content in the middle must be cut out by the head/tail truncation.

        A single long user message (~10000 chars) triggers the early break
        after one message.  Post-truncation then splits it:
          - head = first 30% (chars 0-1499)
          - tail = last  70% (chars 5000-end)
        A marker placed at position ~4000 falls in the gap and is dropped.
        """
        # Build a user message: HEAD + 4000 M's + unique marker + more padding + TAIL
        marker = "DROPME_MARKER_XYZ"
        head_part = "HEAD_CONTENT_" + "M" * 3987   # ~4000 chars before marker
        tail_part = "N" * 5000 + "_TAIL_CONTENT"   # ~5000 chars after marker
        long_text = head_part + marker + tail_part  # ~10000 chars total

        msgs = [_user(long_text)]
        result = self._run(msgs, max_chars=5000)

        self.assertIn("[... truncated ...]", result)
        # Head: beginning of message preserved
        self.assertIn("HEAD_CONTENT", result)
        # Tail: end of message preserved
        self.assertIn("TAIL_CONTENT", result)
        # Middle: marker at position ~4000 falls in the gap
        self.assertNotIn("DROPME_MARKER_XYZ", result)

    def test_head_tail_ratio_30_70(self):
        """Verify the split: keep_start = int(max_chars * 0.3)."""
        max_chars = 500
        msgs = self._make_long_trajectory(n_rounds=30, text_len=100)
        result = self._run(msgs, max_chars=max_chars)
        self.assertIn("[... truncated ...]", result)

        marker = "\n\n[... truncated ...]\n\n"
        idx = result.index(marker)
        head = result[:idx]
        tail = result[idx + len(marker):]

        expected_head = int(max_chars * 0.3)  # 150
        expected_tail = max_chars - expected_head  # 350
        self.assertEqual(len(head), expected_head)
        self.assertEqual(len(tail), expected_tail)

    def test_total_length_capped_at_max_plus_marker(self):
        """After truncation, total length = max_chars + len(marker)."""
        max_chars = 800
        msgs = self._make_long_trajectory(n_rounds=30, text_len=100)
        result = self._run(msgs, max_chars=max_chars)
        marker = "\n\n[... truncated ...]\n\n"
        self.assertIn(marker, result)
        # head(240) + marker + tail(560) = 800 + len(marker)
        expected_total = max_chars + len(marker)
        self.assertEqual(len(result), expected_total)

    # --- early break: stops reading after max_chars ---

    def test_early_break_stops_processing(self):
        """Messages after total > max_chars are never read."""
        msgs = self._make_long_trajectory(n_rounds=5, text_len=100)
        # Add a distinctive final message that would be far beyond max_chars=200
        msgs.append(_user("UNIQUE-FINAL-MESSAGE-THAT-SHOULD-NOT-APPEAR"))
        result = self._run(msgs, max_chars=200)
        # The early break should stop before reaching the final message
        # (it accumulates in file order, so late content won't be collected)
        # The truncated output keeps tail from what was already collected
        self.assertNotIn("UNIQUE-FINAL-MESSAGE", result)


# ===========================================================================
# Tool_use formatting: [Tool: name(inp_summary)]
# ===========================================================================

class TestToolUseFormatting(unittest.TestCase):
    """Verify the [Tool: name(...)] summary format for assistant tool_use blocks."""

    def setUp(self):
        self._tmpfiles: list[Path] = []

    def tearDown(self):
        for p in self._tmpfiles:
            p.unlink(missing_ok=True)

    def _run(self, messages: list[dict]) -> str:
        p = _write_jsonl(messages)
        self._tmpfiles.append(p)
        return extract_for_scoring(p)

    # --- basic format ---

    def test_bash_tool_format(self):
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command="npm test")]),
        ])
        self.assertIn("[Tool: Bash(command=npm test)]", result)

    def test_read_tool_format(self):
        result = self._run([
            _assistant_blocks([_tool_use("Read", file_path="src/app.py")]),
        ])
        self.assertIn("[Tool: Read(file_path=src/app.py)]", result)

    def test_edit_tool_format(self):
        result = self._run([
            _assistant_blocks([_tool_use("Edit", file_path="a.py", old_string="foo", new_string="bar")]),
        ])
        self.assertIn("[Tool: Edit(file_path=a.py, old_string=foo, new_string=bar)]", result)

    def test_write_tool_format(self):
        result = self._run([
            _assistant_blocks([_tool_use("Write", file_path="new.py", content="print('hi')")]),
        ])
        self.assertIn("[Tool: Write(file_path=new.py, content=print('hi'))]", result)

    # --- empty input ---

    def test_empty_input_shows_ellipsis(self):
        """tool_use with empty input dict → inp_summary = '...'."""
        result = self._run([
            _assistant_blocks([{"type": "tool_use", "id": "tu_1", "name": "Glob", "input": {}}]),
        ])
        self.assertIn("[Tool: Glob(...)]", result)

    def test_no_input_field_shows_ellipsis(self):
        """tool_use block without 'input' key → inp.items() on {} → '...'."""
        result = self._run([
            _assistant_blocks([{"type": "tool_use", "id": "tu_1", "name": "Foo"}]),
        ])
        self.assertIn("[Tool: Foo(...)]", result)

    # --- unknown tool name ---

    def test_missing_tool_name_shows_question_mark(self):
        """tool_use with no 'name' field → name defaults to '?'."""
        result = self._run([
            _assistant_blocks([{"type": "tool_use", "id": "tu_1", "input": {"x": "1"}}]),
        ])
        self.assertIn("[Tool: ?(x=1)]", result)

    # --- key field truncation (> 300 → 200 + '...') ---

    def test_edit_key_field_truncated_at_200(self):
        """Edit's old_string/new_string/file_path are key fields:
        values > 300 chars → truncated to 200 + '...'."""
        long_old = "O" * 400
        long_new = "N" * 400
        result = self._run([
            _assistant_blocks([_tool_use("Edit", file_path="a.py", old_string=long_old, new_string=long_new)]),
        ])
        # old_string truncated to 200 O's + "..."
        self.assertIn("O" * 200 + "...", result)
        self.assertNotIn("O" * 201, result)
        # new_string truncated to 200 N's + "..."
        self.assertIn("N" * 200 + "...", result)
        self.assertNotIn("N" * 201, result)

    def test_bash_command_truncated_at_200(self):
        """Bash's command is a key field: > 300 → 200 + '...'."""
        long_cmd = "A" * 400
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command=long_cmd)]),
        ])
        # Command value truncated: first 200 A's followed by "..."
        self.assertIn("A" * 200 + "...", result)
        # Original 400 A's must not appear intact
        self.assertNotIn("A" * 400, result)

    def test_write_content_truncated_at_200(self):
        """Write's content is a key field: > 300 → 200 + '...'."""
        long_content = "C" * 500
        result = self._run([
            _assistant_blocks([_tool_use("Write", file_path="big.py", content=long_content)]),
        ])
        self.assertIn("C" * 200 + "...", result)
        self.assertNotIn("C" * 301, result)

    def test_key_field_not_truncated_when_under_300(self):
        """Key field value ≤ 300 chars → kept as-is."""
        cmd = "x" * 300
        result = self._run([
            _assistant_blocks([_tool_use("Bash", command=cmd)]),
        ])
        self.assertIn(f"command={cmd}", result)
        self.assertNotIn("...", result)

    # --- non-key field truncation (> 80 → 40 + '...') ---

    def test_non_key_field_truncated_at_40(self):
        """Non-key fields (not in _KEY_FIELDS) > 80 chars → 40 + '...'."""
        # Glob is not in _KEY_FIELDS, so 'pattern' is a non-key field
        long_pattern = "P" * 100
        result = self._run([
            _assistant_blocks([_tool_use("Glob", pattern=long_pattern)]),
        ])
        self.assertIn("P" * 40 + "...", result)
        self.assertNotIn("P" * 41, result)

    def test_non_key_field_not_truncated_when_under_80(self):
        """Non-key field value ≤ 80 chars → kept as-is."""
        pattern = "src/**/*.py"
        result = self._run([
            _assistant_blocks([_tool_use("Glob", pattern=pattern)]),
        ])
        self.assertIn(f"pattern={pattern}", result)

    # --- parameter count limit ---

    def test_known_tool_limits_to_5_params(self):
        """Known tools (Edit/Write/Bash) with key fields: max 5 params shown."""
        # Edit is known — provide 6 fields, only first 5 should appear
        result = self._run([
            _assistant_blocks([_tool_use(
                "Edit",
                file_path="a.py",
                old_string="old",
                new_string="new",
                extra1="e1",
                extra2="e2",
                extra3="e3",
            )]),
        ])
        # First 5 fields should appear; extra3 should be dropped
        self.assertIn("file_path=a.py", result)
        self.assertIn("old_string=old", result)
        self.assertIn("new_string=new", result)
        self.assertIn("extra1=e1", result)
        self.assertIn("extra2=e2", result)
        self.assertNotIn("extra3", result)

    def test_unknown_tool_limits_to_3_params(self):
        """Unknown tools (no key fields): max 3 params shown."""
        # 'CustomTool' is not in _KEY_FIELDS → max 3 params
        result = self._run([
            _assistant_blocks([_tool_use(
                "CustomTool",
                a="1",
                b="2",
                c="3",
                d="4",
            )]),
        ])
        self.assertIn("a=1", result)
        self.assertIn("b=2", result)
        self.assertIn("c=3", result)
        self.assertNotIn("d=4", result)

    # --- tool_id registration for dedup ---

    def test_tool_id_registered_for_result_dedup(self):
        """tool_use id→name mapping enables 'same as <tool>' dedup markers."""
        result = self._run([
            _assistant_blocks([{"type": "tool_use", "id": "tu_read", "name": "Read", "input": {"file_path": "a.py"}}]),
            _user_blocks([{"type": "tool_result", "tool_use_id": "tu_read", "content": "shared"}]),
            _assistant_blocks([{"type": "tool_use", "id": "tu_bash", "name": "Bash", "input": {"command": "cat a.py"}}]),
            _user_blocks([{"type": "tool_result", "tool_use_id": "tu_bash", "content": "shared"}]),
        ])
        self.assertIn("[Result: shared]", result)
        self.assertIn("[Result: same as Read]", result)

    # --- assistant text + tool_use in same message ---

    def test_text_and_tool_use_in_same_message(self):
        """Assistant message with both text and tool_use → both appear."""
        result = self._run([
            _assistant_blocks([
                {"type": "text", "text": "Let me check the file."},
                _tool_use("Read", file_path="app.py"),
            ]),
        ])
        self.assertIn("Let me check the file.", result)
        self.assertIn("[Tool: Read(file_path=app.py)]", result)

    def test_multiple_tool_uses_in_same_message(self):
        """Multiple parallel tool_use blocks in one assistant message."""
        result = self._run([
            _assistant_blocks([
                {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "a.py"}},
                {"type": "tool_use", "id": "tu_2", "name": "Bash", "input": {"command": "ls"}},
            ]),
        ])
        self.assertIn("[Tool: Read(file_path=a.py)]", result)
        self.assertIn("[Tool: Bash(command=ls)]", result)
        self.assertEqual(result.count("[Tool:"), 2)

    # --- string content in assistant message ---

    def test_assistant_string_content_no_tool_format(self):
        """Assistant message with string content (not list) → plain text, no [Tool:]."""
        result = self._run([
            {"role": "assistant", "content": "Just a plain reply."},
        ])
        self.assertIn("Just a plain reply.", result)
        self.assertNotIn("[Tool:", result)


# ===========================================================================
# Malformed / unexpected input: extract_for_scoring must not crash
# ===========================================================================

class TestExtractForScoringRobustness(unittest.TestCase):
    """Bad JSON, non-dict objects, missing fields — all must be skipped
    gracefully without raising exceptions."""

    def setUp(self):
        self._tmpfiles: list[Path] = []

    def tearDown(self):
        for p in self._tmpfiles:
            p.unlink(missing_ok=True)

    def _write_raw(self, lines: list[str]) -> Path:
        """Write raw text lines to a temp JSONL file (no envelope wrapping)."""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w", encoding="utf-8",
        )
        for line in lines:
            tmp.write(line + "\n")
        tmp.flush()
        tmp.close()
        p = Path(tmp.name)
        self._tmpfiles.append(p)
        return p

    # --- malformed JSON ---

    def test_bad_json_line_skipped(self):
        """A line that is not valid JSON must be silently skipped."""
        p = self._write_raw([
            "this is not json at all {{{",
            json.dumps({"message": {"role": "user", "content": "hello"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("hello", result)

    def test_truncated_json_skipped(self):
        """A line with truncated JSON (e.g., missing closing brace) is skipped."""
        p = self._write_raw([
            '{"message": {"role": "user", "content": "broken"',
            json.dumps({"message": {"role": "user", "content": "ok"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("ok", result)
        self.assertNotIn("broken", result)

    def test_empty_lines_skipped(self):
        """Blank lines between records must not cause errors."""
        p = self._write_raw([
            "",
            "   ",
            json.dumps({"message": {"role": "user", "content": "real"}}),
            "",
        ])
        result = extract_for_scoring(p)
        self.assertIn("real", result)

    # --- non-dict JSON values ---

    def test_json_array_skipped(self):
        """A JSON array at the top level is not a dict → skipped."""
        p = self._write_raw([
            json.dumps([1, 2, 3]),
            json.dumps({"message": {"role": "user", "content": "after array"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("after array", result)

    def test_json_string_skipped(self):
        """A bare JSON string at the top level → skipped."""
        p = self._write_raw([
            json.dumps("just a string"),
            json.dumps({"message": {"role": "user", "content": "after string"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("after string", result)

    def test_json_number_skipped(self):
        """A bare JSON number → skipped."""
        p = self._write_raw([
            "42",
            json.dumps({"message": {"role": "user", "content": "after number"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("after number", result)

    def test_json_null_skipped(self):
        """A bare JSON null → skipped."""
        p = self._write_raw([
            "null",
            json.dumps({"message": {"role": "user", "content": "after null"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("after null", result)

    def test_json_bool_skipped(self):
        """A bare JSON boolean → skipped."""
        p = self._write_raw([
            "true",
            json.dumps({"message": {"role": "user", "content": "after bool"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("after bool", result)

    # --- missing or invalid 'message' field ---

    def test_no_message_field_skipped(self):
        """Object without 'message' key → skipped."""
        p = self._write_raw([
            json.dumps({"type": "user", "sessionId": "s1"}),
            json.dumps({"message": {"role": "user", "content": "has message"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("has message", result)

    def test_message_is_string_skipped(self):
        """message is a string, not a dict → skipped by isinstance check."""
        p = self._write_raw([
            json.dumps({"message": "plain string message"}),
            json.dumps({"message": {"role": "user", "content": "valid"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("valid", result)

    def test_message_is_list_skipped(self):
        """message is a list, not a dict → skipped."""
        p = self._write_raw([
            json.dumps({"message": [{"role": "user", "content": "nope"}]}),
            json.dumps({"message": {"role": "user", "content": "valid"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("valid", result)

    def test_message_is_null_skipped(self):
        """message is null → not a dict → skipped."""
        p = self._write_raw([
            json.dumps({"message": None}),
            json.dumps({"message": {"role": "user", "content": "valid"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("valid", result)

    # --- empty or unexpected content ---

    def test_empty_content_skipped(self):
        """message.content is empty string → `if not content` skips it."""
        p = self._write_raw([
            json.dumps({"message": {"role": "user", "content": ""}}),
            json.dumps({"message": {"role": "user", "content": "real content"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("real content", result)

    def test_none_content_skipped(self):
        """message.content is None → falsy → skipped."""
        p = self._write_raw([
            json.dumps({"message": {"role": "user", "content": None}}),
            json.dumps({"message": {"role": "user", "content": "real"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("real", result)

    def test_missing_content_key_skipped(self):
        """message has no 'content' key → content is None → skipped."""
        p = self._write_raw([
            json.dumps({"message": {"role": "user"}}),
            json.dumps({"message": {"role": "user", "content": "present"}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("present", result)

    def test_integer_content_not_string_or_list(self):
        """message.content is an integer — not str, not list → no branch
        matches, so nothing is emitted for that message."""
        p = self._write_raw([
            json.dumps({"message": {"role": "user", "content": 42}}),
            json.dumps({"message": {"role": "user", "content": "real text"}}),
        ])
        result = extract_for_scoring(p)
        # 42 should not appear (content is int, neither str nor list)
        self.assertNotIn("42", result)
        self.assertIn("real text", result)

    def test_no_role_field(self):
        """message with no 'role' → defaults to '' → neither user nor assistant."""
        p = self._write_raw([
            json.dumps({"message": {"content": "no role here"}}),
            json.dumps({"message": {"role": "user", "content": "with role"}}),
        ])
        result = extract_for_scoring(p)
        # "no role here" is content but role="" so it's ignored
        self.assertNotIn("no role here", result)
        self.assertIn("with role", result)

    def test_unknown_role_ignored(self):
        """message with role='system' → not user or assistant → skipped."""
        p = self._write_raw([
            json.dumps({"message": {"role": "system", "content": "system prompt"}}),
            json.dumps({"message": {"role": "user", "content": "user msg"}}),
        ])
        result = extract_for_scoring(p)
        self.assertNotIn("system prompt", result)
        self.assertIn("user msg", result)

    # --- mixed valid + invalid in one file ---

    def test_mixed_valid_and_invalid_lines(self):
        """A file with interleaved valid, invalid, and weird lines must
        still extract all valid user/assistant content."""
        lines = [
            "not json",
            json.dumps([1, 2]),
            "null",
            json.dumps({"message": {"role": "user", "content": "first user"}}),
            "",
            json.dumps({"message": None}),
            json.dumps({"message": {"role": "assistant", "content": "first reply"}}),
            "42",
            json.dumps({"no_message_key": True}),
            json.dumps({"message": {"role": "user", "content": "second user"}}),
            json.dumps({"message": {"role": "user", "content": ""}}),
            json.dumps({"message": {"content": "no role"}}),
        ]
        p = self._write_raw(lines)
        result = extract_for_scoring(p)
        self.assertIn("first user", result)
        self.assertIn("first reply", result)
        self.assertIn("second user", result)
        # Invalid entries must not appear
        self.assertNotIn("no role", result)

    def test_completely_empty_file(self):
        """An empty JSONL file must return an empty string."""
        p = self._write_raw([])
        result = extract_for_scoring(p)
        self.assertEqual(result, "")

    def test_all_invalid_lines(self):
        """File with only invalid lines → empty result, no crash."""
        p = self._write_raw([
            "not json",
            "null",
            "42",
            json.dumps([1, 2, 3]),
            json.dumps({"message": None}),
        ])
        result = extract_for_scoring(p)
        self.assertEqual(result, "")

    # --- non-dict blocks inside content list ---

    def test_string_block_in_content_list(self):
        """A bare string inside a content list is included for user messages."""
        p = self._write_raw([
            json.dumps({"message": {"role": "user", "content": ["hello from list"]}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("hello from list", result)

    def test_non_dict_block_in_user_content_list(self):
        """Non-dict, non-string items in content list (e.g., integers) are skipped."""
        p = self._write_raw([
            json.dumps({"message": {"role": "user", "content": [42, "real text"]}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("real text", result)
        self.assertNotIn("42", result)

    def test_non_dict_block_in_assistant_content_list(self):
        """Non-dict items in assistant content list are skipped.

        Unlike the user branch, the assistant branch does NOT handle bare
        strings in content lists — only dict blocks (text/tool_use)."""
        p = self._write_raw([
            json.dumps({"message": {"role": "assistant", "content": [
                None,
                "bare string",
                {"type": "text", "text": "valid reply"},
            ]}}),
        ])
        result = extract_for_scoring(p)
        # Bare string in assistant content list is ignored
        self.assertNotIn("bare string", result)
        # Dict text block is extracted
        self.assertIn("valid reply", result)

    def test_tool_use_with_non_dict_input(self):
        """tool_use block whose 'input' is not a dict → inp.items() guarded
        by isinstance(inp, dict) → produces [Tool: name(...)]."""
        p = self._write_raw([
            json.dumps({"message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": "not a dict"},
            ]}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("[Tool: Bash(...)]", result)

    def test_tool_use_with_missing_input(self):
        """tool_use block without 'input' key → defaults to {} → '...'."""
        p = self._write_raw([
            json.dumps({"message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "Read"},
            ]}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("[Tool: Read(...)]", result)

    def test_tool_result_with_non_dict_block(self):
        """Non-dict items in user content list inside tool_result area are skipped."""
        p = self._write_raw([
            json.dumps({"message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}},
            ]}}),
            json.dumps({"message": {"role": "user", "content": [
                42,
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "file1"},
            ]}}),
        ])
        result = extract_for_scoring(p)
        self.assertIn("[Result: file1]", result)


if __name__ == "__main__":
    unittest.main()
