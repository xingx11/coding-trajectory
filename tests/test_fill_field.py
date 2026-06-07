"""Focused tests for _fill_field regex backfill and section isolation.

Verifies that the tempered-greedy-token regex in _fill_field correctly
confines replacements to the target section, preventing Qwen session_id
from bleeding into the Claude section (and vice versa).

Also covers the passrate re.sub patterns used in _update_metadata_files,
which rely on field-name disambiguation rather than section boundaries.
"""

from __future__ import annotations

import re
import unittest

from ctpipe.finalize import _fill_field


# =====================================================================
# Minimal template (same as existing tests)
# =====================================================================

MINIMAL_TEMPLATE = """\
# Task CT-0001

## Qwen Conversation
- Session id:
- Round count:

## Claude Conversation
- Session id:
- Round count:

- Qwen passrate:
- Claude passrate:
"""


# =====================================================================
# Realistic template (from docs/metadata_template.md)
# =====================================================================

REALISTIC_TEMPLATE = """\
# CT-0001 Metadata

## Codebase

- Project path: /demo
- Source: local project

## Task Label

- Task type: bug-fix
- Application domain: web_frontend
- Language: ts

## Qwen Conversation

- Session id:
- Trajectory file:
- Round count:
- Prompt strategy: same-theme
- Initial prompt:

```text

```

- Follow-up summary:

```text

```

## Claude Conversation

- Session id:
- Trajectory file:
- Round count:
- Prompt strategy: same-theme
- Initial prompt:

```text

```

- Follow-up summary:

```text

```

## Scoring

- Qwen score file:
- Claude score file:
- Qwen passrate:
- Claude passrate:

## Notes

- Are Qwen and Claude prompts identical? yes / no
"""


# =====================================================================
# Direct _fill_field unit tests — section isolation
# =====================================================================


class FillFieldSectionIsolationTest(unittest.TestCase):
    """_fill_field must never write outside the target section."""

    def test_qwen_session_id_does_not_appear_in_claude_section(self) -> None:
        result = _fill_field(MINIMAL_TEMPLATE, "Qwen", "Session id", "QW-ABC")
        lines = result.splitlines()
        in_claude = False
        for line in lines:
            if line.startswith("## Claude"):
                in_claude = True
            elif line.startswith("## "):
                in_claude = False
            if in_claude and "Session id:" in line:
                self.assertNotIn("QW-ABC", line,
                    "Qwen session_id leaked into Claude section")

    def test_claude_session_id_does_not_appear_in_qwen_section(self) -> None:
        result = _fill_field(MINIMAL_TEMPLATE, "Claude", "Session id", "CL-XYZ")
        lines = result.splitlines()
        in_qwen = False
        for line in lines:
            if line.startswith("## Qwen"):
                in_qwen = True
            elif line.startswith("## "):
                in_qwen = False
            if in_qwen and "Session id:" in line:
                self.assertNotIn("CL-XYZ", line,
                    "Claude session_id leaked into Qwen section")

    def test_qwen_fill_leaves_claude_session_id_empty(self) -> None:
        result = _fill_field(MINIMAL_TEMPLATE, "Qwen", "Session id", "QW-ONLY")
        self.assertIn("- Session id: QW-ONLY", result)
        # Claude's Session id line must remain empty (just "- Session id:")
        claude_section_start = result.index("## Claude Conversation")
        claude_section = result[claude_section_start:]
        self.assertIn("- Session id:\n", claude_section,
            "Claude's Session id should still be empty after Qwen fill")

    def test_claude_fill_leaves_qwen_session_id_empty(self) -> None:
        result = _fill_field(MINIMAL_TEMPLATE, "Claude", "Session id", "CL-ONLY")
        self.assertIn("- Session id: CL-ONLY", result)
        qwen_section_start = result.index("## Qwen Conversation")
        qwen_section_end = result.index("## Claude Conversation")
        qwen_section = result[qwen_section_start:qwen_section_end]
        self.assertIn("- Session id:\n", qwen_section,
            "Qwen's Session id should still be empty after Claude fill")

    def test_both_sections_filled_independently(self) -> None:
        result = _fill_field(MINIMAL_TEMPLATE, "Qwen", "Session id", "QW-111")
        result = _fill_field(result, "Claude", "Session id", "CL-222")
        self.assertIn("- Session id: QW-111", result)
        self.assertIn("- Session id: CL-222", result)
        self.assertEqual(result.count("QW-111"), 1, "Qwen id should appear exactly once")
        self.assertEqual(result.count("CL-222"), 1, "Claude id should appear exactly once")

    def test_qwen_round_count_does_not_touch_claude_round_count(self) -> None:
        result = _fill_field(MINIMAL_TEMPLATE, "Qwen", "Round count", "5")
        lines = result.splitlines()
        in_claude = False
        for line in lines:
            if line.startswith("## Claude"):
                in_claude = True
            elif line.startswith("## "):
                in_claude = False
            if in_claude and "Round count:" in line:
                self.assertEqual(line.strip(), "- Round count:",
                    "Claude's Round count was modified by Qwen fill")

    def test_claude_round_count_does_not_touch_qwen_round_count(self) -> None:
        result = _fill_field(MINIMAL_TEMPLATE, "Claude", "Round count", "8")
        lines = result.splitlines()
        in_qwen = False
        for line in lines:
            if line.startswith("## Qwen"):
                in_qwen = True
            elif line.startswith("## "):
                in_qwen = False
            if in_qwen and "Round count:" in line:
                self.assertEqual(line.strip(), "- Round count:",
                    "Qwen's Round count was modified by Claude fill")


# =====================================================================
# Realistic template — extra content between heading and fields
# =====================================================================


class FillFieldRealisticTemplateTest(unittest.TestCase):
    """Test _fill_field against the full metadata_template.md format,
    which has blank lines, code fences, and many fields between headings."""

    def test_qwen_session_id_with_extra_content_between_heading_and_field(self) -> None:
        result = _fill_field(REALISTIC_TEMPLATE, "Qwen", "Session id", "QW-REAL")
        self.assertIn("- Session id: QW-REAL", result)
        # Claude's session id must remain empty
        claude_start = result.index("## Claude Conversation")
        claude_end = result.index("## Scoring")
        claude_section = result[claude_start:claude_end]
        self.assertIn("- Session id:\n", claude_section)

    def test_claude_session_id_with_extra_content_between_heading_and_field(self) -> None:
        result = _fill_field(REALISTIC_TEMPLATE, "Claude", "Session id", "CL-REAL")
        self.assertIn("- Session id: CL-REAL", result)
        qwen_start = result.index("## Qwen Conversation")
        qwen_end = result.index("## Claude Conversation")
        qwen_section = result[qwen_start:qwen_end]
        self.assertIn("- Session id:\n", qwen_section)

    def test_qwen_round_count_with_code_fences_in_section(self) -> None:
        """The Qwen section contains code fences; the regex must not
        be confused by them when locating the Round count field."""
        result = _fill_field(REALISTIC_TEMPLATE, "Qwen", "Round count", "12")
        self.assertIn("- Round count: 12", result)
        # Claude's round count must remain empty
        claude_start = result.index("## Claude Conversation")
        claude_end = result.index("## Scoring")
        claude_section = result[claude_start:claude_end]
        self.assertIn("- Round count:\n", claude_section)

    def test_both_sessions_and_round_counts_on_realistic_template(self) -> None:
        result = _fill_field(REALISTIC_TEMPLATE, "Qwen", "Session id", "QW-S1")
        result = _fill_field(result, "Qwen", "Round count", "5")
        result = _fill_field(result, "Claude", "Session id", "CL-S2")
        result = _fill_field(result, "Claude", "Round count", "8")

        self.assertIn("- Session id: QW-S1", result)
        self.assertIn("- Round count: 5", result)
        self.assertIn("- Session id: CL-S2", result)
        self.assertIn("- Round count: 8", result)

        # Each value appears exactly once
        self.assertEqual(result.count("QW-S1"), 1)
        self.assertEqual(result.count("CL-S2"), 1)


# =====================================================================
# Pre-filled fields — idempotency
# =====================================================================


class FillFieldIdempotencyTest(unittest.TestCase):
    """_fill_field must not overwrite fields that already have values."""

    def test_already_filled_qwen_session_id_not_overwritten(self) -> None:
        prefilled = MINIMAL_TEMPLATE.replace(
            "## Qwen Conversation\n- Session id:",
            "## Qwen Conversation\n- Session id: ORIGINAL-QW",
        )
        result = _fill_field(prefilled, "Qwen", "Session id", "NEW-QW")
        self.assertIn("ORIGINAL-QW", result)
        self.assertNotIn("NEW-QW", result)

    def test_already_filled_claude_session_id_not_overwritten(self) -> None:
        prefilled = MINIMAL_TEMPLATE.replace(
            "## Claude Conversation\n- Session id:",
            "## Claude Conversation\n- Session id: ORIGINAL-CL",
        )
        result = _fill_field(prefilled, "Claude", "Session id", "NEW-CL")
        self.assertIn("ORIGINAL-CL", result)
        self.assertNotIn("NEW-CL", result)

    def test_qwen_prefilled_but_claude_empty_fills_only_claude(self) -> None:
        prefilled = MINIMAL_TEMPLATE.replace(
            "## Qwen Conversation\n- Session id:",
            "## Qwen Conversation\n- Session id: EXISTING",
        )
        result = _fill_field(prefilled, "Qwen", "Session id", "ATTEMPT")
        # Qwen: still has original
        self.assertIn("EXISTING", result)
        self.assertNotIn("ATTEMPT", result)

        # Now fill Claude's empty session id
        result = _fill_field(result, "Claude", "Session id", "CL-NEW")
        self.assertIn("- Session id: CL-NEW", result)
        self.assertIn("EXISTING", result)

    def test_round_count_prefilled_not_overwritten(self) -> None:
        prefilled = MINIMAL_TEMPLATE.replace(
            "- Round count:\n\n## Claude",
            "- Round count: 99\n\n## Claude",
        )
        result = _fill_field(prefilled, "Qwen", "Round count", "5")
        self.assertIn("- Round count: 99", result)
        # Claude's round count is still empty
        claude_start = result.index("## Claude Conversation")
        claude_section = result[claude_start:]
        self.assertIn("- Round count:\n", claude_section)


# =====================================================================
# Edge cases — missing sections, missing fields, weird values
# =====================================================================


class FillFieldEdgeCasesTest(unittest.TestCase):

    def test_missing_section_returns_content_unchanged(self) -> None:
        content = "# Task\n\n## Other Section\n- Session id:\n"
        result = _fill_field(content, "Qwen", "Session id", "QW-1")
        self.assertEqual(result, content,
            "Content should be unchanged when target section is missing")

    def test_missing_field_in_section_returns_content_unchanged(self) -> None:
        content = "## Qwen Conversation\n- Something else:\n"
        result = _fill_field(content, "Qwen", "Session id", "QW-1")
        self.assertEqual(result, content,
            "Content should be unchanged when target field is missing")

    def test_empty_content_returns_empty(self) -> None:
        result = _fill_field("", "Qwen", "Session id", "QW-1")
        self.assertEqual(result, "")

    def test_value_with_special_characters(self) -> None:
        result = _fill_field(MINIMAL_TEMPLATE, "Qwen", "Session id", "sess-abc_123!@#")
        self.assertIn("- Session id: sess-abc_123!@#", result)

    def test_value_with_unicode(self) -> None:
        result = _fill_field(MINIMAL_TEMPLATE, "Qwen", "Session id", "会话-ABC-123")
        self.assertIn("- Session id: 会话-ABC-123", result)

    def test_value_containing_hash_symbols(self) -> None:
        """Value with ## should not confuse the section boundary check."""
        result = _fill_field(MINIMAL_TEMPLATE, "Qwen", "Session id", "id##with##hashes")
        self.assertIn("- Session id: id##with##hashes", result)
        # Claude section must remain unaffected
        claude_start = result.index("## Claude Conversation")
        claude_section = result[claude_start:]
        self.assertIn("- Session id:\n", claude_section)

    def test_value_containing_colon(self) -> None:
        result = _fill_field(MINIMAL_TEMPLATE, "Qwen", "Session id", "urn:session:abc")
        self.assertIn("- Session id: urn:session:abc", result)

    def test_count_is_one_only_first_match_replaced(self) -> None:
        """If somehow the field appears twice in a section, only the first
        empty occurrence should be filled (count=1 in re.sub)."""
        content = (
            "## Qwen Conversation\n"
            "- Session id:\n"
            "- Session id:\n"
            "\n"
            "## Claude Conversation\n"
            "- Session id:\n"
        )
        result = _fill_field(content, "Qwen", "Session id", "FIRST-ONLY")
        self.assertEqual(result.count("FIRST-ONLY"), 1)
        # The second Qwen Session id line should still be empty
        lines = result.splitlines()
        qwen_session_lines = [
            i for i, line in enumerate(lines)
            if "Session id:" in line and i < lines.index("## Claude Conversation")
        ]
        self.assertEqual(len(qwen_session_lines), 2)
        # First line has the value
        self.assertIn("FIRST-ONLY", lines[qwen_session_lines[0]])
        # Second line is still empty
        self.assertEqual(lines[qwen_session_lines[1]].strip(), "- Session id:")

    def test_reversed_section_order(self) -> None:
        """Claude section before Qwen section — isolation must still hold."""
        content = (
            "# Task\n\n"
            "## Claude Conversation\n"
            "- Session id:\n"
            "- Round count:\n"
            "\n"
            "## Qwen Conversation\n"
            "- Session id:\n"
            "- Round count:\n"
        )
        result = _fill_field(content, "Qwen", "Session id", "QW-REV")
        self.assertIn("- Session id: QW-REV", result)
        # Claude section must remain empty
        claude_start = content.index("## Claude Conversation")
        qwen_start = result.index("## Qwen Conversation")
        claude_section = result[claude_start:qwen_start]
        self.assertIn("- Session id:\n", claude_section)

    def test_three_sections_only_target_filled(self) -> None:
        """Three sections with same field name — only target section is filled."""
        content = (
            "## Qwen Conversation\n"
            "- Session id:\n"
            "\n"
            "## Claude Conversation\n"
            "- Session id:\n"
            "\n"
            "## Other Conversation\n"
            "- Session id:\n"
        )
        result = _fill_field(content, "Qwen", "Session id", "QW-MID")
        self.assertEqual(result.count("QW-MID"), 1)
        # Other sections unchanged
        lines = result.splitlines()
        empty_session_lines = [l for l in lines if l.strip() == "- Session id:"]
        self.assertEqual(len(empty_session_lines), 2,
            "Two sections should still have empty Session id")


# =====================================================================
# Passrate regex isolation (from _update_metadata_files lines 235-241)
# =====================================================================


class PassrateRegexIsolationTest(unittest.TestCase):
    """The passrate re.sub patterns use field-name disambiguation.
    Verify they don't cross-contaminate."""

    QWEN_PASSRATE_RE = re.compile(r"(- Qwen passrate:)\s*$", re.MULTILINE)
    CLAUDE_PASSRATE_RE = re.compile(r"(- Claude passrate:)\s*$", re.MULTILINE)

    def test_qwen_passrate_regex_does_not_touch_claude_passrate_line(self) -> None:
        result = self.QWEN_PASSRATE_RE.sub(
            r"\1 0.5000", MINIMAL_TEMPLATE,
        )
        self.assertIn("- Qwen passrate: 0.5000", result)
        # Claude passrate line must still be empty
        self.assertIn("- Claude passrate:\n", result)

    def test_claude_passrate_regex_does_not_touch_qwen_passrate_line(self) -> None:
        result = self.CLAUDE_PASSRATE_RE.sub(
            r"\1 0.9000", MINIMAL_TEMPLATE,
        )
        self.assertIn("- Claude passrate: 0.9000", result)
        # Qwen passrate line must still be empty
        self.assertIn("- Qwen passrate:\n", result)

    def test_both_passrates_filled_independently(self) -> None:
        result = self.QWEN_PASSRATE_RE.sub(r"\1 0.5000", MINIMAL_TEMPLATE)
        result = self.CLAUDE_PASSRATE_RE.sub(r"\1 0.9000", result)
        self.assertIn("- Qwen passrate: 0.5000", result)
        self.assertIn("- Claude passrate: 0.9000", result)

    def test_qwen_passrate_regex_does_not_match_prefilled(self) -> None:
        """If Qwen passrate already has a value, regex should not match."""
        prefilled = MINIMAL_TEMPLATE.replace(
            "- Qwen passrate:", "- Qwen passrate: 0.1234",
        )
        result = self.QWEN_PASSRATE_RE.sub(r"\1 0.9999", prefilled)
        self.assertIn("0.1234", result)
        self.assertNotIn("0.9999", result)

    def test_passrate_regex_on_realistic_template(self) -> None:
        result = self.QWEN_PASSRATE_RE.sub(r"\1 0.5500", REALISTIC_TEMPLATE)
        result = self.CLAUDE_PASSRATE_RE.sub(r"\1 0.8800", result)
        self.assertIn("- Qwen passrate: 0.5500", result)
        self.assertIn("- Claude passrate: 0.8800", result)


# =====================================================================
# Regex pattern properties — verify the tempered greedy token works
# =====================================================================


class FillFieldRegexPropertiesTest(unittest.TestCase):
    """Verify specific properties of the _fill_field regex pattern."""

    def _get_pattern(self, section: str, field: str) -> str:
        """Reconstruct the pattern used by _fill_field for inspection."""
        return rf"(## {section} Conversation(?:(?!^## ).)*?- {field}:)\s*$"

    def test_pattern_uses_tempered_greedy_token(self) -> None:
        """The pattern must contain the negative lookahead (?!^## ) to
        prevent matching across section boundaries."""
        pattern = self._get_pattern("Qwen", "Session id")
        self.assertIn("(?!^## )", pattern,
            "Pattern must use tempered greedy token for section isolation")

    def test_pattern_anchors_to_section_heading(self) -> None:
        pattern = self._get_pattern("Qwen", "Session id")
        self.assertIn("## Qwen Conversation", pattern,
            "Pattern must anchor to the specific section heading")

    def test_pattern_requires_empty_field(self) -> None:
        """The \\s*$ at the end ensures only empty fields are matched."""
        pattern = self._get_pattern("Qwen", "Session id")
        self.assertTrue(pattern.endswith(r"\s*$"),
            "Pattern must end with \\s*$ to match only empty fields")

    def test_regex_does_not_match_across_section_boundary(self) -> None:
        """Direct regex test: the pattern must not match when the field
        is in a different section than the target."""
        pattern = self._get_pattern("Qwen", "Session id")
        content = (
            "## Claude Conversation\n"
            "- Session id:\n"
        )
        match = re.search(pattern, content, flags=re.MULTILINE | re.DOTALL)
        self.assertIsNone(match,
            "Pattern for Qwen should not match Session id in Claude-only content")

    def test_regex_matches_within_correct_section(self) -> None:
        pattern = self._get_pattern("Qwen", "Session id")
        content = (
            "## Qwen Conversation\n"
            "- Session id:\n"
            "\n"
            "## Claude Conversation\n"
            "- Session id:\n"
        )
        match = re.search(pattern, content, flags=re.MULTILINE | re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("## Qwen Conversation", match.group(0))
        self.assertNotIn("## Claude", match.group(0))

    def test_regex_captures_field_label_for_replacement(self) -> None:
        """Group 1 should capture up to and including the field label,
        so the replacement can append the value."""
        pattern = self._get_pattern("Qwen", "Session id")
        content = "## Qwen Conversation\n- Session id:\n"
        match = re.search(pattern, content, flags=re.MULTILINE | re.DOTALL)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "## Qwen Conversation\n- Session id:")


if __name__ == "__main__":
    unittest.main()
