"""Unit tests for rescore.read_task_context metadata parsing."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace

from ctpipe.rescore import build_task_context, read_task_context


def _write_tmp(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8")
    f.write(content)
    f.close()
    return Path(f.name)


def _write_tmp_in(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


NEW_FORMAT_FULL = """\
# CT-TEST Metadata

## Codebase

- Project path: D:\\projects\\myapp
- Source: local project / open-source project

## Project Summary

```text
A Flask + React web app for task management.
```

## Task Label

- Task type: bug-fix
- Application domain: web_dev
- Language: python

## Task Description

- Title: Fix login timeout bug
- Description: Users get logged out after 5 minutes instead of 30
- Acceptance criteria:
  - Session timeout should be 30 minutes
  - Existing sessions should not be affected

## Qwen Conversation

- Session id: abc-123
- Initial prompt:

```text
Fix the session timeout issue in the auth module
```

- Follow-up summary:

```text
- Check the JWT expiry configuration
```

## Claude Conversation

- Session id: def-456
- Initial prompt:

```text
The login session expires too quickly, fix it
```

- Follow-up summary:

```text
- Verify the cookie max-age setting
```
"""

OLD_FORMAT = """\
# CT-TEST Metadata

## Codebase

- Project path: D:\\projects\\oldapp
- Source: local project / open-source project

## Task Label

- Task type: feature-add
- Application domain: devtools
- Language: javascript

## Qwen Conversation

- Session id: old-111
- Initial prompt:

```text
Add dark mode support to the settings page
```

- Follow-up summary:

```text
- Make sure the toggle persists across page reloads
- Test with both light and dark system themes
```

## Claude Conversation

- Session id: old-222
- Initial prompt:

```text
Implement a dark mode toggle in user settings
```

- Follow-up summary:

```text
- Use CSS variables for theme colors
```
"""


class TestReadTaskContextNewFormat(unittest.TestCase):

    def setUp(self):
        self.path = _write_tmp(NEW_FORMAT_FULL)
        self.ctx = read_task_context(self.path)

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_project_path(self):
        self.assertEqual(self.ctx["project_path"], "D:\\projects\\myapp")

    def test_project_name(self):
        self.assertEqual(self.ctx["project_name"], "myapp")

    def test_task_label_fields(self):
        self.assertEqual(self.ctx["task_type"], "bug-fix")
        self.assertEqual(self.ctx["domain"], "web_dev")
        self.assertEqual(self.ctx["language"], "python")

    def test_project_summary(self):
        self.assertIn("Flask", self.ctx["project_summary"])

    def test_task_title(self):
        self.assertEqual(self.ctx["task_title"], "Fix login timeout bug")

    def test_task_description_from_structured_field(self):
        self.assertEqual(self.ctx["task_description"],
                         "Users get logged out after 5 minutes instead of 30")

    def test_acceptance_criteria(self):
        self.assertIn("Session timeout should be 30 minutes", self.ctx["acceptance_criteria"])
        self.assertIn("Existing sessions should not be affected", self.ctx["acceptance_criteria"])

    def test_prompts_collected(self):
        self.assertIn("Fix the session timeout issue", self.ctx["prompts"])
        self.assertIn("login session expires", self.ctx["prompts"])

    def test_followups_collected(self):
        self.assertIn("JWT expiry", self.ctx["followups"])
        self.assertIn("cookie max-age", self.ctx["followups"])

    def test_fallback_not_triggered(self):
        self.assertNotEqual(self.ctx["task_description"],
                            "Fix the session timeout issue in the auth module")

    def test_trailing_whitespace_after_colon(self):
        """Regex must tolerate trailing spaces after 'Initial prompt:'."""
        md = NEW_FORMAT_FULL.replace("- Initial prompt:\n", "- Initial prompt:   \n")
        p = _write_tmp(md)
        try:
            ctx = read_task_context(p)
            self.assertIn("Fix the session timeout issue", ctx["prompts"])
            self.assertEqual(ctx["task_description"],
                             "Users get logged out after 5 minutes instead of 30")
        finally:
            p.unlink(missing_ok=True)

    def test_missing_trailing_newline(self):
        """Code block at EOF without trailing newline must still match."""
        md = (
            "# CT-TEST Metadata\n\n"
            "## Qwen Conversation\n\n"
            "- Initial prompt:\n\n"
            "```text\n"
            "New format prompt text\n"
            "```\n\n"
            "- Follow-up summary:\n\n"
            "```text\n"
            "New followup\n"
            "```"
        )
        p = _write_tmp(md)
        try:
            ctx = read_task_context(p)
            self.assertIn("New format prompt text", ctx["prompts"])
            self.assertIn("New followup", ctx["followups"])
        finally:
            p.unlink(missing_ok=True)


class TestReadTaskContextOldFormat(unittest.TestCase):

    def setUp(self):
        self.path = _write_tmp(OLD_FORMAT)
        self.ctx = read_task_context(self.path)

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_basic_fields(self):
        self.assertEqual(self.ctx["task_type"], "feature-add")
        self.assertEqual(self.ctx["domain"], "devtools")
        self.assertEqual(self.ctx["language"], "javascript")

    def test_no_project_summary(self):
        self.assertNotIn("project_summary", self.ctx)

    def test_no_task_title(self):
        self.assertNotIn("task_title", self.ctx)

    def test_fallback_task_description(self):
        self.assertEqual(self.ctx["task_description"],
                         "Add dark mode support to the settings page")

    def test_prompts_both_models(self):
        self.assertIn("Add dark mode", self.ctx["prompts"])
        self.assertIn("dark mode toggle", self.ctx["prompts"])

    def test_followups(self):
        self.assertIn("toggle persists", self.ctx["followups"])

    def test_trailing_whitespace_after_colon(self):
        """Regex must tolerate trailing spaces in old-format metadata."""
        md = OLD_FORMAT.replace("- Initial prompt:\n", "- Initial prompt:  \n")
        p = _write_tmp(md)
        try:
            ctx = read_task_context(p)
            self.assertIn("Add dark mode", ctx["prompts"])
            self.assertEqual(ctx["task_description"],
                             "Add dark mode support to the settings page")
        finally:
            p.unlink(missing_ok=True)

    def test_missing_trailing_newline(self):
        """Code block at EOF without trailing newline must still match."""
        md = (
            "# CT-TEST Metadata\n\n"
            "## Task Label\n\n"
            "- Task type: feature\n"
            "- Language: python\n\n"
            "## Qwen Conversation\n\n"
            "- Initial prompt:\n\n"
            "```text\n"
            "Old format prompt\n"
            "```\n\n"
            "- Follow-up summary:\n\n"
            "```text\n"
            "Old followup\n"
            "```"
        )
        p = _write_tmp(md)
        try:
            ctx = read_task_context(p)
            self.assertIn("Old format prompt", ctx["prompts"])
            self.assertIn("Old followup", ctx["followups"])
            self.assertEqual(ctx["task_description"], "Old format prompt")
        finally:
            p.unlink(missing_ok=True)


class TestReadTaskContextEdgeCases(unittest.TestCase):

    def test_nonexistent_file(self):
        ctx = read_task_context(Path("/nonexistent/path/to/file.md"))
        self.assertEqual(ctx, {})

    def test_empty_file(self):
        p = _write_tmp("")
        try:
            ctx = read_task_context(p)
            self.assertEqual(ctx, {})
        finally:
            p.unlink(missing_ok=True)

    def test_only_codebase_and_label(self):
        md = """\
# CT Metadata

## Codebase

- Project path: D:\\proj\\x

## Task Label

- Task type: refactor
- Application domain: backend
- Language: go
"""
        p = _write_tmp(md)
        try:
            ctx = read_task_context(p)
            self.assertEqual(ctx["task_type"], "refactor")
            self.assertEqual(ctx["language"], "go")
            self.assertNotIn("task_description", ctx)
            self.assertNotIn("prompts", ctx)
        finally:
            p.unlink(missing_ok=True)

    def test_title_without_description_triggers_fallback(self):
        md = """\
# CT Metadata

## Task Label

- Task type: bug-fix
- Language: rust

## Task Description

- Title: Fix memory leak

## Qwen Conversation

- Initial prompt:

```text
There is a memory leak in the parser module
```
"""
        p = _write_tmp(md)
        try:
            ctx = read_task_context(p)
            self.assertEqual(ctx["task_title"], "Fix memory leak")
            self.assertEqual(ctx["task_description"],
                             "There is a memory leak in the parser module")
        finally:
            p.unlink(missing_ok=True)

    def test_description_without_title_no_fallback(self):
        md = """\
# CT Metadata

## Task Description

- Description: Refactor the auth middleware

## Qwen Conversation

- Initial prompt:

```text
Some unrelated prompt text
```
"""
        p = _write_tmp(md)
        try:
            ctx = read_task_context(p)
            self.assertEqual(ctx["task_description"], "Refactor the auth middleware")
            self.assertNotIn("task_title", ctx)
        finally:
            p.unlink(missing_ok=True)

    def test_multiple_initial_prompts_fallback_uses_first(self):
        md = """\
# CT Metadata

## Task Label

- Task type: feature-add
- Language: python

## Qwen Conversation

- Initial prompt:

```text
First prompt content here
```

## Claude Conversation

- Initial prompt:

```text
Second prompt content here
```
"""
        p = _write_tmp(md)
        try:
            ctx = read_task_context(p)
            self.assertEqual(ctx["task_description"], "First prompt content here")
            self.assertIn("Second prompt", ctx["prompts"])
        finally:
            p.unlink(missing_ok=True)

    def test_single_prompt_no_followups(self):
        md = """\
# CT Metadata

## Qwen Conversation

- Initial prompt:

```text
Only one prompt here
```
"""
        p = _write_tmp(md)
        try:
            ctx = read_task_context(p)
            self.assertEqual(ctx["task_description"], "Only one prompt here")
            self.assertNotIn("followups", ctx)
        finally:
            p.unlink(missing_ok=True)

    def test_acceptance_criteria_multiline(self):
        md = """\
# CT Metadata

## Task Description

- Title: Multi-criteria task
- Description: A task with many criteria
- Acceptance criteria:
  - First criterion
  - Second criterion
  - Third criterion
"""
        p = _write_tmp(md)
        try:
            ctx = read_task_context(p)
            self.assertEqual(ctx["acceptance_criteria"],
                             "First criterion | Second criterion | Third criterion")
        finally:
            p.unlink(missing_ok=True)


class TestBuildTaskContext(unittest.TestCase):
    """Tests for build_task_context: metadata + TaskConfig fallback chain."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.metadata_dir = self.tmpdir / "metadata"
        self.metadata_dir.mkdir()
        self.config = SimpleNamespace(delivery_dir=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _task(self, **overrides):
        defaults = dict(
            id="CT-TEST",
            project_path=Path("/fake/project"),
            task_type="feature",
            domain="web_dev",
            language="python",
            prompt_qwen="来自 TaskConfig 的默认 prompt",
            prompt_claude="",
            followups_qwen=["来自 TaskConfig 的追问"],
            followups_claude=[],
            task_title="",
            task_description="",
            acceptance_criteria=[],
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _write_metadata(self, content: str):
        _write_tmp_in(self.metadata_dir, "CT-TEST.md", content)

    # ---- new format: metadata is authoritative ----

    def test_new_format_complete(self):
        """All fields come from metadata; no fallback fires."""
        project_dir = self.tmpdir / "myproject"
        project_dir.mkdir()
        self._write_metadata(
            "# CT-TEST Metadata\n\n"
            "## Codebase\n\n"
            "- Project path: D:\\projects\\myproject\n\n"
            "## Project Summary\n\n"
            "```text\n"
            "A Flask web app.\n"
            "```\n\n"
            "## Task Label\n\n"
            "- Task type: bug-fix\n"
            "- Application domain: web_dev\n"
            "- Language: python\n\n"
            "## Task Description\n\n"
            "- Title: Fix auth bug\n"
            "- Description: Login fails after timeout\n"
            "- Acceptance criteria:\n"
            "  - Sessions should persist\n\n"
            "## Qwen Conversation\n\n"
            "- Initial prompt:\n\n"
            "```text\n"
            "Fix the auth timeout\n"
            "```\n\n"
            "- Follow-up summary:\n\n"
            "```text\n"
            "- Check JWT expiry\n"
            "```\n"
        )
        task = self._task(project_path=project_dir)
        ctx = build_task_context(task, self.config)

        self.assertEqual(ctx["project_name"], "myproject")
        self.assertEqual(ctx["task_type"], "bug-fix")
        self.assertIn("Flask", ctx["project_summary"])
        self.assertEqual(ctx["task_title"], "Fix auth bug")
        self.assertEqual(ctx["task_description"], "Login fails after timeout")
        self.assertIn("Sessions should persist", ctx["acceptance_criteria"])
        self.assertIn("Fix the auth timeout", ctx["prompts"])
        self.assertIn("JWT expiry", ctx["followups"])

    # ---- old format: no Project Summary / Task Description sections ----

    def test_old_format_fallbacks(self):
        """Old metadata: task_description from prompt, project_summary synthesised."""
        self._write_metadata(OLD_FORMAT)
        task = self._task()
        ctx = build_task_context(task, self.config)

        # Structured fields from metadata
        self.assertEqual(ctx["task_type"], "feature-add")
        self.assertEqual(ctx["language"], "javascript")
        # task_description falls back to first Initial prompt
        self.assertEqual(ctx["task_description"],
                         "Add dark mode support to the settings page")
        # project_summary synthesised from prompts + followups
        self.assertIn("任务描述", ctx["project_summary"])
        self.assertIn("dark mode", ctx["project_summary"])
        self.assertIn("追问要点", ctx["project_summary"])

    # ---- partial metadata: defensive fill from TaskConfig ----

    def test_partial_metadata_defensive_fill(self):
        """Metadata present but without prompts — TaskConfig fills the gap."""
        self._write_metadata(
            "# CT-TEST\n\n"
            "## Codebase\n\n"
            "- Project path: D:\\projects\\partial\n\n"
            "## Task Label\n\n"
            "- Task type: refactor\n"
            "- Language: go\n"
        )
        task = self._task()
        ctx = build_task_context(task, self.config)

        # prompts / followups / task_description filled from TaskConfig
        self.assertEqual(ctx["prompts"], "来自 TaskConfig 的默认 prompt")
        self.assertEqual(ctx["task_description"], "来自 TaskConfig 的默认 prompt")
        self.assertIn("来自 TaskConfig 的追问", ctx["followups"])
        self.assertIn("来自 TaskConfig 的默认 prompt", ctx["project_summary"])

    # ---- metadata file does not exist at all ----

    def test_missing_metadata(self):
        """No metadata file: everything comes from TaskConfig."""
        task = self._task(task_title="Fix Widget")
        ctx = build_task_context(task, self.config)

        self.assertEqual(ctx["project_name"], "project")
        self.assertEqual(ctx["task_type"], "feature")
        self.assertEqual(ctx["prompts"], "来自 TaskConfig 的默认 prompt")
        self.assertEqual(ctx["task_title"], "Fix Widget")
        self.assertEqual(ctx["task_description"], "来自 TaskConfig 的默认 prompt")
        self.assertIn("来自 TaskConfig 的默认 prompt", ctx["project_summary"])

    # ---- regression: new-format structured task_description is preserved ----

    def test_new_format_task_description_not_overwritten(self):
        """Structured - Description must NOT be replaced by the prompt fallback."""
        self._write_metadata(NEW_FORMAT_FULL)
        task = self._task()
        ctx = build_task_context(task, self.config)

        self.assertEqual(ctx["task_description"],
                         "Users get logged out after 5 minutes instead of 30")
        self.assertEqual(ctx["task_title"], "Fix login timeout bug")
        self.assertIn("Flask", ctx["project_summary"])

    # ---- safety net: critical fields filled with defaults + warnings ----

    def test_safety_net_fills_empty_critical_fields(self):
        """When TaskConfig fields are empty, safety net fills defaults and warns."""
        task = self._task(
            task_type="", domain="", language="",
            prompt_qwen="", followups_qwen=[],
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ctx = build_task_context(task, self.config)
        out = buf.getvalue()

        # Critical fields must never be empty
        self.assertEqual(ctx["task_type"], "unknown")
        self.assertEqual(ctx["domain"], "unknown")
        self.assertEqual(ctx["language"], "unknown")
        self.assertEqual(ctx["task_description"], "（任务描述缺失）")

        self.assertIn("WARNING", out)
        self.assertIn("task context fields missing", out)
        self.assertIn("task_description is empty", out)

    def test_safety_net_not_triggered_for_complete_context(self):
        """Complete metadata produces no safety-net warnings."""
        self._write_metadata(NEW_FORMAT_FULL)
        project_dir = self.tmpdir / "myproject"
        project_dir.mkdir()
        task = self._task(project_path=project_dir)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_task_context(task, self.config)

        self.assertNotIn("WARNING", buf.getvalue())

    def test_safety_net_partial_warnings(self):
        """Safety net warns only about the specific missing fields."""
        task = self._task(task_type="", prompt_qwen="")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ctx = build_task_context(task, self.config)
        out = buf.getvalue()

        # Only task_type is empty among critical fields
        self.assertEqual(ctx["task_type"], "unknown")
        # domain/language are still from TaskConfig
        self.assertEqual(ctx["domain"], "web_dev")
        self.assertEqual(ctx["language"], "python")

        self.assertIn("task_type", out)
        # domain should NOT appear in the missing-fields list
        warn_msg = out.split("filled with defaults:")[1].split("\n")[0] if "filled with defaults:" in out else ""
        self.assertNotIn("domain", warn_msg)


if __name__ == "__main__":
    unittest.main()
