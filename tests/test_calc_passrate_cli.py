"""Unit tests for rubrics_templates/calc_passrate.py CLI (main function).

Covers: unscored templates, incomplete rubrics, fully scored files,
--strict flag behaviour, directory scanning, and error paths.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ctpipe.config import (
    REFERENCE_CRITERION_DESCRIPTIONS,
    REFERENCE_CRITERION_NAMES,
)
from ctpipe.toml_utils import Criterion, write_quality_toml

# Make the script importable regardless of working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "rubrics_templates"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import calc_passrate  # noqa: E402

_NAMES = REFERENCE_CRITERION_NAMES[:10]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _criterion(name: str, score: int = 0, rationale: str = "") -> Criterion:
    return Criterion(
        name=name,
        description=REFERENCE_CRITERION_DESCRIPTIONS[name],
        type="likert",
        points=5,
        weight=1.0,
        score=score,
        rationale=rationale,
    )


def _write_scored(path: Path, count: int = 7, score: int = 4) -> None:
    criteria = [_criterion(_NAMES[i], score=score, rationale="good") for i in range(count)]
    write_quality_toml(path, criteria)


def _write_unscored(path: Path, count: int = 6) -> None:
    criteria = [_criterion(_NAMES[i]) for i in range(count)]
    write_quality_toml(path, criteria)


def _write_incomplete(path: Path, scored_count: int = 2, total: int = 6) -> None:
    criteria: list[Criterion] = []
    for i in range(total):
        if i < scored_count:
            criteria.append(_criterion(_NAMES[i], score=3, rationale="partial"))
        else:
            criteria.append(_criterion(_NAMES[i]))
    write_quality_toml(path, criteria)


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Invoke main() with *argv*, returning (exit_code, stdout, stderr)."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with patch.object(sys, "argv", ["calc_passrate.py", *argv]):
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = calc_passrate.main()
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
    return code, stdout.getvalue(), stderr.getvalue()


# ---------------------------------------------------------------------------
# fully scored — numeric output
# ---------------------------------------------------------------------------
class TestScoredOutput(unittest.TestCase):

    def test_scored_file_prints_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.quality.toml"
            _write_scored(f, count=7, score=4)
            code, out, err = _run([str(f)])
            self.assertEqual(code, 0)
            self.assertIn("0.8000", out)
            self.assertEqual(err, "")

    def test_scored_file_strict_still_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.quality.toml"
            _write_scored(f, count=7, score=4)
            code, out, err = _run(["--strict", str(f)])
            self.assertEqual(code, 0)
            self.assertIn("0.8000", out)
            self.assertEqual(err, "")


# ---------------------------------------------------------------------------
# unscored template — warning instead of 0.0000
# ---------------------------------------------------------------------------
class TestUnscoredTemplate(unittest.TestCase):

    def test_unscored_warns_on_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.quality.toml"
            _write_unscored(f, count=6)
            code, out, err = _run([str(f)])
            self.assertEqual(code, 0, "default mode should exit 0")
            self.assertEqual(out, "", "unscored file must NOT print to stdout")
            self.assertIn("ERROR", err)
            self.assertIn("unscored template", err)

    def test_unscored_strict_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.quality.toml"
            _write_unscored(f, count=6)
            code, out, err = _run(["--strict", str(f)])
            self.assertEqual(code, 1)
            self.assertIn("ERROR", err)


# ---------------------------------------------------------------------------
# incomplete rubric — partial scoring
# ---------------------------------------------------------------------------
class TestIncompleteRubric(unittest.TestCase):

    def test_incomplete_warns_on_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.quality.toml"
            _write_incomplete(f, scored_count=2, total=6)
            code, out, err = _run([str(f)])
            self.assertEqual(code, 0, "default mode should exit 0")
            self.assertEqual(out, "")
            self.assertIn("ERROR", err)
            self.assertIn("wrong criteria count", err)

    def test_incomplete_strict_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.quality.toml"
            _write_incomplete(f, scored_count=2, total=6)
            code, out, err = _run(["--strict", str(f)])
            self.assertEqual(code, 1)
            self.assertIn("ERROR", err)


# ---------------------------------------------------------------------------
# directory scanning
# ---------------------------------------------------------------------------
class TestDirectoryScanning(unittest.TestCase):

    def test_directory_processes_all_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_scored(d / "a.quality.toml", score=4)
            _write_scored(d / "b.quality.toml", score=3)
            code, out, err = _run([str(d)])
            self.assertEqual(code, 0)
            self.assertEqual(err, "")
            # both files should appear in stdout
            self.assertIn("a.quality.toml", out)
            self.assertIn("b.quality.toml", out)

    def test_mixed_directory_default_continues(self):
        """Default mode: invalid files warned, scored files printed, exit 0."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_scored(d / "scored.quality.toml", score=4)
            _write_unscored(d / "unscored.quality.toml")
            _write_incomplete(d / "partial.quality.toml", scored_count=1, total=6)

            code, out, err = _run([str(d)])
            self.assertEqual(code, 0)
            # scored file → numeric output on stdout
            self.assertIn("scored.quality.toml", out)
            self.assertIn("0.8000", out)
            # unscored + incomplete → specific errors on stderr
            self.assertIn("ERROR", err)

    def test_mixed_directory_strict_exits_nonzero(self):
        """--strict: any issue makes exit 1, but all files still processed."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_scored(d / "scored.quality.toml", score=4)
            _write_unscored(d / "unscored.quality.toml")

            code, out, err = _run(["--strict", str(d)])
            self.assertEqual(code, 1)
            # scored file still gets its output
            self.assertIn("scored.quality.toml", out)
            self.assertIn("unscored template", err)


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------
class TestErrorPaths(unittest.TestCase):

    def test_nonexistent_file_warns_default_mode(self):
        code, out, err = _run(["/no/such/file.toml"])
        self.assertEqual(code, 0, "non-strict mode exits 0 for any failure")
        self.assertIn("ERROR", err)

    def test_empty_toml_warns_default_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "empty.toml"
            f.write_text("", encoding="utf-8")
            code, out, err = _run([str(f)])
            self.assertEqual(code, 0, "non-strict mode exits 0 for any failure")
            self.assertIn("ERROR", err)


if __name__ == "__main__":
    unittest.main()
