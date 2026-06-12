"""Tests for scripts/health_check.sh — overall_status → exit-code mapping.

The script wraps `ctpipe health --output <file>`, reads overall_status from the
JSON report, and exits 0/1/2 for healthy/degraded/critical (3 on anything else).

We isolate the parse-and-map logic from real ctpipe by injecting CTPIPE_CMD=true:
`true` ignores its arguments and returns 0 without writing the report, so the
report we pre-write on disk is exactly what the script ends up parsing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def _run(report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [BASH, "scripts/health_check.sh", "--output", report.as_posix()],
        cwd=REPO,
        env={**os.environ, "CTPIPE_CMD": "true"},
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [("healthy", 0), ("degraded", 1), ("critical", 2)],
)
def test_exit_code_matches_status(tmp_path: Path, status: str, expected_code: int) -> None:
    report = tmp_path / "health_report.json"
    report.write_text(f'{{"overall_status": "{status}"}}\n', encoding="utf-8")
    proc = _run(report)
    assert proc.returncode == expected_code, (
        f"status={status!r}: expected exit {expected_code}, got {proc.returncode}\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )


def test_exit_code_3_on_unknown_status(tmp_path: Path) -> None:
    report = tmp_path / "health_report.json"
    report.write_text('{"overall_status": "bogus"}\n', encoding="utf-8")
    proc = _run(report)
    assert proc.returncode == 3, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


def test_exit_code_3_on_missing_field(tmp_path: Path) -> None:
    report = tmp_path / "health_report.json"
    report.write_text('{"other": "x"}\n', encoding="utf-8")
    proc = _run(report)
    assert proc.returncode == 3, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
