"""Unit tests for ctpipe.trajectory — parsing, lookup, and utility functions.

Covers everything except extract_for_scoring (which has its own test file).
All tests are pure pytest-style, use tmp_path for filesystem isolation,
and reuse helpers from conftest where applicable.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from ctpipe.trajectory import (
    TrajectoryInfo,
    _extract_user_text,
    expected_delivery_path,
    find_delivery_trajectory,
    find_trajectory_for_run,
    parse_trajectory,
    trajectory_filename,
)
from conftest import write_trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> Path:
    """Write records as JSON lines and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _write_raw(path: Path, text: str) -> Path:
    """Write raw text to a file and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ===========================================================================
# TrajectoryInfo.detected_provider
# ===========================================================================

class TestDetectedProvider:

    def test_qwen_detected(self):
        info = TrajectoryInfo(file_path=Path("x.jsonl"), models={"qwen-max-latest"})
        assert info.detected_provider == "qwen"

    def test_claude_detected(self):
        info = TrajectoryInfo(file_path=Path("x.jsonl"), models={"claude-sonnet-4-20250514"})
        assert info.detected_provider == "claude"

    def test_unknown_when_no_models(self):
        info = TrajectoryInfo(file_path=Path("x.jsonl"))
        assert info.detected_provider == "unknown"

    def test_unknown_when_unrelated_model(self):
        info = TrajectoryInfo(file_path=Path("x.jsonl"), models={"gpt-4o"})
        assert info.detected_provider == "unknown"

    def test_qwen_takes_priority_over_claude(self):
        """When both models appear, qwen is returned (checked first)."""
        info = TrajectoryInfo(
            file_path=Path("x.jsonl"),
            models={"claude-sonnet-4-20250514", "qwen-max-latest"},
        )
        assert info.detected_provider == "qwen"

    def test_case_insensitive_match(self):
        info = TrajectoryInfo(file_path=Path("x.jsonl"), models={"Qwen-Max-Latest"})
        assert info.detected_provider == "qwen"

    def test_case_insensitive_claude(self):
        info = TrajectoryInfo(file_path=Path("x.jsonl"), models={"Claude-3-Opus"})
        assert info.detected_provider == "claude"


# ===========================================================================
# trajectory_filename
# ===========================================================================

class TestTrajectoryFilename:

    def test_standard_task_id(self):
        assert trajectory_filename("CT-0038", "qwen") == "qwen-0038.jsonl"

    def test_standard_task_id_claude(self):
        assert trajectory_filename("CT-0001", "claude") == "claude-0001.jsonl"

    def test_bare_id_no_hyphen(self):
        assert trajectory_filename("mytask", "qwen") == "qwen-mytask.jsonl"

    def test_multi_hyphen_task_id(self):
        assert trajectory_filename("CT-A-0001", "qwen") == "qwen-A-0001.jsonl"


# ===========================================================================
# expected_delivery_path
# ===========================================================================

class TestExpectedDeliveryPath:

    def test_path_structure(self):
        result = expected_delivery_path(Path("/delivery"), "qwen", "CT-0038")
        assert result == Path("/delivery/trajectories/qwen/qwen-0038.jsonl")

    def test_path_is_path_object(self):
        result = expected_delivery_path(Path("/d"), "claude", "CT-0001")
        assert isinstance(result, Path)


# ===========================================================================
# find_delivery_trajectory
# ===========================================================================

class TestFindDeliveryTrajectory:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmpdir = tmp_path

    def _traj_dir(self, model: str = "qwen") -> Path:
        d = self.tmpdir / "trajectories" / model
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -----------------------------------------------------------------------
    # 1. 新命名优先 (expected path = {model}-{num}.jsonl)
    # -----------------------------------------------------------------------

    def test_returns_expected_path_when_present(self):
        d = self._traj_dir()
        expected = d / "qwen-0038.jsonl"
        expected.write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(self.tmpdir, "qwen", "CT-0038") == expected

    def test_new_naming_wins_over_legacy(self):
        d = self._traj_dir()
        expected = d / "qwen-0038.jsonl"
        expected.write_text("new", encoding="utf-8")
        (d / "CT-0038.jsonl").write_text("old", encoding="utf-8")
        assert find_delivery_trajectory(self.tmpdir, "qwen", "CT-0038") == expected

    def test_new_naming_wins_over_session_id(self):
        d = self._traj_dir()
        expected = d / "qwen-0038.jsonl"
        expected.write_text("{}", encoding="utf-8")
        (d / "sess-abc.jsonl").write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-0038", session_id="sess-abc",
        ) == expected

    def test_new_naming_for_claude(self):
        d = self._traj_dir("claude")
        expected = d / "claude-0042.jsonl"
        expected.write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(self.tmpdir, "claude", "CT-0042") == expected

    # -----------------------------------------------------------------------
    # 2. 旧命名回退 (legacy = {task_id}.jsonl)
    # -----------------------------------------------------------------------

    def test_falls_back_to_legacy_filename(self):
        d = self._traj_dir()
        legacy = d / "CT-0038.jsonl"
        legacy.write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(self.tmpdir, "qwen", "CT-0038") == legacy

    def test_legacy_wins_over_session_id(self):
        d = self._traj_dir()
        legacy = d / "CT-0038.jsonl"
        legacy.write_text("legacy", encoding="utf-8")
        (d / "sess-xyz.jsonl").write_text("session", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-0038", session_id="sess-xyz",
        ) == legacy

    def test_legacy_with_no_hyphen_in_task_id(self):
        d = self._traj_dir()
        legacy = d / "bareid.jsonl"
        legacy.write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(self.tmpdir, "qwen", "bareid") == legacy

    # -----------------------------------------------------------------------
    # 3. session_id 文件名命中
    # -----------------------------------------------------------------------

    def test_falls_back_to_session_id(self):
        d = self._traj_dir()
        session_file = d / "sess-abc-123.jsonl"
        session_file.write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-0038", session_id="sess-abc-123",
        ) == session_file

    def test_session_id_file_with_uuid_style_name(self):
        sid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        d = self._traj_dir()
        (d / f"{sid}.jsonl").write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-0038", session_id=sid,
        ) == d / f"{sid}.jsonl"

    def test_session_id_ignored_when_none(self):
        d = self._traj_dir()
        (d / "sess-xyz.jsonl").write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-0038", session_id=None,
        ) is None

    def test_session_id_ignored_when_empty_string(self):
        d = self._traj_dir()
        (d / ".jsonl").write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-0038", session_id="",
        ) is None

    def test_session_id_file_missing_falls_through_to_glob(self):
        d = self._traj_dir()
        (d / "qwen-0038.jsonl").write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-0038", session_id="sess-not-here",
        ) == d / "qwen-0038.jsonl"

    def test_session_id_wins_over_glob(self):
        d = self._traj_dir()
        session_file = d / "sess-preferred.jsonl"
        session_file.write_text("session", encoding="utf-8")
        (d / "qwen-0038.jsonl").write_text("glob", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-9999", session_id="sess-preferred",
        ) == session_file

    # -----------------------------------------------------------------------
    # 4. glob 分支
    # -----------------------------------------------------------------------

    def test_glob_matches_canonical_stem(self):
        d = self._traj_dir()
        candidate = d / "qwen-0038.jsonl"
        candidate.write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(self.tmpdir, "qwen", "CT-0038") == candidate

    def test_glob_matches_task_id_stem(self):
        d = self._traj_dir()
        candidate = d / "CT-0038.jsonl"
        candidate.write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(self.tmpdir, "qwen", "CT-0038") == candidate

    def test_glob_both_stems_present_expected_wins(self):
        d = self._traj_dir()
        expected = d / "qwen-0038.jsonl"
        expected.write_text("expected", encoding="utf-8")
        (d / "CT-0038.jsonl").write_text("legacy", encoding="utf-8")
        assert find_delivery_trajectory(self.tmpdir, "qwen", "CT-0038") == expected

    def test_glob_no_match_for_unrelated_files(self):
        d = self._traj_dir()
        (d / "qwen-9999.jsonl").write_text("{}", encoding="utf-8")
        (d / "other-task.jsonl").write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(self.tmpdir, "qwen", "CT-0038") is None

    # -----------------------------------------------------------------------
    # 5. 全找不到 → None
    # -----------------------------------------------------------------------

    def test_returns_none_when_dir_missing(self):
        assert find_delivery_trajectory(self.tmpdir, "qwen", "CT-0038") is None

    def test_returns_none_when_dir_empty(self):
        self._traj_dir()
        assert find_delivery_trajectory(self.tmpdir, "qwen", "CT-0038") is None

    def test_returns_none_when_no_match_in_dir(self):
        d = self._traj_dir()
        (d / "unrelated-file.jsonl").write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(self.tmpdir, "qwen", "CT-0038") is None

    def test_returns_none_with_session_id_but_no_file(self):
        d = self._traj_dir()
        (d / "other.jsonl").write_text("{}", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-0038", session_id="sess-missing",
        ) is None

    # -----------------------------------------------------------------------
    # 6. 完整优先级链
    # -----------------------------------------------------------------------

    def test_priority_new_over_legacy_and_session_and_glob(self):
        d = self._traj_dir()
        new = d / "qwen-0038.jsonl"
        new.write_text("new", encoding="utf-8")
        (d / "CT-0038.jsonl").write_text("legacy", encoding="utf-8")
        (d / "sess-full.jsonl").write_text("session", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-0038", session_id="sess-full",
        ) == new

    def test_priority_legacy_over_session_when_new_missing(self):
        d = self._traj_dir()
        legacy = d / "CT-0038.jsonl"
        legacy.write_text("legacy", encoding="utf-8")
        (d / "sess-full.jsonl").write_text("session", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-0038", session_id="sess-full",
        ) == legacy

    def test_priority_session_over_glob_when_new_and_legacy_missing(self):
        d = self._traj_dir()
        session = d / "sess-glob.jsonl"
        session.write_text("session", encoding="utf-8")
        (d / "other-random.jsonl").write_text("noise", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-0050", session_id="sess-glob",
        ) == session

    def test_full_chain_only_unrelated_files_left(self):
        d = self._traj_dir()
        (d / "random-noise.jsonl").write_text("noise", encoding="utf-8")
        (d / "another-task.jsonl").write_text("other", encoding="utf-8")
        assert find_delivery_trajectory(
            self.tmpdir, "qwen", "CT-0077", session_id="sess-nope",
        ) is None


# ===========================================================================
# parse_trajectory
# ===========================================================================

class TestParseTrajectory:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmpdir = tmp_path

    def _write(self, name: str, records: list[dict]) -> Path:
        return _write_jsonl(self.tmpdir / name, records)

    # --- basic metadata extraction ---

    def test_session_id_extracted(self):
        p = self._write("a.jsonl", [{"sessionId": "sess-001", "type": "system"}])
        assert parse_trajectory(p).session_id == "sess-001"

    def test_first_session_id_wins(self):
        p = self._write("a.jsonl", [
            {"sessionId": "first", "type": "system"},
            {"sessionId": "second", "type": "system"},
        ])
        assert parse_trajectory(p).session_id == "first"

    def test_line_count(self):
        p = self._write("a.jsonl", [
            {"type": "system"},
            {"type": "user", "message": {"role": "user", "content": "hi"}},
            {"type": "assistant", "message": {"role": "assistant", "content": "hello"}},
        ])
        assert parse_trajectory(p).line_count == 3

    def test_line_count_skips_blank_lines(self):
        p = _write_raw(self.tmpdir / "blank.jsonl",
            '{"type":"system"}\n\n\n{"type":"user","message":{"role":"user","content":"hi"}}\n')
        assert parse_trajectory(p).line_count == 2

    def test_line_count_includes_malformed_json(self):
        p = _write_raw(self.tmpdir / "mixed.jsonl",
            '{"type":"system"}\n'
            'this is not valid json\n'
            '{also broken\n'
            '{"type":"user","message":{"role":"user","content":"ok"}}\n')
        info = parse_trajectory(p)
        assert info.line_count == 4

    def test_line_count_includes_non_dict_json(self):
        p = _write_raw(self.tmpdir / "nonobj.jsonl",
            '{"type":"system"}\n'
            '[1, 2, 3]\n'
            '"just a string"\n'
            '42\n'
            'null\n')
        assert parse_trajectory(p).line_count == 5

    def test_line_count_only_blank_and_whitespace_skipped(self):
        p = _write_raw(self.tmpdir / "ws.jsonl",
            '{"type":"system"}\n'
            '   \n'
            '\t\n'
            '{"type":"user","message":{"role":"user","content":"q"}}\n')
        assert parse_trajectory(p).line_count == 2

    def test_line_count_realistic_trajectory(self):
        p = self._write("real.jsonl", [
            {"sessionId": "s1", "type": "system"},
            {"type": "user", "message": {"role": "user", "content": "q1"}},
            {"type": "assistant", "message": {"role": "assistant", "model": "qwen-max", "content": "a1"}},
            {"type": "user", "message": {"role": "user", "content": "q2"}},
            {"type": "assistant", "message": {"role": "assistant", "model": "qwen-max", "content": "a2"}},
            {"type": "user", "message": {"role": "user", "content": "q3"}},
            {"type": "assistant", "message": {"role": "assistant", "model": "qwen-max", "content": "a3"}},
        ])
        assert parse_trajectory(p).line_count == 7

    # --- user turns ---

    def test_user_turns_counted(self):
        p = self._write("a.jsonl", [
            {"type": "user", "message": {"role": "user", "content": "q1"}},
            {"type": "assistant", "message": {"role": "assistant", "content": "a1"}},
            {"type": "user", "message": {"role": "user", "content": "q2"}},
            {"type": "assistant", "message": {"role": "assistant", "content": "a2"}},
        ])
        assert parse_trajectory(p).user_turns == 2

    def test_zero_user_turns(self):
        p = self._write("a.jsonl", [{"type": "system"}])
        assert parse_trajectory(p).user_turns == 0

    def test_user_turns_uses_top_level_type_not_message_role(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "user", "content": "mismatched"}},
        ])
        assert parse_trajectory(p).user_turns == 0

    def test_user_turns_type_user_without_message(self):
        p = self._write("a.jsonl", [
            {"type": "user"},
            {"type": "user", "message": {"role": "user", "content": "normal"}},
        ])
        assert parse_trajectory(p).user_turns == 2

    def test_user_turns_no_type_field(self):
        p = self._write("a.jsonl", [
            {"message": {"role": "user", "content": "no type at top level"}},
            {"info": "metadata only"},
        ])
        assert parse_trajectory(p).user_turns == 0

    def test_user_turns_ignores_malformed_lines(self):
        p = _write_raw(self.tmpdir / "bad.jsonl",
            'not json at all\n'
            '{"type":"user","message":{"role":"user","content":"valid"}}\n'
            '{broken\n')
        info = parse_trajectory(p)
        assert info.line_count == 3
        assert info.user_turns == 1

    def test_user_turns_realistic_mix(self):
        p = self._write("mix.jsonl", [
            {"sessionId": "s1", "type": "system"},
            {"type": "user", "message": {"role": "user", "content": "start"}},
            {"type": "assistant", "message": {"role": "assistant", "model": "claude-3-opus", "content": "ok"}},
            {"type": "user", "message": {"role": "user", "content": "next"}},
            {"type": "assistant", "message": {"role": "assistant", "model": "claude-3-opus", "content": "done"}},
            {"type": "user", "message": {"role": "user", "content": "fix this"}},
            {"type": "assistant", "message": {"role": "assistant", "model": "claude-3-opus", "content": "fixed"}},
        ])
        info = parse_trajectory(p)
        assert info.user_turns == 3
        assert info.line_count == 7

    # --- session_id ---

    def test_session_id_skips_empty_string(self):
        p = self._write("a.jsonl", [
            {"sessionId": "", "type": "system"},
            {"sessionId": "real-id", "type": "user"},
        ])
        assert parse_trajectory(p).session_id == "real-id"

    def test_session_id_not_overwritten_once_set(self):
        p = self._write("a.jsonl", [
            {"sessionId": "original", "type": "system"},
            {"sessionId": "replacement", "type": "user"},
            {"sessionId": "another", "type": "assistant"},
        ])
        assert parse_trajectory(p).session_id == "original"

    def test_session_id_from_later_line_if_early_missing(self):
        p = self._write("a.jsonl", [
            {"type": "system"},
            {"type": "user", "message": {"role": "user", "content": "q"}},
            {"sessionId": "late-id", "type": "assistant", "message": {"role": "assistant", "content": "a"}},
        ])
        assert parse_trajectory(p).session_id == "late-id"

    def test_session_id_remains_empty_when_none_present(self):
        p = self._write("a.jsonl", [
            {"type": "system"},
            {"type": "user", "message": {"role": "user", "content": "q"}},
        ])
        assert parse_trajectory(p).session_id == ""

    # --- model extraction ---

    def test_model_extracted_from_assistant(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "model": "qwen-max", "content": "hi"}},
        ])
        assert "qwen-max" in parse_trajectory(p).models

    def test_multiple_models(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "model": "qwen-max", "content": "a"}},
            {"type": "assistant", "message": {"role": "assistant", "model": "qwen-plus", "content": "b"}},
        ])
        assert parse_trajectory(p).models == {"qwen-max", "qwen-plus"}

    def test_no_model_from_user_message(self):
        p = self._write("a.jsonl", [
            {"type": "user", "message": {"role": "user", "model": "gpt-4", "content": "hi"}},
        ])
        assert "gpt-4" not in parse_trajectory(p).models

    def test_model_not_collected_without_model_field(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "content": "no model"}},
        ])
        assert parse_trajectory(p).models == set()

    def test_model_empty_string_not_collected(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "model": "", "content": "x"}},
        ])
        assert parse_trajectory(p).models == set()

    def test_model_none_not_collected(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "model": None, "content": "x"}},
        ])
        assert parse_trajectory(p).models == set()

    def test_model_requires_assistant_role(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "user", "model": "sneaky-model", "content": "x"}},
        ])
        assert parse_trajectory(p).models == set()

    def test_model_duplicate_deduped_by_set(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "model": "qwen-max", "content": "a"}},
            {"type": "assistant", "message": {"role": "assistant", "model": "qwen-max", "content": "b"}},
            {"type": "assistant", "message": {"role": "assistant", "model": "qwen-max", "content": "c"}},
        ])
        assert parse_trajectory(p).models == {"qwen-max"}

    # --- cwd extraction ---

    def test_cwd_extracted(self):
        p = self._write("a.jsonl", [
            {"type": "system", "cwd": "/home/user/project"},
            {"type": "user", "cwd": "/home/user/project/src"},
        ])
        assert parse_trajectory(p).cwd_values == {"/home/user/project", "/home/user/project/src"}

    def test_no_cwd(self):
        p = self._write("a.jsonl", [{"type": "system"}])
        assert parse_trajectory(p).cwd_values == set()

    # --- timestamps ---

    def test_last_timestamp(self):
        p = self._write("a.jsonl", [
            {"type": "system", "timestamp": "2025-01-01T00:00:00Z"},
            {"type": "user", "timestamp": "2025-01-01T00:01:00Z"},
            {"type": "assistant", "timestamp": "2025-01-01T00:02:00Z"},
        ])
        assert parse_trajectory(p).last_ts == "2025-01-01T00:02:00Z"

    def test_first_user_timestamp(self):
        p = self._write("a.jsonl", [
            {"type": "system", "timestamp": "2025-01-01T00:00:00Z"},
            {"type": "user", "timestamp": "2025-01-01T00:01:00Z"},
            {"type": "user", "timestamp": "2025-01-01T00:05:00Z"},
        ])
        assert parse_trajectory(p).first_user_ts == "2025-01-01T00:01:00Z"

    def test_no_timestamps(self):
        p = self._write("a.jsonl", [{"type": "system"}])
        info = parse_trajectory(p)
        assert info.last_ts is None
        assert info.first_user_ts is None

    # --- first user query ---

    def test_first_user_query_string_content(self):
        p = self._write("a.jsonl", [
            {"type": "user", "message": {"role": "user", "content": "Fix the bug in auth.py"}},
            {"type": "user", "message": {"role": "user", "content": "Now add tests"}},
        ])
        assert parse_trajectory(p).first_user_query == "Fix the bug in auth.py"

    def test_first_user_query_block_content(self):
        p = self._write("a.jsonl", [
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "Please fix this"},
                {"type": "text", "text": "and that"},
            ]}},
        ])
        assert parse_trajectory(p).first_user_query == "Please fix this\nand that"

    def test_first_user_query_empty_when_no_user_messages(self):
        p = self._write("a.jsonl", [
            {"type": "system"},
            {"type": "assistant", "message": {"role": "assistant", "content": "hi"}},
        ])
        assert parse_trajectory(p).first_user_query == ""

    # --- detected_provider integration ---

    def test_detected_provider_qwen(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "model": "qwen-max-latest", "content": "hi"}},
        ])
        assert parse_trajectory(p).detected_provider == "qwen"

    def test_detected_provider_claude(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "model": "claude-sonnet-4-20250514", "content": "hi"}},
        ])
        assert parse_trajectory(p).detected_provider == "claude"

    def test_detected_provider_unknown_when_no_model(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "content": "no model field"}},
        ])
        info = parse_trajectory(p)
        assert info.models == set()
        assert info.detected_provider == "unknown"

    def test_detected_provider_unknown_for_unrelated_model(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "model": "gpt-4o-mini", "content": "x"}},
        ])
        assert parse_trajectory(p).detected_provider == "unknown"

    def test_detected_provider_qwen_case_insensitive(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "model": "Qwen-Max-Latest", "content": "x"}},
        ])
        assert parse_trajectory(p).detected_provider == "qwen"

    def test_detected_provider_qwen_wins_over_claude(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "model": "claude-sonnet-4-20250514", "content": "a"}},
            {"type": "assistant", "message": {"role": "assistant", "model": "qwen-max-latest", "content": "b"}},
        ])
        info = parse_trajectory(p)
        assert info.models == {"claude-sonnet-4-20250514", "qwen-max-latest"}
        assert info.detected_provider == "qwen"

    def test_detected_provider_only_claude(self):
        p = self._write("a.jsonl", [
            {"type": "assistant", "message": {"role": "assistant", "model": "claude-3-opus", "content": "a"}},
            {"type": "assistant", "message": {"role": "assistant", "model": "claude-3-sonnet", "content": "b"}},
        ])
        assert parse_trajectory(p).detected_provider == "claude"

    # --- malformed input handling ---

    def test_skips_malformed_json_lines(self):
        p = _write_raw(self.tmpdir / "bad.jsonl",
            '{"type":"system"}\n'
            'this is not json\n'
            '{"type":"user","message":{"role":"user","content":"ok"}}\n')
        info = parse_trajectory(p)
        assert info.line_count == 3
        assert info.user_turns == 1

    def test_skips_non_dict_lines(self):
        p = _write_raw(self.tmpdir / "arr.jsonl",
            '{"type":"system"}\n'
            '[1, 2, 3]\n'
            '"just a string"\n')
        info = parse_trajectory(p)
        assert info.line_count == 3
        assert info.user_turns == 0

    def test_empty_file(self):
        p = _write_raw(self.tmpdir / "empty.jsonl", "")
        info = parse_trajectory(p)
        assert info.line_count == 0
        assert info.session_id == ""
        assert info.user_turns == 0
        assert info.detected_provider == "unknown"

    def test_file_path_stored(self):
        p = self._write("a.jsonl", [{"type": "system"}])
        assert parse_trajectory(p).file_path == p

    # --- conftest write_trajectory helper ---

    def test_conftest_write_trajectory_parses_correctly(self, tmp_path):
        """write_trajectory from conftest produces a file parse_trajectory can read."""
        p = tmp_path / "trajectories" / "qwen" / "qwen-0001.jsonl"
        write_trajectory(p, session_id="test-sess", model_name="qwen", user_turns=3)
        info = parse_trajectory(p)
        assert info.session_id == "test-sess"
        assert info.user_turns == 3
        assert info.line_count >= 12
        assert "qwen-model-v1" in info.models


# ===========================================================================
# _extract_user_text
# ===========================================================================

class TestExtractUserText:

    def test_string_content(self):
        assert _extract_user_text("hello world") == "hello world"

    def test_list_of_strings(self):
        assert _extract_user_text(["line1", "line2"]) == "line1\nline2"

    def test_list_with_text_blocks(self):
        content = [{"type": "text", "text": "block one"}, {"type": "text", "text": "block two"}]
        assert _extract_user_text(content) == "block one\nblock two"

    def test_list_skips_non_text_blocks(self):
        content = [
            {"type": "text", "text": "visible"},
            {"type": "image", "source": {"type": "base64", "data": "xxx"}},
            {"type": "tool_result", "content": "result"},
        ]
        assert _extract_user_text(content) == "visible"

    def test_list_mixed_strings_and_blocks(self):
        content = ["raw string", {"type": "text", "text": "block"}]
        assert _extract_user_text(content) == "raw string\nblock"

    def test_empty_string(self):
        assert _extract_user_text("") == ""

    def test_empty_list(self):
        assert _extract_user_text([]) == ""

    def test_none_returns_empty(self):
        assert _extract_user_text(None) == ""

    def test_integer_returns_empty(self):
        assert _extract_user_text(42) == ""

    def test_dict_returns_empty(self):
        assert _extract_user_text({"type": "text", "text": "hi"}) == ""

    def test_text_block_with_missing_text_key(self):
        assert _extract_user_text([{"type": "text"}]) == ""


# ===========================================================================
# find_trajectory_for_run
# ===========================================================================

class TestFindTrajectoryForRun:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmpdir = tmp_path
        self.fake_projects = tmp_path / "projects"
        self.fake_projects.mkdir()
        self.run_dir = tmp_path / "runs" / "myproject"
        self.run_dir.mkdir(parents=True)
        self.start_time = time.time() - 100

    def _proj_dir(self, name: str = "D--A3Code-myproject") -> Path:
        d = self.fake_projects / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _patch_project_hash(self, proj_dir: Path):
        return (
            patch("ctpipe.trajectory.project_hash_dir", return_value=proj_dir),
            patch("ctpipe.trajectory.CLAUDE_PROJECTS_DIR", self.fake_projects),
        )

    # --- primary lookup by session id (fast path) ---

    def test_finds_by_session_id_fast_path(self):
        pd = self._proj_dir()
        session_file = pd / "sess-abc.jsonl"
        session_file.write_text('{"sessionId":"sess-abc"}', encoding="utf-8")
        os.utime(session_file, (self.start_time + 50, self.start_time + 50))
        p1, p2 = self._patch_project_hash(pd)
        with p1, p2:
            result = find_trajectory_for_run(
                self.run_dir, self.start_time, expected_session_id="sess-abc")
        assert result == session_file

    def test_ignores_session_file_older_than_start(self):
        pd = self._proj_dir()
        session_file = pd / "sess-abc.jsonl"
        session_file.write_text('{"sessionId":"sess-abc"}', encoding="utf-8")
        os.utime(session_file, (self.start_time - 200, self.start_time - 200))
        p1, p2 = self._patch_project_hash(pd)
        with p1, p2:
            result = find_trajectory_for_run(
                self.run_dir, self.start_time, expected_session_id="sess-abc")
        assert result is None

    # --- content-based session id matching ---

    def test_finds_by_session_id_in_content(self):
        pd = self._proj_dir()
        jsonl = pd / "random-name.jsonl"
        jsonl.write_text(
            json.dumps({"sessionId": "sess-xyz", "type": "system"}) + "\n", encoding="utf-8")
        os.utime(jsonl, (self.start_time + 50, self.start_time + 50))
        p1, p2 = self._patch_project_hash(pd)
        with p1, p2:
            result = find_trajectory_for_run(
                self.run_dir, self.start_time, expected_session_id="sess-xyz")
        assert result == jsonl

    def test_session_id_deep_in_file(self):
        pd = self._proj_dir()
        jsonl = pd / "deep.jsonl"
        lines = [json.dumps({"type": "system", "info": f"padding {i}"}) for i in range(10)]
        lines.append(json.dumps({"sessionId": "sess-deep", "type": "user"}))
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.utime(jsonl, (self.start_time + 50, self.start_time + 50))
        p1, p2 = self._patch_project_hash(pd)
        with p1, p2:
            result = find_trajectory_for_run(
                self.run_dir, self.start_time, expected_session_id="sess-deep")
        assert result == jsonl

    def test_session_id_beyond_50_lines_not_scanned(self):
        pd = self._proj_dir()
        jsonl = pd / "toolong.jsonl"
        lines = [json.dumps({"type": "system", "info": f"padding {i}"}) for i in range(60)]
        lines.append(json.dumps({"sessionId": "sess-hidden", "type": "user"}))
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.utime(jsonl, (self.start_time + 50, self.start_time + 50))
        p1, p2 = self._patch_project_hash(pd)
        with p1, p2:
            result = find_trajectory_for_run(
                self.run_dir, self.start_time, expected_session_id="sess-hidden")
        # Falls back to most-recent file (which is this one)
        assert result == jsonl

    # --- fallback: most recent file ---

    def test_falls_back_to_most_recent(self):
        pd = self._proj_dir()
        old_file = pd / "old.jsonl"
        old_file.write_text("{}", encoding="utf-8")
        os.utime(old_file, (self.start_time + 10, self.start_time + 10))
        new_file = pd / "new.jsonl"
        new_file.write_text("{}", encoding="utf-8")
        os.utime(new_file, (self.start_time + 80, self.start_time + 80))
        p1, p2 = self._patch_project_hash(pd)
        with p1, p2:
            result = find_trajectory_for_run(self.run_dir, self.start_time)
        assert result == new_file

    def test_no_candidates_before_start_time(self):
        pd = self._proj_dir()
        old_file = pd / "old.jsonl"
        old_file.write_text("{}", encoding="utf-8")
        os.utime(old_file, (self.start_time - 200, self.start_time - 200))
        p1, p2 = self._patch_project_hash(pd)
        with p1, p2:
            result = find_trajectory_for_run(self.run_dir, self.start_time)
        assert result is None

    # --- project dir missing ---

    def test_returns_none_when_proj_dir_missing(self):
        nonexistent = self.fake_projects / "does-not-exist"
        p1, p2 = self._patch_project_hash(nonexistent)
        with p1, p2:
            result = find_trajectory_for_run(self.run_dir, self.start_time)
        assert result is None

    # --- fallback: search sibling project hash dirs ---

    def test_fallback_searches_sibling_dirs_by_session(self):
        other_proj = self.fake_projects / "E--other-project"
        other_proj.mkdir(parents=True)
        session_file = other_proj / "sess-fallback.jsonl"
        session_file.write_text('{"sessionId":"sess-fallback"}', encoding="utf-8")
        nonexistent = self.fake_projects / "does-not-exist"
        p1, p2 = self._patch_project_hash(nonexistent)
        with p1, p2:
            result = find_trajectory_for_run(
                self.run_dir, self.start_time, expected_session_id="sess-fallback")
        assert result == session_file

    def test_fallback_returns_none_when_no_sibling_match(self):
        nonexistent = self.fake_projects / "does-not-exist"
        p1, p2 = self._patch_project_hash(nonexistent)
        with p1, p2:
            result = find_trajectory_for_run(
                self.run_dir, self.start_time, expected_session_id="sess-nowhere")
        assert result is None

    def test_fallback_skipped_without_session_id(self):
        nonexistent = self.fake_projects / "does-not-exist"
        other_proj = self.fake_projects / "E--other"
        other_proj.mkdir(parents=True)
        (other_proj / "some-file.jsonl").write_text("{}", encoding="utf-8")
        p1, p2 = self._patch_project_hash(nonexistent)
        with p1, p2:
            result = find_trajectory_for_run(self.run_dir, self.start_time)
        assert result is None

    # --- non-jsonl files ignored ---

    def test_ignores_non_jsonl_files(self):
        pd = self._proj_dir()
        (pd / "notes.txt").write_text("not a trajectory", encoding="utf-8")
        (pd / "data.csv").write_text("a,b,c", encoding="utf-8")
        p1, p2 = self._patch_project_hash(pd)
        with p1, p2:
            result = find_trajectory_for_run(self.run_dir, self.start_time)
        assert result is None

    # --- malformed JSON in content scan ---

    def test_content_scan_skips_malformed_json(self):
        pd = self._proj_dir()
        jsonl = pd / "messy.jsonl"
        lines = [
            "not valid json",
            "{also broken",
            json.dumps({"sessionId": "sess-found", "type": "user"}),
        ]
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.utime(jsonl, (self.start_time + 50, self.start_time + 50))
        p1, p2 = self._patch_project_hash(pd)
        with p1, p2:
            result = find_trajectory_for_run(
                self.run_dir, self.start_time, expected_session_id="sess-found")
        assert result == jsonl

    # --- session id match by filename stem ---

    def test_matches_by_filename_stem(self):
        pd = self._proj_dir()
        jsonl = pd / "sess-byname.jsonl"
        jsonl.write_text('{"type":"system"}', encoding="utf-8")
        os.utime(jsonl, (self.start_time + 50, self.start_time + 50))
        p1, p2 = self._patch_project_hash(pd)
        with p1, p2:
            result = find_trajectory_for_run(
                self.run_dir, self.start_time, expected_session_id="sess-byname")
        assert result == jsonl


# ===========================================================================
# Integration: parse_trajectory + write_trajectory + detected_provider
# ===========================================================================

class TestParseTrajectoryIntegration:

    def test_full_trajectory_parsing(self, tmp_path):
        p = tmp_path / "full.jsonl"
        records = [
            {"sessionId": "sess-full", "type": "system", "cwd": "/proj",
             "timestamp": "2025-06-01T10:00:00Z"},
            {"type": "user", "sessionId": "sess-full", "cwd": "/proj",
             "timestamp": "2025-06-01T10:00:01Z",
             "message": {"role": "user", "content": "Add a login page"}},
            {"type": "assistant", "timestamp": "2025-06-01T10:00:05Z",
             "message": {"role": "assistant", "model": "qwen-max-latest",
                         "content": "Creating login page..."}},
            {"type": "user", "timestamp": "2025-06-01T10:01:00Z",
             "message": {"role": "user", "content": "Also add a logout button"}},
            {"type": "assistant", "timestamp": "2025-06-01T10:01:10Z",
             "message": {"role": "assistant", "model": "qwen-max-latest", "content": "Done."}},
        ]
        _write_jsonl(p, records)
        info = parse_trajectory(p)

        assert info.session_id == "sess-full"
        assert info.line_count == 5
        assert info.user_turns == 2
        assert info.models == {"qwen-max-latest"}
        assert info.cwd_values == {"/proj"}
        assert info.first_user_ts == "2025-06-01T10:00:01Z"
        assert info.last_ts == "2025-06-01T10:01:10Z"
        assert info.first_user_query == "Add a login page"
        assert info.detected_provider == "qwen"

    def test_conftest_write_trajectory_end_to_end(self, tmp_path):
        """End-to-end: write_trajectory → parse_trajectory → verify all fields."""
        p = tmp_path / "trajectories" / "claude" / "claude-0099.jsonl"
        write_trajectory(p, session_id="e2e-session", model_name="claude", user_turns=5)
        info = parse_trajectory(p)
        assert info.session_id == "e2e-session"
        assert info.user_turns == 5
        assert info.line_count >= 12
        assert "claude-model-v1" in info.models
        assert info.detected_provider == "claude"
