"""JSONL trajectory parsing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ctpipe.project_hash import project_hash_dir


@dataclass
class TrajectoryInfo:
    file_path: Path
    session_id: str = ""
    models: set[str] = field(default_factory=set)
    cwd_values: set[str] = field(default_factory=set)
    first_user_ts: str | None = None
    last_ts: str | None = None
    line_count: int = 0

    @property
    def detected_provider(self) -> str:
        if any("qwen" in m.lower() for m in self.models):
            return "qwen"
        if any("claude" in m.lower() for m in self.models):
            return "claude"
        return "unknown"


def trajectory_filename(task_id: str) -> str:
    return f"{task_id}.jsonl"


def expected_delivery_path(delivery_dir: Path, model_name: str, task_id: str) -> Path:
    return delivery_dir / "trajectories" / model_name / trajectory_filename(task_id)


def find_delivery_trajectory(
    delivery_dir: Path,
    model_name: str,
    task_id: str,
    session_id: str | None = None,
) -> Path | None:
    expected = expected_delivery_path(delivery_dir, model_name, task_id)
    if expected.exists():
        return expected

    traj_dir = delivery_dir / "trajectories" / model_name
    if not traj_dir.is_dir():
        return None

    if session_id:
        by_session = traj_dir / f"{session_id}.jsonl"
        if by_session.exists():
            return by_session

    for candidate in sorted(traj_dir.glob("*.jsonl")):
        if candidate.stem == task_id:
            return candidate
    return None


def parse_trajectory(jsonl_path: Path) -> TrajectoryInfo:
    info = TrajectoryInfo(file_path=jsonl_path)
    for raw in jsonl_path.open("r", encoding="utf-8", errors="replace"):
        raw = raw.strip()
        if not raw:
            continue
        info.line_count += 1
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("sessionId"):
            info.session_id = obj["sessionId"]
        if obj.get("cwd"):
            info.cwd_values.add(obj["cwd"])
        ts = obj.get("timestamp")
        if ts:
            info.last_ts = ts
            if info.first_user_ts is None and obj.get("type") == "user":
                info.first_user_ts = ts
        msg = obj.get("message")
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("model"):
            info.models.add(msg["model"])
    return info


def find_trajectory_for_run(
    run_dir: Path,
    start_time: float,
    expected_session_id: str | None = None,
) -> Path | None:
    """Find the trajectory JSONL file produced by a run in run_dir.

    Strategy:
    1. Compute project hash from run_dir → look in ~/.claude/projects/<hash>/
    2. Filter JSONL files with mtime > start_time
    3. If expected_session_id given, match by sessionId inside the file
    4. Otherwise take most recent by mtime
    """
    proj_dir = project_hash_dir(run_dir)
    if not proj_dir.is_dir():
        return None

    candidates: list[Path] = []
    for f in proj_dir.iterdir():
        if f.suffix == ".jsonl" and f.stat().st_mtime > start_time:
            candidates.append(f)

    if not candidates:
        return None

    if expected_session_id:
        for f in candidates:
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                first_line = fh.readline()
            if first_line.strip():
                try:
                    obj = json.loads(first_line)
                except json.JSONDecodeError:
                    continue
                if obj.get("sessionId") == expected_session_id:
                    return f

    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]


def extract_for_scoring(jsonl_path: Path, max_chars: int = 100_000) -> str:
    """Extract a condensed text representation of the trajectory for AI scoring.

    Keeps user messages, assistant text, tool call summaries.
    Truncates tool results and skips base64/binary content.
    """
    parts: list[str] = []
    total = 0

    for raw in jsonl_path.open("r", encoding="utf-8", errors="replace"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
        content = msg.get("content")
        if not content:
            continue

        if role == "user":
            text = _extract_text(content)
            if text:
                chunk = f"\n=== USER ===\n{text}\n"
                parts.append(chunk)
                total += len(chunk)

        elif role == "assistant":
            text_parts: list[str] = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            name = block.get("name", "?")
                            inp = block.get("input", {})
                            inp_str = json.dumps(inp, ensure_ascii=False)
                            if len(inp_str) > 500:
                                inp_str = inp_str[:500] + "..."
                            text_parts.append(f"[Tool: {name}({inp_str})]")
                        elif block.get("type") == "tool_result":
                            result_content = block.get("content", "")
                            if isinstance(result_content, str):
                                if len(result_content) > 500:
                                    result_content = result_content[:500] + "..."
                                text_parts.append(f"[Result: {result_content}]")
            if text_parts:
                combined = "\n".join(text_parts)
                chunk = f"\n=== ASSISTANT ===\n{combined}\n"
                parts.append(chunk)
                total += len(chunk)

        if total > max_chars:
            break

    result = "".join(parts)
    if len(result) > max_chars:
        keep_start = int(max_chars * 0.3)
        keep_end = max_chars - keep_start
        result = result[:keep_start] + "\n\n[... truncated ...]\n\n" + result[-keep_end:]
    return result


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)
    return ""
