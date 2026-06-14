"""JSONL trajectory parsing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ctpipe.config import model_stem
from ctpipe.project_hash import CLAUDE_PROJECTS_DIR, project_hash_dir


@dataclass
class TrajectoryInfo:
    file_path: Path
    session_id: str = ""
    models: set[str] = field(default_factory=set)
    cwd_values: set[str] = field(default_factory=set)
    first_user_ts: str | None = None
    last_ts: str | None = None
    line_count: int = 0
    user_turns: int = 0
    first_user_query: str = ""

    @property
    def detected_provider(self) -> str:
        if any("qwen" in m.lower() for m in self.models):
            return "qwen"
        if any("claude" in m.lower() for m in self.models):
            return "claude"
        return "unknown"


def trajectory_filename(task_id: str, model: str) -> str:
    return f"{model_stem(task_id, model)}.jsonl"


def expected_delivery_path(delivery_dir: Path, model_name: str, task_id: str) -> Path:
    return delivery_dir / "trajectories" / model_name / trajectory_filename(task_id, model_name)


def find_delivery_trajectory(
    delivery_dir: Path,
    model_name: str,
    task_id: str,
    session_id: str | None = None,
) -> Path | None:
    expected = expected_delivery_path(delivery_dir, model_name, task_id)
    if expected.exists():
        return expected

    legacy = delivery_dir / "trajectories" / model_name / f"{task_id}.jsonl"
    if legacy.exists():
        return legacy

    traj_dir = delivery_dir / "trajectories" / model_name
    if not traj_dir.is_dir():
        return None

    if session_id:
        by_session = traj_dir / f"{session_id}.jsonl"
        if by_session.exists():
            return by_session

    canonical_stem = model_stem(task_id, model_name)
    for candidate in sorted(traj_dir.glob("*.jsonl")):
        if candidate.stem == canonical_stem or candidate.stem == task_id:
            return candidate
    return None


def parse_trajectory(jsonl_path: Path) -> TrajectoryInfo:
    info = TrajectoryInfo(file_path=jsonl_path)
    first_query_found = False
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            info.line_count += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("type") == "user":
                info.user_turns += 1
            if obj.get("sessionId") and not info.session_id:
                info.session_id = obj["sessionId"]
            if obj.get("cwd"):
                info.cwd_values.add(obj["cwd"])
            ts = obj.get("timestamp")
            if ts:
                info.last_ts = ts
                if info.first_user_ts is None and obj.get("type") == "user":
                    info.first_user_ts = ts
            msg = obj.get("message")
            if isinstance(msg, dict):
                if msg.get("role") == "assistant" and msg.get("model"):
                    info.models.add(msg["model"])
                # Extract first user query text for B1-d consistency check
                if not first_query_found and msg.get("role") == "user":
                    content = msg.get("content")
                    if content:
                        first_query_found = True
                        info.first_user_query = _extract_user_text(content)
    return info


def _extract_user_text(content: object) -> str:
    """Extract plain text from a user message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


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

    Fallback: Claude Code may write JSONL to the parent process's project hash
    directory instead of the cwd-based one.  If the primary lookup fails, scan
    all sibling project hash dirs for a matching session-id file.
    """
    proj_dir = project_hash_dir(run_dir)

    # Fallback: if primary proj_dir doesn't exist and we have a session id,
    # search all project hash dirs for {session_id}.jsonl
    if not proj_dir.is_dir() and expected_session_id and CLAUDE_PROJECTS_DIR.is_dir():
        for d in CLAUDE_PROJECTS_DIR.iterdir():
            if d.is_dir():
                candidate = d / f"{expected_session_id}.jsonl"
                if candidate.is_file():
                    proj_dir = d
                    break
        else:
            return None
    elif not proj_dir.is_dir():
        return None

    # Fast path: JSONL filename is {session_id}.jsonl
    if expected_session_id:
        direct = proj_dir / f"{expected_session_id}.jsonl"
        if direct.is_file() and direct.stat().st_mtime > start_time:
            return direct

    # Collect candidates: all JSONL files modified after run start
    candidates: list[tuple[Path, float]] = []
    for f in proj_dir.iterdir():
        if f.suffix == ".jsonl":
            mtime = f.stat().st_mtime
            if mtime > start_time:
                candidates.append((f, mtime))

    if not candidates:
        return None

    # Try matching by session_id in filename or file content
    if expected_session_id:
        for f, _ in candidates:
            if f.stem == expected_session_id:
                return f
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                for line_num, line in enumerate(fh):
                    if line_num >= 50:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    if obj.get("sessionId") == expected_session_id:
                        return f

    # Fallback: take most recent file
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def extract_for_scoring(jsonl_path: Path, max_chars: int = 50_000) -> str:
    """Extract a condensed text representation of the trajectory for AI scoring.

    Keeps user messages, assistant text, tool call summaries.
    Truncates tool results and skips base64/binary content.
    """
    # Key tool fields that carry important scoring evidence
    _KEY_FIELDS: dict[str, set[str]] = {
        "Edit": {"old_string", "new_string", "file_path"},
        "Write": {"file_path", "content"},
        "Bash": {"command"},
    }

    parts: list[str] = []
    total = 0
    seen_snippet_tool: dict[str, str] = {}   # snippet -> tool name that first produced it
    seen_error_snippets: set[str] = set()     # dedup set for error results
    seen_normal_snippets: set[str] = set()     # dedup set for normal results
    tool_name_by_id: dict[str, str] = {}

    with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            msg = obj.get("message")
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            content = msg.get("content")
            if not content:
                continue

            if role == "user":
                text_parts: list[str] = []
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_result":
                                is_error = block.get("is_error", False)
                                result_content = block.get("content") or ""
                                if isinstance(result_content, list):
                                    result_content = " ".join(
                                        b.get("text", "") for b in result_content
                                        if isinstance(b, dict) and b.get("type") == "text"
                                    )
                                result_content = str(result_content).strip()
                                snippet = result_content[:300]
                                if is_error:
                                    if snippet and snippet in seen_error_snippets:
                                        tn = seen_snippet_tool[snippet]
                                        entry = f"[Result: ERROR same as {tn}]".rstrip()
                                    else:
                                        entry = f"[Result: ERROR] {snippet}".rstrip()
                                        if snippet:
                                            seen_error_snippets.add(snippet)
                                            seen_snippet_tool[snippet] = tool_name_by_id.get(
                                                block.get("tool_use_id", ""), ""
                                            )
                                    text_parts.append(entry)
                                elif result_content:
                                    if snippet in seen_normal_snippets:
                                        tn = seen_snippet_tool[snippet]
                                        text_parts.append(f"[Result: same as {tn}]")
                                    else:
                                        seen_normal_snippets.add(snippet)
                                        seen_snippet_tool[snippet] = tool_name_by_id.get(
                                            block.get("tool_use_id", ""), ""
                                        )
                                        text_parts.append(f"[Result: {snippet}]")
                                else:
                                    text_parts.append("[Result: ok]")
                        elif isinstance(block, str):
                            text_parts.append(block)
                if text_parts:
                    combined = "\n".join(text_parts)
                    chunk = f"\n=== USER ===\n{combined}\n"
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
                                tool_id = block.get("id", "")
                                if tool_id:
                                    tool_name_by_id[tool_id] = name
                                inp = block.get("input", {})
                                key_fields = _KEY_FIELDS.get(name, set())
                                summary_parts = []
                                for k, v in (inp.items() if isinstance(inp, dict) else []):
                                    sv = str(v)
                                    if k in key_fields:
                                        if len(sv) > 300:
                                            sv = sv[:200] + "..."
                                    else:
                                        if len(sv) > 80:
                                            sv = sv[:40] + "..."
                                    summary_parts.append(f"{k}={sv}")
                                    if len(summary_parts) >= (5 if key_fields else 3):
                                        break
                                inp_summary = ", ".join(summary_parts) or "..."
                                text_parts.append(f"[Tool: {name}({inp_summary})]")
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

