"""gen subcommand: auto-generate tasks from GitHub projects."""

from __future__ import annotations

import asyncio
import json
import re
import time
import tomllib
from contextlib import nullcontext
from pathlib import Path

from ctpipe import strip_claude_wrapper
from ctpipe.config import BatchConfig, build_claude_env
from ctpipe.distribution import DISTRIBUTION, TaskSlot, expand_slot_for_batch, sample_slots

VALID_TASK_TYPES = {slot.task_type for slot in DISTRIBUTION}
from ctpipe.github_search import search_and_clone, search_and_clone_gitee
from ctpipe.toml_utils import Criterion, write_quality_toml

CRITERIA_NAMES = [
    "user_experience_and_interaction",
    "task_planning_and_execution_control",
    "semantic_understanding_and_logical_reasoning",
    "instruction_compliance_and_constraint_adherence",
    "engineering_quality_and_completeness",
    "delivery_completeness_and_usability",
    "architecture_boundaries_and_security_compliance",
]

SCAN_IGNORE = {
    "node_modules", ".venv", "__pycache__", ".git", "dist", ".next", ".nuxt",
    "build", "target", ".gradle", ".idea", ".vscode", "vendor", "coverage",
    ".tox", "eggs", ".mypy_cache", ".pytest_cache",
}

SCAN_IGNORE_SUFFIXES = {".egg-info"}

IDEA_PROMPT = """You are a coding-task designer. Propose ONE realistic coding task for this project.
Requirements: realistic, requires cross-file understanding, clear acceptance criteria.
Return ONLY valid JSON (no markdown):
{"task_title": "short title", "task_description": "2-3 sentences", "key_files": ["file1"], "acceptance_criteria": ["c1", "c2"]}"""

MULTI_IDEA_PROMPT = """You are a coding-task designer. Propose {count} DISTINCT coding tasks for this project.
Each task must match the type assigned in its spec below. Tasks with the same type must still be distinct tasks.
Requirements: realistic, requires cross-file understanding, clear acceptance criteria.
Return ONLY a valid JSON array (no markdown):
[{{"task_title": "title", "task_description": "desc", "task_type": "type", "key_files": ["f1"], "acceptance_criteria": ["c1"]}}]"""


EXPAND_PROMPT = """Expand this task idea into a full specification.
Rules: prompts start with "You are working in a local project directory. Read the relevant code before changing anything. Do not commit, and do not write tokens or secrets into files."
prompt_qwen/prompt_claude: same task, may differ in phrasing. followups_qwen: 2-3 items. followups_claude: 4-5 items. criteria_descriptions: exactly 7, each starting with "Evaluates whether".
Criteria: user_experience_and_interaction, task_planning_and_execution_control, semantic_understanding_and_logical_reasoning, instruction_compliance_and_constraint_adherence, engineering_quality_and_completeness, delivery_completeness_and_usability, architecture_boundaries_and_security_compliance.
Return ONLY valid JSON (no markdown):
{"prompt_qwen":"...","prompt_claude":"...","followups_qwen":["..."],"followups_claude":["..."],"criteria_descriptions":["Evaluates whether ..."]}"""


def _scan_project(project_path: Path, max_chars: int = 1500) -> str:
    """Build a concise project summary: README excerpt + tree + deps."""
    parts: list[str] = []

    for readme_name in ("README.md", "readme.md", "README.rst", "README"):
        readme_path = project_path / readme_name
        if readme_path.is_file():
            try:
                content = readme_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()[:20]
                parts.append(f"## README\n" + "\n".join(lines) + "\n")
            except Exception:
                pass
            break

    tree_lines: list[str] = []
    _walk_tree(project_path, "", tree_lines, depth=0, max_depth=2, max_lines=40)
    parts.append("## Tree\n" + "\n".join(tree_lines[:40]) + "\n")

    dep_files = ["package.json", "pyproject.toml", "Cargo.toml", "go.mod", "pom.xml",
                 "build.gradle", "Gemfile", "composer.json"]
    for dep_name in dep_files:
        dep_path = project_path / dep_name
        if dep_path.is_file():
            try:
                content = dep_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()[:15]
                parts.append(f"## {dep_name}\n" + "\n".join(lines) + "\n")
            except Exception:
                pass
            break

    result = "\n".join(parts)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n[... truncated ...]"
    return result


def _should_ignore(name: str) -> bool:
    if name in SCAN_IGNORE or name.startswith("."):
        return True
    return any(name.endswith(s) for s in SCAN_IGNORE_SUFFIXES)


def _walk_tree(
    path: Path, prefix: str, lines: list[str],
    depth: int = 0, max_depth: int = 4, max_lines: int = 200,
) -> None:
    if depth > max_depth or len(lines) >= max_lines:
        return
    try:
        entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return

    for entry in entries:
        if _should_ignore(entry.name):
            continue
        if len(lines) >= max_lines:
            return
        lines.append(f"{prefix}{entry.name}{'/' if entry.is_dir() else ''}")
        if entry.is_dir():
            _walk_tree(entry, prefix + "  ", lines, depth + 1, max_depth, max_lines)


async def _call_claude_p(
    prompt: str,
    env: dict[str, str],
    model: str = "",
    timeout: int = 150,
) -> str:
    """Low-level helper: call claude -p and return stdout."""
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "text",
        "--dangerously-skip-permissions",
        "--setting-sources", "local",
        "--bare",
    ]
    if model:
        cmd += ["--model", model]

    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        *cmd, env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"  [claude -p] Timed out after {time.time() - t0:.0f}s (limit={timeout}s), killing...")
        proc.kill()
        await proc.communicate()
        return ""

    elapsed = time.time() - t0
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        out = stdout.decode("utf-8", errors="replace").strip()
        detail = err or out
        print(f"  [claude -p] Exited with {proc.returncode} after {elapsed:.0f}s: {detail[:300]}")

    return stdout.decode("utf-8", errors="replace")


async def _call_gen_idea(
    project_summary: str,
    task_type: str,
    domain: str,
    language: str,
    env: dict[str, str],
    model: str = "",
    timeout: int = 120,
) -> str:
    """Stage 1: generate a lightweight task idea (~100 output tokens)."""
    user_prompt = (
        f"{IDEA_PROMPT}\n\n---\n\n"
        f"## Target\n- Task type: {task_type}\n- Domain: {domain}\n- Language: {language}\n\n"
        f"## Project Summary\n\n{project_summary}\n\n"
        f"Propose ONE task. Return JSON only."
    )
    return await _call_claude_p(user_prompt, env, model=model, timeout=timeout)


async def _call_gen_expand(
    idea_json: str,
    project_summary: str,
    task_type: str,
    domain: str,
    language: str,
    env: dict[str, str],
    model: str = "",
    timeout: int = 180,
) -> str:
    """Stage 2: expand the idea into full prompts/followups/criteria."""
    user_prompt = (
        f"{EXPAND_PROMPT}\n\n---\n\n"
        f"## Target\n- Task type: {task_type}\n- Domain: {domain}\n- Language: {language}\n\n"
        f"## Task Idea\n\n{idea_json}\n\n"
        f"## Project Context (abbreviated)\n\n{project_summary[:2000]}\n\n"
        f"Expand this idea into the full specification. Return JSON only."
    )
    return await _call_claude_p(user_prompt, env, model=model, timeout=timeout)


def _parse_idea_output(raw: str) -> dict | None:
    """Parse stage-1 idea JSON output."""
    cleaned = strip_claude_wrapper(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    if "task_title" not in data or "task_description" not in data:
        return None
    return data


async def _call_gen_multi_ideas(
    project_summary: str,
    task_specs: list[tuple[str, str, str]],
    env: dict[str, str],
    model: str = "",
    timeout: int = 180,
) -> str:
    """Stage 1 batch: generate multiple task ideas in one call.

    task_specs is a list of (task_type, domain, language) triples.
    """
    prompt_text = MULTI_IDEA_PROMPT.format(count=len(task_specs))
    spec_lines = "\n".join(
        f"  {i+1}. type={ttype}, domain={dom}, language={lang}"
        for i, (ttype, dom, lang) in enumerate(task_specs)
    )
    user_prompt = (
        f"{prompt_text}\n\n---\n"
        f"Task specs (generate one idea per line, in order):\n{spec_lines}\n\n"
        f"{project_summary}\n\nReturn JSON array only."
    )
    return await _call_claude_p(user_prompt, env, model=model, timeout=timeout)


def _parse_multi_ideas(raw: str) -> list[dict] | None:
    """Parse stage-1 batch output: a JSON array of idea objects."""
    cleaned = strip_claude_wrapper(raw)

    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\[[\s\S]*\]', cleaned)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None

    if not isinstance(data, list) or len(data) < 1:
        return None

    valid = []
    for item in data:
        if isinstance(item, dict) and "task_title" in item and "task_description" in item:
            valid.append(item)

    return valid if valid else None


def _parse_gen_output(raw: str) -> dict | None:
    cleaned = strip_claude_wrapper(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    required = ["prompt_qwen", "prompt_claude", "followups_qwen",
                 "followups_claude", "criteria_descriptions"]
    if not all(k in data for k in required):
        return None
    if len(data["criteria_descriptions"]) != 7:
        return None
    if not isinstance(data["followups_qwen"], list) or len(data["followups_qwen"]) < 2:
        return None
    if not isinstance(data["followups_claude"], list) or len(data["followups_claude"]) < 3:
        return None

    return data


def _next_task_id(config: BatchConfig) -> str:
    """Find the next available CT-xxxx ID."""
    max_num = 0
    for task in config.tasks:
        match = re.match(r"CT-(\d+)", task.id)
        if match:
            max_num = max(max_num, int(match.group(1)))

    toml_path = config.base_dir / "tasks.toml"
    if toml_path.exists():
        try:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            for t in data.get("task", []):
                match = re.match(r"CT-(\d+)", t.get("id", ""))
                if match:
                    max_num = max(max_num, int(match.group(1)))
        except Exception:
            pass

    return f"CT-{max_num + 1:04d}"


def _write_rubric_templates(
    config: BatchConfig,
    task_id: str,
    descriptions: list[str],
) -> None:
    """Write rubric TOML templates for both qwen and claude."""
    criteria = [
        Criterion(
            name=name,
            description=desc,
            type="likert",
            points=5,
            weight=1.0,
            score=0,
            rationale="",
        )
        for name, desc in zip(CRITERIA_NAMES, descriptions)
    ]

    for model in ("qwen", "claude"):
        dest_dir = config.rubrics_dir / model
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{task_id}.quality.toml"
        write_quality_toml(dest_path, criteria)
        print(f"  [rubric] {dest_path}")


def _escape_toml_ml(s: str) -> str:
    """Escape a string for TOML multi-line basic string."""
    s = s.replace("\\", "\\\\")
    while '"""' in s:
        s = s.replace('"""', '""\\"')
    if s.endswith('"'):
        s = s[:-1] + '\\"'
    return s


def _format_toml_entry(
    task_id: str,
    project_path: str,
    clone_method: str,
    task_type: str,
    domain: str,
    language: str,
    prompt_qwen: str,
    prompt_claude: str,
    followups_qwen: list[str],
    followups_claude: list[str],
) -> str:
    """Format a [[task]] TOML block."""
    def fmt_followups(items: list[str]) -> str:
        lines = []
        for item in items:
            escaped = item.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            lines.append(f'  "{escaped}",')
        return "[\n" + "\n".join(lines) + "\n]"

    return (
        f'\n[[task]]\n'
        f'id = "{task_id}"\n'
        f'project_path = "{project_path}"\n'
        f'clone_method = "{clone_method}"\n'
        f'task_type = "{task_type}"\n'
        f'domain = "{domain}"\n'
        f'language = "{language}"\n'
        f'prompt_qwen = """{_escape_toml_ml(prompt_qwen)}"""\n'
        f'followups_qwen = {fmt_followups(followups_qwen)}\n'
        f'prompt_claude = """{_escape_toml_ml(prompt_claude)}"""\n'
        f'followups_claude = {fmt_followups(followups_claude)}\n'
    )


def _build_gen_env(config: BatchConfig) -> dict[str, str]:
    if not config.claude.auth_token or not config.claude.base_url or not config.claude.model:
        raise ValueError("Claude config is incomplete in .env (need CLAUDE_AUTH_TOKEN, CLAUDE_BASE_URL, CLAUDE_MODEL)")
    return build_claude_env(config.claude)


def _load_used_repos(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return set(data.get("_gen_used_repos", []))
    except Exception:
        return set()


def _save_used_repo(state_path: Path, repo_name: str) -> None:
    data: dict = {}
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    used = data.get("_gen_used_repos", [])
    if repo_name not in used:
        used.append(repo_name)
    data["_gen_used_repos"] = used
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


async def generate_single(
    slot: TaskSlot,
    task_id: str,
    config: BatchConfig,
    env: dict[str, str],
    clone_dir: Path,
    used_repos: set[str],
    state_path: Path,
    from_local: str | None = None,
    dry_run: bool = False,
    toml_lock: asyncio.Lock | None = None,
    total_timeout: int = 900,
    source: str = "github",
) -> bool:
    """Generate a single task from a slot."""
    t_start = time.time()
    print(f"\n[{task_id}] Generating: {slot.task_type} / {slot.domain} / {slot.language}")

    def _elapsed() -> str:
        return f"{time.time() - t_start:.0f}s"

    project_path: Path
    repo_name = ""

    if dry_run:
        if from_local:
            print(f"  [dry-run] Would use local project: {from_local}")
        else:
            print(f"  [dry-run] Would search GitHub for {slot.domain}/{slot.language} projects")
        print(f"  [dry-run] Would generate task: type={slot.task_type}, domain={slot.domain}, lang={slot.language}")
        return True

    if from_local:
        project_path = Path(from_local)
        if not project_path.is_dir():
            print(f"  ERROR: local path not found: {from_local}")
            return False
    else:
        print(f"  [{_elapsed()}] Searching {source} for {slot.domain}/{slot.language} projects...")
        if source == "gitee":
            result = await asyncio.to_thread(
                search_and_clone_gitee, slot.domain, slot.language, task_id, clone_dir,
                used_repos, config.gitee_token, config.http_proxy,
            )
        else:
            result = await asyncio.to_thread(
                search_and_clone, slot.domain, slot.language, task_id, clone_dir,
                used_repos, config.github_token, config.http_proxy,
            )
        if not result:
            print(f"  [{_elapsed()}] {source} search/clone failed")
            return False
        repo, project_path = result
        repo_name = repo.full_name
        used_repos.add(repo_name)
        _save_used_repo(state_path, repo_name)
        print(f"  [{_elapsed()}] Got project: {repo_name}")

    print(f"  [{_elapsed()}] Scanning project: {project_path}")
    summary = _scan_project(project_path)
    print(f"  [{_elapsed()}] Project summary: {len(summary)} chars")

    remaining = total_timeout - (time.time() - t_start)
    if remaining < 30:
        print(f"  [{_elapsed()}] ERROR: not enough time left for AI calls")
        return False

    # Stage 1: generate task idea (lightweight, fast, with retry)
    print(f"  [{_elapsed()}] [stage 1] Generating task idea...")
    idea: dict | None = None
    idea_raw = ""
    stage1_timeout = min(180, int(remaining * 0.4))
    for attempt in range(2):
        if attempt > 0:
            print(f"  [{_elapsed()}] [stage 1 retry] Retrying idea generation...")
        idea_raw = await _call_gen_idea(summary, slot.task_type, slot.domain, slot.language, env, model=config.claude.model, timeout=stage1_timeout)
        if not idea_raw:
            print(f"  [{_elapsed()}] [stage 1] Empty response (timeout or error)")
            continue
        idea = _parse_idea_output(idea_raw)
        if idea is not None:
            break
        print(f"  [{_elapsed()}] [stage 1] Could not parse output")

    if idea is None:
        print(f"  [{_elapsed()}] ERROR: stage 1 failed (timeout or unparseable output)")
        return False

    idea_json = json.dumps(idea, ensure_ascii=False)
    print(f"  [{_elapsed()}] [stage 1] Task idea: {idea.get('task_title', '?')}")

    remaining = total_timeout - (time.time() - t_start)
    if remaining < 30:
        print(f"  [{_elapsed()}] ERROR: not enough time left for stage 2")
        return False

    # Stage 2: expand idea into full prompts/followups (with retry)
    data: dict | None = None
    last_raw = ""
    stage2_timeout = min(240, int(remaining))
    for attempt in range(2):
        label = "retry" if attempt > 0 else "stage 2"
        print(f"  [{_elapsed()}] [{label}] Expanding into full specification...")
        raw = await _call_gen_expand(idea_json, summary, slot.task_type, slot.domain, slot.language, env, model=config.claude.model, timeout=stage2_timeout)
        if not raw:
            print(f"  [{_elapsed()}] [{label}] Empty response (timeout)")
            continue
        last_raw = raw
        data = _parse_gen_output(raw)
        if data is not None:
            break
        print(f"  [{_elapsed()}] [{label}] Could not parse output, {'retrying...' if attempt == 0 else 'giving up'}")

    if data is None:
        draft_path = config.base_dir / f"gen_draft_{task_id}.txt"
        draft_content = f"=== IDEA ===\n{idea_raw}\n\n=== EXPAND (last non-empty) ===\n{last_raw or '(all attempts timed out)'}"
        draft_path.write_text(draft_content, encoding="utf-8")
        print(f"  [{_elapsed()}] ERROR: stage 2 failed after retries, saved to {draft_path.name}")
        return False

    _write_rubric_templates(config, task_id, data["criteria_descriptions"])

    project_path_str = str(project_path).replace("\\", "\\\\")
    toml_entry = _format_toml_entry(
        task_id=task_id,
        project_path=project_path_str,
        clone_method="git",
        task_type=slot.task_type,
        domain=slot.domain,
        language=slot.language,
        prompt_qwen=data["prompt_qwen"],
        prompt_claude=data["prompt_claude"],
        followups_qwen=data["followups_qwen"],
        followups_claude=data["followups_claude"],
    )

    toml_path = config.base_dir / "tasks.toml"
    if toml_lock:
        async with toml_lock:
            with toml_path.open("a", encoding="utf-8") as f:
                f.write(toml_entry)
    else:
        with toml_path.open("a", encoding="utf-8") as f:
            f.write(toml_entry)
    print(f"  [{_elapsed()}] [tasks.toml] Appended {task_id}")

    if repo_name:
        print(f"  Source: {repo_name}")
    print(f"  [{_elapsed()}] OK: {slot.task_type} / {slot.domain} / {slot.language}")
    return True


async def generate_batch(
    slots: list[TaskSlot],
    task_ids: list[str],
    config: BatchConfig,
    env: dict[str, str],
    clone_dir: Path,
    used_repos: set[str],
    state_path: Path,
    from_local: str | None = None,
    dry_run: bool = False,
    toml_lock: asyncio.Lock | None = None,
    repo_lock: asyncio.Lock | None = None,
    api_sem: asyncio.Semaphore | None = None,
    total_timeout: int = 900,
    source: str = "github",
) -> int:
    """Generate multiple tasks from a single project.

    Clones one repo, generates N ideas in a single API call,
    then expands each idea separately. Returns number of successes.
    """
    t_start = time.time()
    n = len(slots)
    id_range = f"{task_ids[0]}..{task_ids[-1]}" if n > 1 else task_ids[0]
    task_types = [s.task_type for s in slots]
    domain = slots[0].domain
    language = slots[0].language

    print(f"\n[{id_range}] Batch generating {n} tasks: {', '.join(task_types)}")
    print(f"  Search: {domain} / {language} (from first slot)")

    def _elapsed() -> str:
        return f"{time.time() - t_start:.0f}s"

    project_path: Path
    repo_name = ""

    if dry_run:
        if from_local:
            print(f"  [dry-run] Would use local project: {from_local}")
        else:
            print(f"  [dry-run] Would search {source} for {domain}/{language} projects")
        print(f"  [dry-run] Would generate {n} tasks: {', '.join(task_types)}")
        return n

    if from_local:
        project_path = Path(from_local)
        if not project_path.is_dir():
            print(f"  ERROR: local path not found: {from_local}")
            return 0
    else:
        print(f"  [{_elapsed()}] Searching {source} for {domain}/{language} projects...")
        lock_ctx = repo_lock if repo_lock else nullcontext()
        async with lock_ctx:
            if source == "gitee":
                result = await asyncio.to_thread(
                    search_and_clone_gitee, domain, language, "_projects", clone_dir,
                    used_repos, config.gitee_token, config.http_proxy,
                )
            else:
                result = await asyncio.to_thread(
                    search_and_clone, domain, language, "_projects", clone_dir,
                    used_repos, config.github_token, config.http_proxy,
                )
            if result:
                repo, project_path = result
                repo_name = repo.full_name
                used_repos.add(repo_name)
                _save_used_repo(state_path, repo_name)
        if not from_local and not repo_name:
            print(f"  [{_elapsed()}] {source} search/clone failed")
            return 0
        print(f"  [{_elapsed()}] Got project: {repo_name}")

    print(f"  [{_elapsed()}] Scanning project: {project_path}")
    summary = _scan_project(project_path)
    print(f"  [{_elapsed()}] Project summary: {len(summary)} chars")

    remaining = total_timeout - (time.time() - t_start)
    if remaining < 60:
        print(f"  [{_elapsed()}] ERROR: not enough time left")
        return 0

    # Stage 1: generate N ideas in one API call
    print(f"  [{_elapsed()}] [stage 1] Generating {n} task ideas in batch...")
    ideas: list[dict] | None = None
    ideas_raw = ""
    stage1_timeout = min(240, int(remaining * 0.3))
    for attempt in range(2):
        if attempt > 0:
            print(f"  [{_elapsed()}] [stage 1 retry] Retrying batch idea generation...")
        sem_ctx = api_sem if api_sem else nullcontext()
        async with sem_ctx:
            ideas_raw = await _call_gen_multi_ideas(
                summary, [(s.task_type, s.domain, s.language) for s in slots], env,
                model=config.claude.model, timeout=stage1_timeout,
            )
        if not ideas_raw:
            print(f"  [{_elapsed()}] [stage 1] Empty response")
            continue
        ideas = _parse_multi_ideas(ideas_raw)
        if ideas is not None and len(ideas) >= n:
            break
        if ideas is not None and len(ideas) < n and attempt == 0:
            print(f"  [{_elapsed()}] [stage 1] Got {len(ideas)}/{n} ideas, retrying for full count...")
            continue
        if ideas is not None:
            break
        print(f"  [{_elapsed()}] [stage 1] Could not parse output")

    if ideas is None:
        print(f"  [{_elapsed()}] ERROR: batch stage 1 failed")
        return 0

    # Fill gaps with single-idea generation if batch returned fewer than expected
    if len(ideas) < n:
        print(f"  [{_elapsed()}] [stage 1] Only {len(ideas)}/{n} ideas, filling gaps individually...")
        for gap_idx in range(len(ideas), n):
            remaining_time = total_timeout - (time.time() - t_start)
            if remaining_time < 30:
                break
            gap_slot = slots[gap_idx]
            sem_ctx = api_sem if api_sem else nullcontext()
            async with sem_ctx:
                gap_raw = await _call_gen_idea(
                    summary, gap_slot.task_type, gap_slot.domain, gap_slot.language, env,
                    model=config.claude.model, timeout=min(120, int(remaining_time * 0.3)),
                )
            if gap_raw:
                gap_idea = _parse_idea_output(gap_raw)
                if gap_idea:
                    gap_idea.setdefault("task_type", gap_slot.task_type)
                    ideas.append(gap_idea)
                    print(f"  [{_elapsed()}] [fill] Got idea: {gap_idea.get('task_title', '?')}")

    print(f"  [{_elapsed()}] [stage 1] Got {len(ideas)} ideas")
    for i, idea in enumerate(ideas):
        print(f"    {i+1}. {idea.get('task_title', '?')}")

    # Reorder ideas to match slot task_types by greedy matching
    if len(ideas) >= n:
        reordered: list[dict | None] = [None] * n
        used_indices: set[int] = set()
        for j in range(n):
            target_type = slots[j].task_type
            for k, idea in enumerate(ideas[:n]):
                if k not in used_indices and idea.get("task_type") == target_type:
                    reordered[j] = idea
                    used_indices.add(k)
                    break
        unmatched = [ideas[k] for k in range(n) if k not in used_indices]
        for j in range(n):
            if reordered[j] is None and unmatched:
                idea = unmatched.pop(0)
                print(f"  [warn] Idea '{idea.get('task_title', '?')}' (type={idea.get('task_type', '?')}) assigned to slot type={slots[j].task_type}")
                reordered[j] = idea
        ideas = [x for x in reordered if x is not None]

    # Stage 2: expand ideas concurrently
    actual_count = min(len(ideas), n)
    remaining_for_stage2 = total_timeout - (time.time() - t_start)
    if remaining_for_stage2 < 30:
        print(f"  [{_elapsed()}] ERROR: not enough time left for stage 2")
        return 0
    stage2_timeout = min(240, int(remaining_for_stage2))

    async def _expand_one(idx: int) -> bool:
        idea = ideas[idx]
        task_id = task_ids[idx]
        s = slots[idx]
        idea_json = json.dumps(idea, ensure_ascii=False)

        print(f"  [{_elapsed()}] [{task_id}] Expanding: {idea.get('task_title', '?')}...")
        data: dict | None = None
        last_raw = ""
        for attempt in range(2):
            sem_ctx = api_sem if api_sem else nullcontext()
            async with sem_ctx:
                raw = await _call_gen_expand(
                    idea_json, summary, s.task_type, s.domain, s.language, env,
                    model=config.claude.model, timeout=stage2_timeout,
                )
            if not raw:
                continue
            last_raw = raw
            data = _parse_gen_output(raw)
            if data is not None:
                break

        if data is None:
            draft_path = config.base_dir / f"gen_draft_{task_id}.txt"
            draft_content = f"=== IDEA ===\n{idea_json}\n\n=== EXPAND ===\n{last_raw or '(empty)'}"
            draft_path.write_text(draft_content, encoding="utf-8")
            print(f"  [{_elapsed()}] [{task_id}] Stage 2 failed, saved draft")
            return False

        _write_rubric_templates(config, task_id, data["criteria_descriptions"])

        project_path_str = str(project_path).replace("\\", "\\\\")
        toml_entry = _format_toml_entry(
            task_id=task_id,
            project_path=project_path_str,
            clone_method="git",
            task_type=s.task_type,
            domain=s.domain,
            language=s.language,
            prompt_qwen=data["prompt_qwen"],
            prompt_claude=data["prompt_claude"],
            followups_qwen=data["followups_qwen"],
            followups_claude=data["followups_claude"],
        )

        toml_path = config.base_dir / "tasks.toml"
        if toml_lock:
            async with toml_lock:
                with toml_path.open("a", encoding="utf-8") as f:
                    f.write(toml_entry)
        else:
            with toml_path.open("a", encoding="utf-8") as f:
                f.write(toml_entry)
        print(f"  [{_elapsed()}] [{task_id}] OK: {s.task_type}")
        return True

    results = await asyncio.gather(
        *[_expand_one(idx) for idx in range(actual_count)],
        return_exceptions=True,
    )
    successes = sum(1 for r in results if r is True)

    if repo_name:
        print(f"  Source: {repo_name}")
    print(f"  [{_elapsed()}] Batch done: {successes}/{actual_count} succeeded")
    return successes


async def generate(
    config: BatchConfig,
    count: int,
    domain: str | None = None,
    language: str | None = None,
    task_type: str | None = None,
    from_local: str | None = None,
    clone_dir: Path | None = None,
    dry_run: bool = False,
    total_timeout: int = 900,
    per_project: int = 1,
    source: str = "github",
) -> int:
    """Main generation entry point."""
    state_path = config.delivery_dir / "pipeline_state.json"
    used_repos = _load_used_repos(state_path)

    if per_project > 1:
        num_projects = -(-count // per_project)  # ceil division
        sampled = sample_slots(num_projects, domain=domain, language=language, task_type=task_type)
        label = f"{len(sampled)} project slots (each expands to {per_project} tasks)"
    else:
        sampled = sample_slots(count, domain=domain, language=language, task_type=task_type)
        label = f"{len(sampled)} task slots"

    if not sampled:
        print(f"ERROR: no matching slots in distribution table")
        return 1

    print(f"Sampled {label} from distribution")
    for i, (idx, slot) in enumerate(sampled, 1):
        print(f"  {i}. [{idx}] {slot.task_type} / {slot.domain} / {slot.language} (w={slot.weight})")

    env = _build_gen_env(config)
    effective_clone_dir = clone_dir or config.runs_root

    base_url = config.claude.base_url
    model = config.claude.model
    proxy = env.get("HTTPS_PROXY", "") or env.get("HTTP_PROXY", "")
    has_key = bool(config.claude.auth_token)
    print(f"\nConfig: model={model}, base_url={base_url[:60]}{'...' if len(base_url) > 60 else ''}")
    print(f"  auth_token={'set' if has_key else 'MISSING'}, proxy={proxy or 'none'}, no_proxy={env.get('NO_PROXY', 'none')}")
    print(f"  clone_dir={effective_clone_dir}, total_timeout={total_timeout}s, per_project={per_project}")

    start_id = _next_task_id(config)
    m = re.match(r"CT-(\d+)", start_id)
    if not m:
        print(f"ERROR: invalid task ID format: {start_id}")
        return 1
    start_num = int(m.group(1))

    toml_lock = asyncio.Lock()
    repo_lock = asyncio.Lock()
    api_sem = asyncio.Semaphore(config.max_parallel)
    total_success = 0
    total_count = 0
    task_idx = 0

    if per_project > 1:
        print(f"\nExpanding {len(sampled)} slots into {per_project} tasks each (max {config.max_parallel} concurrent API calls)")

        async def _run_batch(batch_idx: int, slot: TaskSlot) -> tuple[int, int]:
            batch_slots = expand_slot_for_batch(slot, per_project)
            batch_size = len(batch_slots)
            offset = batch_idx * per_project
            batch_ids = [f"CT-{start_num + offset + i:04d}" for i in range(batch_size)]
            successes = await generate_batch(
                batch_slots, batch_ids, config, env, effective_clone_dir,
                used_repos, state_path, from_local, dry_run,
                toml_lock=toml_lock, repo_lock=repo_lock,
                api_sem=api_sem, total_timeout=total_timeout,
                source=source,
            )
            return successes, batch_size

        results = await asyncio.gather(
            *[_run_batch(i, slot) for i, (_, slot) in enumerate(sampled)],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                print(f"  ERROR: batch failed with exception: {r}")
                total_count += per_project
            else:
                total_success += r[0]
                total_count += r[1]
    else:
        total_count = len(sampled)
        for i, (_, slot) in enumerate(sampled):
            task_id = f"CT-{start_num + i:04d}"
            ok = await generate_single(
                slot, task_id, config, env, effective_clone_dir,
                used_repos, state_path, from_local, dry_run,
                toml_lock=toml_lock, total_timeout=total_timeout,
                source=source,
            )
            if ok:
                total_success += 1

    failed = total_count - total_success
    print(f"\n{'=' * 60}")
    print(f"Generation complete: {total_success} succeeded, {failed} failed, {total_count} total")
    return 0 if failed == 0 else 1
