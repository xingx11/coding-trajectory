"""gen subcommand: auto-generate tasks from GitHub projects."""

from __future__ import annotations

import asyncio
import json
import re
import time
import tomllib
from contextlib import nullcontext
from pathlib import Path

import threading

from ctpipe import strip_claude_wrapper
from ctpipe.config import (
    REFERENCE_CRITERION_DESCRIPTIONS,
    REFERENCE_CRITERION_NAMES,
    BatchConfig,
    build_reference_dimension_table,
    build_validated_env,
)
from ctpipe.distribution import DISTRIBUTION, TaskSlot, expand_slot_for_batch, sample_slots

from ctpipe.github_search import search_and_clone, search_and_clone_gitee, clone_project, GitHubRepo
from ctpipe.rescore import parse_dimension_output
from ctpipe.toml_utils import Criterion, escape_toml_basic, escape_toml_multiline, write_quality_toml
from ctpipe.project_scan import scan_project

# Default timeouts for gen stages (seconds)
_GEN_TOTAL_TIMEOUT = 900
_GEN_STEP_TIMEOUT = 180
# Criteria customization gets a dedicated budget so it is never starved by
# stage1/stage2 having already consumed the shared per-batch total_timeout.
_GEN_CRITERIA_TIMEOUT = 180
# Backoff (seconds) before retrying a stage after an empty/timeout response,
# to avoid hammering a slow upstream endpoint on immediate retry.
_GEN_RETRY_BACKOFF = 4

# Process-internal lock fallback when filelock is not installed
_repo_lock = threading.Lock()

IDEA_PROMPT = """You are a coding-task designer. Propose ONE realistic coding task for this project.
Requirements: realistic, requires cross-file understanding, clear acceptance criteria.
IMPORTANT: task_title and task_description MUST be written in Chinese (简体中文).
Return ONLY valid JSON (no markdown):
{"task_title": "简短中文标题", "task_description": "2-3句中文描述", "key_files": ["file1"], "acceptance_criteria": ["c1", "c2"]}"""

MULTI_IDEA_PROMPT = """You are a coding-task designer. Propose {count} DISTINCT coding tasks for this project.
Each task must match the type assigned in its spec below. Tasks with the same type must still be distinct tasks.
Requirements: realistic, requires cross-file understanding, clear acceptance criteria.
IMPORTANT: task_title and task_description MUST be written in Chinese (简体中文).
Return ONLY a valid JSON array (no markdown):
[{{"task_title": "中文标题", "task_description": "中文描述", "task_type": "type", "key_files": ["f1"], "acceptance_criteria": ["c1"]}}]"""


EXPAND_PROMPT = """Expand this task idea into a full specification.

CRITICAL — Language & style rules for prompt, followups_qwen, followups_claude:
1. ALL prompts and followups MUST be written in Chinese (简体中文).
2. Write them as a real human developer would type into an AI coding assistant — short, casual, natural.
3. The initial prompt should be ONE short sentence describing the overall goal, like a developer's first message. Examples: "帮我给这个项目加一个命令行状态查看功能", "这个项目有个并发 bug，帮我修一下". Do NOT include long requirement lists in the initial prompt.
4. followups are progressive refinements — each one asks for ONE specific next step, building on what the previous prompt/followup asked for. They should read like a natural conversation: "现在加上颜色显示", "再写几个测试", "处理一下边界情况". Keep each followup to 1-2 short sentences max.
5. The prompt field is shared by BOTH models (qwen and claude receive the EXACT same first message). followups_qwen and followups_claude must have the SAME number of items (3-4 items each).
6. Do NOT start prompts with boilerplate like "你正在一个本地项目目录中工作" or any formulaic prefix. Just state the request directly.
7. The progressive flow should be: initial prompt = broad goal → followup 1 = core implementation detail → followup 2 = enhancement or edge case → followup 3+ = testing, polish, docs.
8. followups for both models must cover the same functional areas in the same order. The phrasing may differ but the substantive requirements must be equivalent. Do NOT give one model easier or more specific sub-tasks.
9. Do NOT reference files, functions, or technical details in prompts that do not actually exist in the project. If a prompt mentions a specific file path or class name, it must be verifiable in the codebase.

Return ONLY valid JSON (no markdown):
{"prompt":"...","followups_qwen":["..."],"followups_claude":["..."]}"""


async def _call_claude_p(
    prompt: str,
    env: dict[str, str],
    model: str = "",
    timeout: int = 150,
) -> str:
    """Low-level helper: call claude -p and return stdout.

    Passes prompt via stdin to avoid exposing content in process listings.
    """
    cmd = [
        "claude", "-p",
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
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")), timeout=timeout,
        )
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
    timeout: int = _GEN_STEP_TIMEOUT,
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


def _try_parse_json(raw: str, expect_array: bool = False):
    """Parse JSON from AI output with regex fallback."""
    cleaned = strip_claude_wrapper(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pattern = r'\[[\s\S]*\]' if expect_array else r'\{[\s\S]*\}'
        match = re.search(pattern, cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _parse_idea_output(raw: str) -> dict | None:
    """Parse stage-1 idea JSON output."""
    data = _try_parse_json(raw)
    if not isinstance(data, dict):
        return None
    if "task_title" not in data or "task_description" not in data:
        return None
    return data


async def _call_gen_multi_ideas(
    project_summary: str,
    task_specs: list[tuple[str, str, str]],
    env: dict[str, str],
    model: str = "",
    timeout: int = _GEN_STEP_TIMEOUT,
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
    data = _try_parse_json(raw, expect_array=True)

    if not isinstance(data, list) or len(data) < 1:
        return None

    valid = []
    for item in data:
        if isinstance(item, dict) and "task_title" in item and "task_description" in item:
            valid.append(item)

    return valid if valid else None


def _parse_gen_output(raw: str) -> dict | None:
    data = _try_parse_json(raw)
    if not isinstance(data, dict):
        return None

    # New format: single "prompt" field
    if "prompt" in data:
        required = ["prompt", "followups_qwen", "followups_claude"]
        if not all(k in data for k in required):
            return None
        if not isinstance(data["followups_qwen"], list) or len(data["followups_qwen"]) < 2:
            return None
        if not isinstance(data["followups_claude"], list) or len(data["followups_claude"]) < 3:
            return None
        return data

    # Backward compat: old format with prompt_qwen/prompt_claude
    if "prompt_qwen" in data:
        required = ["prompt_qwen", "prompt_claude", "followups_qwen",
                     "followups_claude"]
        if not all(k in data for k in required):
            return None
        if not isinstance(data["followups_qwen"], list) or len(data["followups_qwen"]) < 2:
            return None
        if not isinstance(data["followups_claude"], list) or len(data["followups_claude"]) < 3:
            return None
        # Merge into single prompt (use prompt_qwen as canonical)
        data["prompt"] = data.pop("prompt_qwen")
        data.pop("prompt_claude", None)
        return data

    return None


def _next_task_id(config: BatchConfig) -> str:
    """Find the next available CT-xxxx ID.

    Checks config.tasks, tasks.toml, and the delivery task manifest
    to avoid ID collisions.
    """
    from ctpipe.config import load_task_manifest

    max_num = 0
    for task in config.tasks:
        m = re.match(r"CT-(\d+)", task.id)
        if m:
            max_num = max(max_num, int(m.group(1)))

    toml_path = config.base_dir / "tasks.toml"
    if toml_path.exists():
        try:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            for t in data.get("task", []):
                m = re.match(r"CT-(\d+)", t.get("id", ""))
                if m:
                    max_num = max(max_num, int(m.group(1)))
        except Exception as exc:
            print(f"  WARNING: could not parse tasks.toml for ID scan: {exc}")

    manifest_tasks = load_task_manifest(config.task_manifest_path)
    for task in manifest_tasks:
        m = re.match(r"CT-(\d+)", task.id)
        if m:
            max_num = max(max_num, int(m.group(1)))

    return f"CT-{max_num + 1:04d}"


def _write_rubric_templates(
    config: BatchConfig,
    task_id: str,
    custom_criteria: list[Criterion] | None = None,
) -> None:
    """Write rubric TOML templates for both qwen and claude.

    Args:
        custom_criteria: If provided, use these customized criteria (with
            project-specific descriptions). Otherwise falls back to the
            first 7 candidate dimensions from config.
    """
    if custom_criteria:
        criteria = custom_criteria
    else:
        print(f"  [{task_id}] WARNING: using generic reference descriptions (AI customization failed)")
        default_names = REFERENCE_CRITERION_NAMES[:7]
        criteria = [
            Criterion(
                name=name,
                description=REFERENCE_CRITERION_DESCRIPTIONS[name],
                type="likert",
                points=5,
                weight=1.0,
                score=0,
                rationale="",
            )
            for name in default_names
        ]

    for model in ("qwen", "claude"):
        dest_dir = config.rubrics_dir / model
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{task_id}.quality.toml"
        write_quality_toml(dest_path, criteria)
        print(f"  [rubric] {dest_path}")


async def _gen_customize_criteria(
    project_summary: str,
    idea: dict,
    task_type: str,
    domain: str,
    language: str,
    env: dict[str, str],
    model: str = "",
    timeout: int = _GEN_STEP_TIMEOUT,
    project_name: str = "",
) -> list[Criterion] | None:
    """Generate customized criteria during gen stage (no trajectory needed).

    Uses project summary + task idea to build a prompt for AI dimension
    selection and description customization.  Returns criteria list on
    success, None on failure (caller falls back to generic template).
    """
    candidates = build_reference_dimension_table()
    project_name = project_name or "unknown"
    task_title = idea.get("task_title", "")
    task_desc = idea.get("task_description", "")

    prompt = f"""你是评分维度设计师。根据以下项目信息和任务描述，从 20 个参考维度中选择 7-10 个最适合评价本次任务完成质量的维度，并为每个维度写定制化的 description。

## 项目信息

- 项目名称：{project_name}
- 任务类型：{task_type}
- 应用领域：{domain}
- 编程语言：{language}
- 任务标题：{task_title}
- 任务描述：{task_desc}

## 项目技术概要

{project_summary[:3000]}

## 20 个参考维度

{candidates}

## 维度选择规则

1. 从 20 个参考维度中选 7-10 个最能评价本次任务完成质量的维度
2. 必须基于任务描述中真实出现的任务特征选择
3. 如果某个维度与本任务无关，不要选择它
4. 优先选择与核心目标、验收标准和最终可用性最相关的维度
5. 与架构边界、安全合规相关的维度设 weight = 2.0，其余设 1.0
6. 维度 name 必须是定制化的英文 snake_case 标识名（如 `cirq_export_data_flow_comprehension`），体现项目名/技术栈/具体操作，不要使用通用维度名
7. 每个维度必须保持原子性：一个维度只评一件事，不能同时要求 A 和 B。错误示例："该维度评价模型是否意识到 hooks.js 是 hooks.ts 的编译产物，并在修改源码后同步修改编译产物"——这混合了两个独立能力，必须拆分
8. 所选维度之间不能有实质性重叠。如果两个维度评价的是同一类能力的不同说法，只保留更精确的一个

## description 定制规则（极其重要）

每个选中维度的 description 必须满足：

1. **保留 1-5 分档位结构**：必须包含 1 分、2 分、3 分、4 分、5 分各档位的具体定义
2. **融入项目特征**：把项目名称（{project_name}）、技术栈（{language}）、具体任务（{task_title}）融入到各档位的描述中
3. 不能是通用模板，必须让人一看就知道这是针对什么项目什么任务的评分标准
4. 使用中文，写成一行字符串
5. 每个 description 的各档位定义中，每档只描述一个判断条件。禁止在某一档中用"并且/且/同时"连接多个独立条件

### 定制化示例

通用模板（不合格）：
"模型理解意图并进行逻辑推理的准确性如何？1分：完全误解意图...5分：精准整合上下文..."

定制化（合格）：
"在 turbulenz_engine 输入设备键盘事件重复触发修复任务中，模型对 onFocusIn/onFocusOut 事件注册链路的理解和修复逻辑推理是否准确？1分：完全误解键盘事件重复触发的根因，把问题归到无关模块。2分：定位到 inputapp 但修复方案不对，函数引用不一致导致 removeEventListener 无效。3分：理解主链路但遗漏鼠标/触摸事件的类似问题。4分：正确定位并修复键盘事件链路，函数引用一致。5分：精准定位 inputapp.ts 和 inputdevice.ts 的事件注册逻辑，用最小改动修复且覆盖所有输入类型"

## 输出格式

只输出合法 TOML，不要包含 Markdown 代码块、标题或解释。
所有字符串字段使用普通双引号 `"..."`，不使用 TOML 多行字符串。
description 内容中禁止出现英文双引号 `"`，如需引用请使用中文引号""。
score 设为 0，rationale 设为空字符串。

[[criterion]]
name = "维度名称"
description = "定制化的中文评分标准（包含1-5分档位定义，融入项目特征）"
type = "likert"
points = 5
weight = 1.0
score = 0
rationale = ""
"""

    for attempt in range(2):
        raw = await _call_claude_p(prompt, env, model=model, timeout=timeout)
        if not raw:
            continue
        parsed, reason = parse_dimension_output(raw)
        if parsed:
            print(f"  [criteria] Customized {len(parsed)} dimensions for this task")
            return parsed
        if attempt == 0:
            print(f"  [criteria] Customize attempt 1 failed: {reason}, retrying...")

    print(f"  [criteria] Customization failed, falling back to generic template")
    return None


def _format_toml_entry(
    task_id: str,
    project_path: str,
    clone_method: str,
    task_type: str,
    domain: str,
    language: str,
    prompt: str,
    followups_qwen: list[str],
    followups_claude: list[str],
    task_title: str = "",
    task_description: str = "",
    acceptance_criteria: list[str] | None = None,
) -> str:
    """Format a [[task]] TOML block with properly escaped values."""
    def fmt_followups(items: list[str]) -> str:
        lines = []
        for item in items:
            lines.append(f'  "{escape_toml_basic(item)}",')
        return "[\n" + "\n".join(lines) + "\n]"

    parts = [
        f'\n[[task]]\n'
        f'id = "{escape_toml_basic(task_id)}"\n'
        f'project_path = "{escape_toml_basic(project_path)}"\n'
        f'clone_method = "{escape_toml_basic(clone_method)}"\n'
        f'task_type = "{escape_toml_basic(task_type)}"\n'
        f'domain = "{escape_toml_basic(domain)}"\n'
        f'language = "{escape_toml_basic(language)}"\n'
    ]

    if task_title:
        parts.append(f'task_title = "{escape_toml_basic(task_title)}"\n')
    if task_description:
        parts.append(f'task_description = """{escape_toml_multiline(task_description)}"""\n')
    if acceptance_criteria:
        parts.append(f'acceptance_criteria = {fmt_followups(acceptance_criteria)}\n')

    parts.append(
        f'prompt = """{escape_toml_multiline(prompt)}"""\n'
        f'followups_qwen = {fmt_followups(followups_qwen)}\n'
        f'followups_claude = {fmt_followups(followups_claude)}\n'
    )

    return "".join(parts)


def _load_used_repos(state_path: Path, base_dir: Path | None = None) -> set[str]:
    repos: set[str] = set()
    # Source 1: current delivery state
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            repos.update(data.get("_gen_used_repos", []))
        except Exception as exc:
            print(f"  WARNING: could not read used repos from state file: {exc}")
    # Source 2: all other delivery_*/pipeline_state.json (cross-delivery dedup)
    if base_dir:
        for ps in sorted(base_dir.glob("delivery_*/pipeline_state.json")):
            if ps == state_path:
                continue
            try:
                d = json.loads(ps.read_text(encoding="utf-8"))
                found = d.get("_gen_used_repos", [])
                repos.update(found)
            except Exception:
                pass
    if repos and base_dir:
        print(f"  Loaded {len(repos)} used repos (cross-delivery dedup)")
    return repos


def _save_used_repo(state_path: Path, repo_name: str) -> None:
    try:
        import filelock as _filelock_mod
        lock_path = state_path.with_suffix(".lock")
        lock = _filelock_mod.FileLock(lock_path, timeout=10)
    except ImportError:
        lock = None

    def _do_save() -> None:
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
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(state_path)

    if lock is not None:
        with lock:
            _do_save()
    else:
        with _repo_lock:
            _do_save()


def _clone_from_pool(
    repo_name: str,
    dest_root: Path,
    task_id: str,
    http_proxy: str = "",
    source: str = "github",
) -> tuple[GitHubRepo, Path] | None:
    """Clone a repo from the pool by its full_name."""
    # Determine clone URL from full_name and source
    if source == "gitee":
        clone_url = f"https://gitee.com/{repo_name}.git"
    else:
        clone_url = f"https://github.com/{repo_name}.git"

    repo = GitHubRepo(
        full_name=repo_name,
        clone_url=clone_url,
        description="",
        language="",
        stars=0,
        updated_at="",
    )
    path = clone_project(repo, dest_root, task_id, http_proxy=http_proxy)
    if path:
        return repo, path
    return None


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
    total_timeout: int = _GEN_TOTAL_TIMEOUT,
    source: str = "github",
    pool_assignments: dict[str, list[str]] | None = None,
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
        result = None
        # Try pool first
        if pool_assignments:
            pool_key = f"{slot.domain}:{slot.language}"
            available = [r for r in pool_assignments.get(pool_key, []) if r not in used_repos]
            if available:
                pick = available[0]
                print(f"  [{_elapsed()}] Cloning from pool: {pick}")
                result = await asyncio.to_thread(
                    _clone_from_pool, pick, clone_dir, task_id, config.http_proxy, source,
                )
                if not result:
                    print(f"  [{_elapsed()}] Pool clone failed, falling back to search")
            else:
                print(f"  [{_elapsed()}] Pool exhausted for {pool_key}, falling back to search")

        # Fallback to search
        if not result:
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
    summary = scan_project(project_path)
    print(f"  [{_elapsed()}] Project summary: {len(summary)} chars")

    remaining = total_timeout - (time.time() - t_start)
    if remaining < 30:
        print(f"  [{_elapsed()}] ERROR: not enough time left for AI calls")
        return False

    # Stage 1: generate task idea (lightweight, fast, with retry)
    print(f"  [{_elapsed()}] [stage 1] Generating task idea...")
    idea: dict | None = None
    idea_raw = ""
    stage1_timeout = min(_GEN_STEP_TIMEOUT,int(remaining * 0.4))
    for attempt in range(2):
        if attempt > 0:
            await asyncio.sleep(_GEN_RETRY_BACKOFF)
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

    # Dedicated timeout: criteria must not inherit the (possibly exhausted)
    # remaining budget, otherwise it gets floored to ~30s and always times out.
    custom = await _gen_customize_criteria(
        summary, idea, slot.task_type, slot.domain, slot.language,
        env, model=config.claude.model, timeout=_GEN_CRITERIA_TIMEOUT,
        project_name=project_path.name,
    )
    _write_rubric_templates(config, task_id, custom_criteria=custom)

    project_path_str = str(project_path).replace("\\", "\\\\")
    toml_entry = _format_toml_entry(
        task_id=task_id,
        project_path=project_path_str,
        clone_method="git",
        task_type=slot.task_type,
        domain=slot.domain,
        language=slot.language,
        prompt=data["prompt"],
        followups_qwen=data["followups_qwen"],
        followups_claude=data["followups_claude"],
        task_title=idea.get("task_title", ""),
        task_description=idea.get("task_description", ""),
        acceptance_criteria=idea.get("acceptance_criteria", []),
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
    total_timeout: int = _GEN_TOTAL_TIMEOUT,
    source: str = "github",
    pool_assignments: dict[str, list[str]] | None = None,
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
        result = None
        lock_ctx = repo_lock if repo_lock else nullcontext()
        async with lock_ctx:
            # Try pool first
            if pool_assignments:
                pool_key = f"{domain}:{language}"
                available = [r for r in pool_assignments.get(pool_key, []) if r not in used_repos]
                if available:
                    pick = available[0]
                    print(f"  [{_elapsed()}] Cloning from pool: {pick}")
                    result = await asyncio.to_thread(
                        _clone_from_pool, pick, clone_dir, "_projects", config.http_proxy, source,
                    )
                    if not result:
                        print(f"  [{_elapsed()}] Pool clone failed, falling back to search")
                else:
                    print(f"  [{_elapsed()}] Pool exhausted for {pool_key}, falling back to search")

            # Fallback to search
            if not result:
                print(f"  [{_elapsed()}] Searching {source} for {domain}/{language} projects...")
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
    summary = scan_project(project_path)
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
            await asyncio.sleep(_GEN_RETRY_BACKOFF)
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

        # Dedicated timeout + respect the concurrency limit. Previously this
        # used the leftover shared budget (floored to 30s -> guaranteed timeout
        # -> generic template) and bypassed the semaphore (all tasks' criteria
        # fired at once, overloading a slow endpoint).
        sem_ctx = api_sem if api_sem else nullcontext()
        async with sem_ctx:
            custom = await _gen_customize_criteria(
                summary, idea, s.task_type, s.domain, s.language,
                env, model=config.claude.model, timeout=_GEN_CRITERIA_TIMEOUT,
                project_name=project_path.name,
            )
        _write_rubric_templates(config, task_id, custom_criteria=custom)

        project_path_str = str(project_path).replace("\\", "\\\\")
        toml_entry = _format_toml_entry(
            task_id=task_id,
            project_path=project_path_str,
            clone_method="git",
            task_type=s.task_type,
            domain=s.domain,
            language=s.language,
            prompt=data["prompt"],
            followups_qwen=data["followups_qwen"],
            followups_claude=data["followups_claude"],
            task_title=idea.get("task_title", ""),
            task_description=idea.get("task_description", ""),
            acceptance_criteria=idea.get("acceptance_criteria", []),
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


async def _clone_only_run(
    sampled: list[tuple[int, TaskSlot]],
    config: BatchConfig,
    clone_dir: Path,
    used_repos: set[str],
    state_path: Path,
    dry_run: bool = False,
    source: str = "github",
) -> int:
    """Search and clone repos without AI task generation."""
    groups: dict[tuple[str, str], TaskSlot] = {}
    for _, slot in sampled:
        key = (slot.domain, slot.language)
        if key not in groups:
            groups[key] = slot

    print(f"\nClone-only mode: {len(groups)} unique (domain, language) groups")
    print(f"  clone_dir={clone_dir}, source={source}, proxy={config.http_proxy or 'none'}")

    if dry_run:
        print("\n[dry-run] Would search and clone:")
        for i, ((dom, lang), slot) in enumerate(groups.items(), 1):
            print(f"  {i}. {dom} / {lang}")
        return 0

    cloned: list[dict[str, str]] = []
    for (dom, lang), slot in groups.items():
        print(f"\n  Searching {source} for {dom}/{lang} projects...")
        if source == "gitee":
            result = await asyncio.to_thread(
                search_and_clone_gitee, dom, lang, "_projects", clone_dir,
                used_repos, config.gitee_token, config.http_proxy,
            )
        else:
            result = await asyncio.to_thread(
                search_and_clone, dom, lang, "_projects", clone_dir,
                used_repos, config.github_token, config.http_proxy,
            )
        if result:
            repo, project_path = result
            used_repos.add(repo.full_name)
            _save_used_repo(state_path, repo.full_name)
            cloned.append({
                "repo": repo.full_name,
                "path": str(project_path),
                "domain": dom,
                "language": lang,
            })
            print(f"  OK: {repo.full_name} -> {project_path}")
        else:
            print(f"  FAILED: no repo found for {dom}/{lang}")

    if cloned:
        manifest_path = clone_dir / "_cloned_repos.json"
        existing: list[dict[str, str]] = []
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing_paths = {e["path"] for e in existing}
        for entry in cloned:
            if entry["path"] not in existing_paths:
                existing.append(entry)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n{'=' * 60}")
        print(f"Cloned {len(cloned)}/{len(groups)} repos:")
        for i, entry in enumerate(cloned, 1):
            print(f"  {i}. {entry['repo']} -> {entry['path']} ({entry['domain']} / {entry['language']})")
        print(f"\nManifest: {manifest_path}")
        print(f"\nTo generate tasks from a cloned repo:")
        print(f'  python -m ctpipe gen --from-local "{cloned[0]["path"]}" --count 3')
    else:
        print(f"\n{'=' * 60}")
        print("No repos cloned successfully.")

    return 0 if cloned else 1


async def _analyze_local(
    sampled: list[tuple[int, TaskSlot]],
    config: BatchConfig,
    project_path: Path,
    dry_run: bool = False,
    total_timeout: int = _GEN_TOTAL_TIMEOUT,
) -> int:
    """Analyze a local project via Claude Code and write task entries directly."""
    if not project_path.is_dir():
        print(f"ERROR: project path not found: {project_path}")
        return 1

    template_path = config.docs_dir / "analyze_prompt_template.md"
    if not template_path.exists():
        print(f"ERROR: prompt template not found: {template_path}")
        return 1

    template = template_path.read_text(encoding="utf-8")

    start_id = _next_task_id(config)
    m = re.match(r"CT-(\d+)", start_id)
    if not m:
        print(f"ERROR: invalid task ID format: {start_id}")
        return 1
    start_num = int(m.group(1))
    count = len(sampled)
    task_ids = [f"CT-{start_num + i:04d}" for i in range(count)]

    slots_text = "\n".join(
        f"  {i+1}. task_type={slot.task_type}, domain={slot.domain}, language={slot.language}"
        for i, (_, slot) in enumerate(sampled)
    )

    project_path_escaped = str(project_path).replace("\\", "\\\\")
    tasks_toml_path = str(config.base_dir / "tasks.toml").replace("\\", "\\\\")
    rubrics_dir = str(config.rubrics_dir).replace("\\", "\\\\")

    prompt = template
    prompt = prompt.replace("{{COUNT}}", str(count))
    prompt = prompt.replace("{{SLOTS}}", slots_text)
    prompt = prompt.replace("{{PROJECT_PATH_ESCAPED}}", project_path_escaped)
    prompt = prompt.replace("{{TASKS_TOML_PATH}}", tasks_toml_path)
    prompt = prompt.replace("{{RUBRICS_DIR}}", rubrics_dir)
    prompt = prompt.replace("{{TASK_IDS}}", ", ".join(task_ids))

    if dry_run:
        print(f"\n[dry-run] Rendered prompt ({len(prompt)} chars):")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        print(f"\nTask IDs: {', '.join(task_ids)}")
        print(f"Project: {project_path}")
        print(f"tasks.toml: {config.base_dir / 'tasks.toml'}")
        print(f"Rubrics: {config.rubrics_dir}")
        return 0

    env = build_validated_env(config.claude)
    print(f"\n[analyze] Project: {project_path}")
    print(f"[analyze] Task IDs: {', '.join(task_ids)}")
    print(f"[analyze] Calling Claude Code to analyze and generate tasks...")

    result = await _call_claude_p(prompt, env, model=config.claude.model, timeout=total_timeout)

    if not result:
        print("[analyze] ERROR: empty response (timeout or error)")
        return 1

    ok_count = 0
    for tid in task_ids:
        qwen_rubric = config.rubrics_dir / "qwen" / f"{tid}.quality.toml"
        claude_rubric = config.rubrics_dir / "claude" / f"{tid}.quality.toml"
        if qwen_rubric.exists() and claude_rubric.exists():
            ok_count += 1
            print(f"  [{tid}] rubric files OK")
        else:
            print(f"  [{tid}] WARNING: rubric files missing")

    print(f"\n[analyze] Done: {ok_count}/{count} tasks have rubric files")
    print(f"[analyze] Check tasks.toml for the new [[task]] entries")
    return 0


async def generate(
    config: BatchConfig,
    count: int,
    domain: str | None = None,
    language: str | None = None,
    task_type: str | None = None,
    from_local: str | None = None,
    clone_dir: Path | None = None,
    dry_run: bool = False,
    clone_only: bool = False,
    analyze: bool = False,
    total_timeout: int = _GEN_TOTAL_TIMEOUT,
    per_project: int = 1,
    source: str = "github",
) -> int:
    """Main generation entry point."""
    state_path = config.state_path
    used_repos = _load_used_repos(state_path, config.base_dir)

    # Load pool assignments if available
    pool_assignments: dict[str, list[str]] | None = None
    if config.person_id:
        from ctpipe.pool import load_pool_assignments
        pool_assignments = load_pool_assignments(config.base_dir, config.person_id)
        if pool_assignments:
            total_pool = sum(len(v) for v in pool_assignments.values())
            print(f"Loaded pool assignments for person {config.person_id}: {total_pool} repos across {len(pool_assignments)} domain/language pairs")
        else:
            print(f"No pool assignments found for person {config.person_id} (will use search)")

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

    effective_clone_dir = clone_dir or config.runs_root

    if clone_only:
        return await _clone_only_run(sampled, config, effective_clone_dir, used_repos, state_path, dry_run, source)

    if analyze:
        if not from_local:
            print("ERROR: --analyze requires --from-local <path>")
            return 1
        return await _analyze_local(sampled, config, Path(from_local), dry_run, total_timeout)

    env = build_validated_env(config.claude)

    base_url = config.claude.base_url
    model = config.claude.model
    proxy = config.http_proxy
    has_key = bool(config.claude.auth_token)
    print(f"\nConfig: model={model}, base_url={base_url[:60]}{'...' if len(base_url) > 60 else ''}")
    print(f"  auth_token={'set' if has_key else 'MISSING'}, proxy={proxy or 'none'}")
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
                source=source, pool_assignments=pool_assignments,
            )
            return successes, batch_size

        results = await asyncio.gather(
            *[_run_batch(i, slot) for i, (_, slot) in enumerate(sampled)],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                print(f"  ERROR: batch failed with exception: {str(r)[:200]}")
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
                source=source, pool_assignments=pool_assignments,
            )
            if ok:
                total_success += 1

    failed = total_count - total_success
    print(f"\n{'=' * 60}")
    print(f"Generation complete: {total_success} succeeded, {failed} failed, {total_count} total")
    return 0 if failed == 0 else 1
