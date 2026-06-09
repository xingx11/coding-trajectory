"""rescore subcommand: re-score with customized dimensions and descriptions."""

from __future__ import annotations

import asyncio
import re
import tomllib
from pathlib import Path

from ctpipe import strip_claude_wrapper
from ctpipe.config import (
    MAX_CRITERIA_COUNT,
    MAX_SCORING_RETRIES,
    MIN_CRITERIA_COUNT,
    SCORING_TIMEOUT,
    TRAJECTORY_MAX_CHARS,
    TRAJECTORY_SUMMARY_CHARS,
    BatchConfig,
    TaskConfig,
    build_bad_pattern_table,
    build_reference_dimension_table,
    build_validated_env,
    check_passrate_thresholds,
    is_valid_criterion_name,
    select_delivery_tasks,
)
from ctpipe.score import call_scoring_ai, extract_toml_section
from ctpipe.state import PipelineState
from ctpipe.toml_utils import Criterion, calc_passrate, has_score_tiers, write_quality_toml, write_rubric_pair
from ctpipe.trajectory import extract_for_scoring


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def read_task_context(metadata_path: Path) -> dict[str, str]:
    """Extract task context from a metadata .md file."""
    ctx: dict[str, str] = {}
    if not metadata_path.exists():
        return ctx

    text = metadata_path.read_text(encoding="utf-8")

    # Project path
    m = re.search(r"Project path:\s*(.+)", text)
    if m:
        ctx["project_path"] = m.group(1).strip()
        ctx["project_name"] = Path(m.group(1).strip()).name

    # Task type / domain / language
    m = re.search(r"Task type:\s*(.+)", text)
    if m:
        ctx["task_type"] = m.group(1).strip()
    m = re.search(r"Application domain:\s*(.+)", text)
    if m:
        ctx["domain"] = m.group(1).strip()
    m = re.search(r"Language:\s*(.+)", text)
    if m:
        ctx["language"] = m.group(1).strip()

    # Initial prompts
    prompts: list[str] = []
    for block in re.finditer(r"Initial prompt:\s*\n```text\n(.+?)\n```", text, re.DOTALL):
        prompts.append(block.group(1).strip())
    if prompts:
        ctx["prompts"] = " | ".join(prompts)

    # Follow-up summaries
    followups: list[str] = []
    for block in re.finditer(r"Follow-up summary:\s*\n```text\n(.+?)\n```", text, re.DOTALL):
        followups.append(block.group(1).strip())
    if followups:
        ctx["followups"] = " | ".join(followups)

    # Project summary
    m = re.search(r"## Project Summary\s*\n```text\n(.+?)\n```", text, re.DOTALL)
    if m:
        ctx["project_summary"] = m.group(1).strip()

    # Task description fields
    m = re.search(r"- Title:\s*(.+)", text)
    if m:
        ctx["task_title"] = m.group(1).strip()
    m = re.search(r"- Description:\s*(.+)", text)
    if m:
        ctx["task_description"] = m.group(1).strip()
    # Acceptance criteria (list items indented under the heading)
    m = re.search(r"- Acceptance criteria:\n((?:\s+- .+\n?)+)", text)
    if m:
        items = [line.strip().lstrip("- ") for line in m.group(1).strip().splitlines()]
        ctx["acceptance_criteria"] = " | ".join(items)

    return ctx


def build_task_context(task, config) -> dict[str, str]:
    """Build task context from metadata, with fallback to task config."""
    metadata_path = config.delivery_dir / "metadata" / f"{task.id}.md"
    task_ctx = read_task_context(metadata_path)
    if not task_ctx:
        task_ctx = {
            "project_name": task.project_path.name,
            "task_type": task.task_type,
            "domain": task.domain,
            "language": task.language,
            "prompts": task.prompt_qwen,
        }
    if "project_summary" not in task_ctx and task.project_path.is_dir():
        from ctpipe.project_scan import scan_project
        task_ctx["project_summary"] = scan_project(task.project_path)

    # Supplement with structured fields from TaskConfig
    if not task_ctx.get("task_title") and getattr(task, "task_title", ""):
        task_ctx["task_title"] = task.task_title
    if not task_ctx.get("task_description") and getattr(task, "task_description", ""):
        task_ctx["task_description"] = task.task_description
    if not task_ctx.get("acceptance_criteria") and getattr(task, "acceptance_criteria", None):
        task_ctx["acceptance_criteria"] = " | ".join(task.acceptance_criteria)

    return task_ctx


# ---------------------------------------------------------------------------
# Round 1: Dimension selection + description customization
# ---------------------------------------------------------------------------

def build_dimension_prompt(
    task_ctx: dict[str, str],
    qwen_summary: str,
    claude_summary: str,
) -> str:
    """Build prompt for AI to select dimensions and customize descriptions."""
    candidates = build_reference_dimension_table()

    project_name = task_ctx.get("project_name", "unknown")
    task_type = task_ctx.get("task_type", "unknown")
    domain = task_ctx.get("domain", "unknown")
    language = task_ctx.get("language", "unknown")
    prompts = task_ctx.get("prompts", "")
    followups = task_ctx.get("followups", "")
    project_summary = task_ctx.get("project_summary", "")

    project_summary_section = ""
    if project_summary:
        project_summary_section = f"""
## 项目技术概要（README、目录结构、依赖）

{project_summary}
"""

    return f"""你是评分维度设计师。根据以下项目信息和两条轨迹摘要，从 20 个参考维度中选择 7-10 个最能反映本次任务质量差异的维度，并为每个维度写定制化的 description。

## 项目信息

- 项目名称：{project_name}
- 任务类型：{task_type}
- 应用领域：{domain}
- 编程语言：{language}
- 任务描述：{prompts}
- 追问要点：{followups}
{project_summary_section}
## Qwen 轨迹摘要（前 {TRAJECTORY_SUMMARY_CHARS} 字符）

{qwen_summary[:TRAJECTORY_SUMMARY_CHARS]}

## Claude 轨迹摘要（前 {TRAJECTORY_SUMMARY_CHARS} 字符）

{claude_summary[:TRAJECTORY_SUMMARY_CHARS]}

## 20 个参考维度

{candidates}

## 维度选择规则

1. 从 20 个参考维度中选 7-10 个最能区分本次轨迹质量差异的维度
2. 必须基于轨迹中真实出现的任务特征和行为证据选择
3. 如果某个维度没有可观察证据，不要选择它
4. 优先选择与核心目标、失败点、验收标准和最终可用性最相关的维度
5. 与架构边界、安全合规相关的维度设 weight = 2.0，其余设 1.0
6. 维度 name 必须是定制化的英文 snake_case 标识名（如 `cirq_export_data_flow_comprehension`），体现项目名/技术栈/具体操作，不要使用通用维度名

## description 定制规则（极其重要）

每个选中维度的 description 必须满足：

1. **保留 1-5 分档位结构**：必须包含 1 分、2 分、3 分、4 分、5 分各档位的具体定义
2. **融入项目特征**：把项目名称（{project_name}）、技术栈（{language}）、具体问题（{prompts}）融入到各档位的描述中
3. 不能是通用模板，必须让人一看就知道这是针对什么项目什么任务的评分标准
4. 使用中文，写成一行字符串

### 定制化示例

通用模板（不合格）：
"模型理解意图并进行逻辑推理的准确性如何？1分：完全误解意图...5分：精准整合上下文..."

定制化（合格）：
"在 turbulenz_engine 输入设备键盘事件重复触发修复任务中，模型对 onFocusIn/onFocusOut 事件注册链路的理解和修复逻辑推理是否准确？1分：完全误解键盘事件重复触发的根因，把问题归到无关模块...2分：定位到 inputapp 但修复方案不对，函数引用不一致导致 removeEventListener 无效...3分：理解主链路但遗漏鼠标/触摸事件的类似问题...4分：正确定位并修复键盘事件链路，函数引用一致...5分：精准定位 inputapp.ts 和 inputdevice.ts 的事件注册逻辑，用最小改动修复且覆盖所有输入类型"

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


def _fix_toml_description_quotes(text: str) -> str:
    """Fix unescaped double quotes inside description values.

    AI sometimes generates description strings containing bare double quotes
    like: description = "在项目中使用 "某功能" 时..."
    which breaks TOML parsing.  Replace inner quotes with Chinese quotes.
    """
    def _fix_match(m: re.Match) -> str:
        inner = m.group(1)
        fixed = inner.replace('"', '\u201c')
        return f'description = "{fixed}"'

    return re.sub(r'description\s*=\s*"(.*)"', _fix_match, text)


def parse_dimension_output(raw: str) -> tuple[list[Criterion] | None, str]:
    """Parse AI-generated dimension TOML into Criterion list.

    Returns (criteria, error_reason). error_reason is empty on success.
    """
    cleaned = extract_toml_section(raw)

    try:
        data = tomllib.loads(cleaned)
    except Exception:
        # Retry with fixed quotes
        try:
            data = tomllib.loads(_fix_toml_description_quotes(cleaned))
        except Exception as e:
            return None, f"TOML parse error: {e}"

    items = data.get("criterion", [])
    if not (MIN_CRITERIA_COUNT <= len(items) <= MAX_CRITERIA_COUNT):
        return None, f"criterion count {len(items)} not in [{MIN_CRITERIA_COUNT}, {MAX_CRITERIA_COUNT}]"

    seen: set[str] = set()
    result: list[Criterion] = []

    for item in items:
        name = item.get("name", "")
        if not is_valid_criterion_name(name):
            return None, f"invalid criterion name: {name!r}"
        if name in seen:
            return None, f"duplicate criterion: {name}"
        seen.add(name)

        desc = item.get("description", "")
        if not desc or len(desc) < 20:
            return None, f"description too short for {name}: {len(desc)} chars"
        if not has_score_tiers(desc):
            return None, f"missing 1-5 score tiers in description for {name}"

        result.append(Criterion(
            name=name,
            description=desc,
            type="likert",
            points=5,
            weight=item.get("weight", 1.0),
            score=0,
            rationale="",
        ))

    return result, ""


# ---------------------------------------------------------------------------
# Round 2: Scoring with customized dimensions
# ---------------------------------------------------------------------------

def _build_rescore_prompt(
    criteria: list[Criterion],
    model_name: str,
    passrate_hint: str,
    task_context: dict[str, str] | None = None,
) -> str:
    """Build scoring prompt with customized dimensions and passrate guidance."""
    bad_patterns = build_bad_pattern_table()

    dim_lines: list[str] = []
    for c in criteria:
        dim_lines.append(f"- `{c.name}` (weight={c.weight}): {c.description}")
    dim_table = "\n".join(dim_lines)

    # Build task context block if available
    context_block = ""
    if task_context:
        ctx_parts: list[str] = []
        if task_context.get("project_name"):
            ctx_parts.append(f"- 项目：{task_context['project_name']}")
        if task_context.get("language"):
            ctx_parts.append(f"- 技术栈：{task_context['language']}")
        if task_context.get("task_type"):
            ctx_parts.append(f"- 任务类型：{task_context['task_type']}")
        if task_context.get("task_title"):
            ctx_parts.append(f"- 任务标题：{task_context['task_title']}")
        if task_context.get("task_description"):
            ctx_parts.append(f"- 任务描述：{task_context['task_description']}")
        if task_context.get("acceptance_criteria"):
            ctx_parts.append(f"- 验收标准：{task_context['acceptance_criteria']}")
        if task_context.get("project_summary"):
            ctx_parts.append(f"- 项目概要：\n{task_context['project_summary']}")
        if ctx_parts:
            context_block = "\n## 任务背景\n\n" + "\n".join(ctx_parts) + "\n"

    return f"""你是评分 AI，负责评价一次 coding-assistant（{model_name}）执行轨迹的质量。
{context_block}
## 评分维度（已定制化，必须完整使用）

你必须使用以下 {len(criteria)} 个评分维度及其定制化描述，不得增加或减少：

{dim_table}

注意：这些 description 已根据当前项目和任务特征量身定制。
你必须在输出的 TOML 中完整保留这些定制化 description，不得替换为通用模板。

## 评分规则

- 分数为 1-5 的整数
- weight 必须使用评分模板中指定的权重值（与架构边界/安全合规相关的维度为 2.0，其余为 1.0）
- description 必须使用上面提供的定制化描述，保持一行字符串，原样复制不得修改

## passrate 约束提示

{passrate_hint}

这是一个软约束——你应该在合理范围内考虑这个目标，但评分必须有真实证据支撑。
如果轨迹质量确实无法支持目标 passrate，按实际情况评分并在 rationale 中说明。

## rationale 写作要求

- 必须使用中文
- 必须引用轨迹中的具体证据：哪些文件被修改、运行了什么命令、出了什么错、遗漏了什么
- 每条 rationale 必须独立且不同，禁止跨维度复制相似句式
- 1-3 句话，简洁具体
- 不要提及 task ID 或与另一个模型做比较

## Bad Pattern 识别

请检查轨迹是否命中以下 Bad Pattern：

{bad_patterns}

如有命中在 TOML 之外单独说明。

## 输出格式

只输出合法 TOML，不要包含 Markdown 代码块、标题或解释文本。
所有字符串字段使用普通双引号 `"..."`，不使用 TOML 多行字符串。
description 和 rationale 都必须写成一行。

[[criterion]]
name = "维度名称"
description = "定制化描述（原样复制）"
type = "likert"
points = 5
weight = 1.0
score = 3
rationale = "中文评分理由"

如果命中 Bad Pattern，在 TOML 之后另起一行写：
Bad Pattern 命中：
- xxx：具体说明
如果未命中，写：
Bad Pattern 命中：未发现明确命中。
"""


def _parse_rescore_output(
    raw: str,
    expected_criteria: list[Criterion],
) -> list[Criterion] | None:
    """Parse scored TOML, preserving customized descriptions from expected_criteria."""
    cleaned = extract_toml_section(raw)

    try:
        data = tomllib.loads(cleaned)
    except Exception:
        return None

    scored = data.get("criterion", [])
    if len(scored) != len(expected_criteria):
        return None

    expected_map = {c.name: c for c in expected_criteria}
    seen: set[str] = set()
    result: list[Criterion] = []

    for item in scored:
        name = item.get("name", "")
        if name not in expected_map or name in seen:
            return None
        seen.add(name)

        raw_score = item.get("score", 0)
        if not isinstance(raw_score, int) or raw_score < 1 or raw_score > 5:
            return None

        rationale = item.get("rationale", "")
        if not rationale:
            return None

        # Use the customized description from expected_criteria
        ec = expected_map[name]
        result.append(Criterion(
            name=name,
            description=ec.description,
            type="likert",
            points=5,
            weight=ec.weight,
            score=raw_score,
            rationale=rationale,
        ))

    if seen != set(expected_map.keys()):
        return None

    return result


# ---------------------------------------------------------------------------
# Main rescore logic
# ---------------------------------------------------------------------------

async def _rescore_task(
    task: TaskConfig,
    config: BatchConfig,
    state: PipelineState,
    env: dict[str, str],
    models: list[str],
) -> bool:
    """Rescore a single task: dimension selection, then scoring both models."""
    task_id = task.id
    print(f"\n{'='*60}")
    print(f"[{task_id}] Starting rescore")
    print(f"{'='*60}")

    # 1. Read task context from metadata
    task_ctx = build_task_context(task, config)
    print(f"  Project: {task_ctx.get('project_name', '?')}, "
          f"Type: {task_ctx.get('task_type', '?')}, "
          f"Lang: {task_ctx.get('language', '?')}")

    # 2. Read JSONL trajectories
    qwen_jsonl = config.delivery_dir / "trajectories" / "qwen" / f"{task_id}.jsonl"
    claude_jsonl = config.delivery_dir / "trajectories" / "claude" / f"{task_id}.jsonl"

    if not qwen_jsonl.exists() or not claude_jsonl.exists():
        print(f"  [{task_id}] ERROR: Missing JSONL files")
        return False

    print(f"  [{task_id}] Extracting trajectories...")
    qwen_text = extract_for_scoring(qwen_jsonl, max_chars=TRAJECTORY_MAX_CHARS)
    claude_text = extract_for_scoring(claude_jsonl, max_chars=TRAJECTORY_MAX_CHARS)
    print(f"  [{task_id}] Qwen: {len(qwen_text)} chars, Claude: {len(claude_text)} chars")

    # 3. Round 1: Dimension selection + description customization
    print(f"  [{task_id}] Round 1: Selecting dimensions and customizing descriptions...")
    dim_prompt = build_dimension_prompt(task_ctx, qwen_text, claude_text)

    custom_criteria: list[Criterion] | None = None
    for attempt in range(1, MAX_SCORING_RETRIES + 1):
        raw = await call_scoring_ai(dim_prompt, env, model=config.claude.model, timeout=SCORING_TIMEOUT)
        if not raw:
            print(f"  [{task_id}] Empty dimension response (attempt {attempt})")
            continue
        parsed, reason = parse_dimension_output(raw)
        if parsed:
            custom_criteria = parsed
            break
        print(f"  [{task_id}] Dimension parse failed (attempt {attempt}): {reason}")

    if not custom_criteria:
        print(f"  [{task_id}] ERROR: Failed to generate custom dimensions")
        for model_name in models:
            state.set(task_id, "score", model=model_name,
                      status="failed", error="rescore: dimension selection failed")
        return False

    print(f"  [{task_id}] Selected {len(custom_criteria)} dimensions: "
          f"{[c.name for c in custom_criteria]}")

    # Write customized rubric templates
    write_rubric_pair(config.rubrics_dir, task_id, custom_criteria)

    # 4. Round 2: Score with customized dimensions
    results: dict[str, list[Criterion]] = {}

    # Score qwen first
    if "qwen" in models:
        print(f"  [{task_id}/qwen] Round 2: Scoring with custom dimensions...")
        passrate_hint = (
            "本模型是 Qwen。期望 passrate 应低于 0.7（即表现有明显不足）。"
            "请根据轨迹中的实际证据严格评分，关注失败点、遗漏和不足。"
        )
        qwen_scored = await _score_with_criteria(
            task_id, "qwen", qwen_text, custom_criteria, passrate_hint, env, config,
            task_context=task_ctx,
        )
        if qwen_scored:
            results["qwen"] = qwen_scored
        else:
            print(f"  [{task_id}/qwen] ERROR: Scoring failed")
            state.set(task_id, "score", model="qwen",
                      status="failed", error="rescore: Round 2 scoring failed")

    # Score claude
    if "claude" in models:
        qwen_pr = ""
        if "qwen" in results:
            qp = calc_passrate(results["qwen"])
            qwen_pr = f"Qwen 的 passrate 为 {qp:.4f}。"

        print(f"  [{task_id}/claude] Round 2: Scoring with custom dimensions...")
        passrate_hint = (
            f"本模型是 Claude。{qwen_pr}"
            f"期望 Claude passrate 应高于 0.7，且与 Qwen 的 passrate 差距应大于 25%。"
            f"请根据轨迹中的实际证据评分，关注完成度、工程质量和验证充分性。"
        )
        claude_scored = await _score_with_criteria(
            task_id, "claude", claude_text, custom_criteria, passrate_hint, env, config,
            task_context=task_ctx,
        )
        if claude_scored:
            results["claude"] = claude_scored
        else:
            print(f"  [{task_id}/claude] ERROR: Scoring failed")
            state.set(task_id, "score", model="claude",
                      status="failed", error="rescore: Round 2 scoring failed")

    # 5. Write results and validate
    success = True
    for model_name, scored in results.items():
        output_path = config.delivery_dir / "scores" / model_name / f"{task_id}.quality.toml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_quality_toml(output_path, scored)
        pr = calc_passrate(scored)
        state.set(task_id, "score", model=model_name, status="done", passrate=round(pr, 4))
        print(f"  [{task_id}/{model_name}] Written: passrate={pr:.4f}")

    # Validate thresholds
    if "qwen" in results and "claude" in results:
        qp = calc_passrate(results["qwen"])
        cp = calc_passrate(results["claude"])
        gap = (cp - qp) / qp if qp > 0 else 999

        threshold_issues = check_passrate_thresholds(task_id, qp, cp, True, True)
        status = "FAIL" if threshold_issues else "PASS"
        print(f"  [{task_id}] {status}: qwen={qp:.4f} claude={cp:.4f} gap={gap:.1%}")
        if threshold_issues:
            for issue in threshold_issues:
                print(f"  {issue}")
            success = False

    return success


async def _score_with_criteria(
    task_id: str,
    model_name: str,
    trajectory_text: str,
    criteria: list[Criterion],
    passrate_hint: str,
    env: dict[str, str],
    config: BatchConfig,
    task_context: dict[str, str] | None = None,
) -> list[Criterion] | None:
    """Score a single model's trajectory using customized criteria."""
    system_prompt = _build_rescore_prompt(criteria, model_name, passrate_hint, task_context)
    user_prompt = (
        f"{system_prompt}\n\n"
        f"---\n\n"
        f"## 待评分轨迹（{model_name}）\n\n{trajectory_text}\n\n"
        f"请根据上述规则输出评分 TOML。"
    )

    last_raw = ""
    for attempt in range(1, MAX_SCORING_RETRIES + 1):
        retry_hint = "" if attempt == 1 else "\n注意：请只输出合法 TOML，不要用 markdown 代码块包裹。"
        print(f"    [{task_id}/{model_name}] Calling AI (attempt {attempt})...")
        raw = await call_scoring_ai(
            user_prompt + retry_hint, env, model=config.claude.model, timeout=SCORING_TIMEOUT,
        )
        if not raw:
            print(f"    [{task_id}/{model_name}] Empty response")
            continue

        last_raw = raw
        scored = _parse_rescore_output(raw, criteria)
        if scored:
            return scored
        print(f"    [{task_id}/{model_name}] Parse failed")

    # Save draft for human review on final failure
    if last_raw:
        draft_path = config.delivery_dir / "scores" / model_name / f"{task_id}.rescore_draft.txt"
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(last_raw, encoding="utf-8")
        print(f"    [{task_id}/{model_name}] Saved draft to {draft_path.name}")

    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def rescore_all(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    models: list[str] | None = None,
) -> int:
    """Rescore tasks with customized dimensions and descriptions.

    Returns 0 on success, 1 if any task failed thresholds.
    """
    state = PipelineState(config.state_path)
    models = models or ["qwen", "claude"]
    tasks = select_delivery_tasks(config, task_ids)

    if not config.claude.auth_token or not config.claude.base_url or not config.claude.model:
        print("ERROR: Claude scoring config is incomplete in .env")
        return 1

    env = build_validated_env(config.claude)
    sem = asyncio.Semaphore(config.max_parallel * 2)

    all_passed = True

    async def bounded_rescore(task: TaskConfig) -> bool:
        async with sem:
            return await _rescore_task(task, config, state, env, models)

    print(f"Rescoring {len(tasks)} tasks with customized dimensions...")

    with state.batch():
        results = await asyncio.gather(
            *[bounded_rescore(t) for t in tasks],
            return_exceptions=True,
        )
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                print(f"[{task.id}] ERROR: {result}")
                for model_name in models:
                    state.set(task.id, "score", model=model_name,
                              status="failed", error=f"rescore exception: {result}")
                all_passed = False
            elif not result:
                all_passed = False

    passed = sum(1 for r in results if r is True)
    failed = len(results) - passed
    print(f"\nRescore complete: {passed} passed, {failed} failed out of {len(results)} tasks")

    return 0 if all_passed else 1
