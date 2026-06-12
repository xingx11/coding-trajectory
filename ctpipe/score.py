"""score subcommand: AI-generate initial quality scores from trajectories."""

from __future__ import annotations

import asyncio
import tomllib

from ctpipe import strip_claude_wrapper
from ctpipe.config import (
    MAX_CRITERIA_COUNT,
    MAX_SCORING_RETRIES,
    MIN_CRITERIA_COUNT,
    SCORING_TIMEOUT,
    TRAJECTORY_MAX_CHARS,
    BatchConfig,
    TaskConfig,
    build_bad_pattern_table,
    build_reference_dimension_table,
    build_validated_env,
    check_passrate_thresholds,
    is_valid_criterion_name,
    select_delivery_tasks,
)
from ctpipe.state import PipelineState
from ctpipe.toml_utils import Criterion, calc_passrate, has_custom_descriptions, has_score_tiers, read_quality_toml, write_quality_toml, write_rubric_pair
from ctpipe.trajectory import extract_for_scoring


def build_context_block(task_context: dict[str, str] | None) -> str:
    """Build a 任务背景 section from task context dict. Shared by score and rescore."""
    if not task_context:
        return ""
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
        return "\n## 任务背景\n\n" + "\n".join(ctx_parts) + "\n"
    return ""


def _build_scoring_prompt(
    fixed_criteria: list[str] | None = None,
    custom_criteria: list[Criterion] | None = None,
    task_context: dict[str, str] | None = None,
    passrate_hint: str | None = None,
) -> str:
    """Build the scoring system prompt.

    Args:
        fixed_criteria: If provided, AI must score exactly these dimensions.
            If None, AI selects 6-10 from the 20 reference dimensions.
        custom_criteria: If provided, use these customized descriptions
            (project-specific, with score tiers). Takes precedence over
            fixed_criteria for description text.
        task_context: If provided, inject task/project background into prompt.
        passrate_hint: If provided, inject passrate soft constraint into prompt.
    """
    bad_patterns = build_bad_pattern_table()

    if custom_criteria:
        # Custom mode: use project-specific descriptions from rubric template
        dim_lines: list[str] = []
        for c in custom_criteria:
            dim_lines.append(f"- `{c.name}` (weight={c.weight}): {c.description}")
        dim_table = "\n".join(dim_lines)
        dim_count = len(custom_criteria)
        selection_instruction = (
            f"你必须使用以下 {dim_count} 个评分维度及其定制化描述，不得增加或减少：\n\n"
            f"{dim_table}\n\n"
            f"注意：这些 description 已根据当前项目和任务特征量身定制，"
            f"你必须在输出的 TOML 中完整保留这些定制化 description，不得替换为通用模板。"
        )
    elif fixed_criteria:
        # Fixed mode: score exactly these dimensions
        dim_table = build_reference_dimension_table(fixed_criteria)
        dim_count = len(fixed_criteria)
        selection_instruction = (
            f"你必须使用以下 {dim_count} 个评分维度，不得增加或减少：\n\n"
            f"{dim_table}"
        )
    else:
        # Free-selection mode: pick 6-10 from 20 reference dimensions
        dim_table = build_reference_dimension_table()
        selection_instruction = (
            f"从以下 20 个参考维度中选择 7-{MAX_CRITERIA_COUNT} 个"
            f"最能反映本次轨迹质量差异的维度：\n\n"
            f"{dim_table}\n\n"
            f"选择规则：\n"
            f"- 基于 JSONL 轨迹中真实出现的任务特征和行为证据选择\n"
            f"- 如果某个维度没有可观察证据，不要选择它\n"
            f"- 优先选择与核心目标、失败点、验收标准和最终可用性最相关的维度\n\n"
            f"description 定制化要求：\n"
            f"- 输出的 description 必须融入本次任务的项目名称、技术栈、具体操作等信息\n"
            f"- 不要照搬上面的通用参考描述，要根据轨迹中的实际内容进行定制化改写\n"
            f"- 保留 1-5 分档位结构，但各档位描述要具体到本项目的场景"
        )

    # Build task context block if available
    context_block = build_context_block(task_context)

    # Build passrate constraint block if hint provided
    passrate_block = ""
    if passrate_hint:
        passrate_block = (
            "\n## passrate 约束提示\n\n"
            f"{passrate_hint}\n\n"
            "这是一个软约束——你应该在合理范围内考虑这个目标，但评分必须有真实证据支撑。\n"
            "如果轨迹质量确实无法支持目标 passrate，按实际情况评分并在 rationale 中说明。\n"
        )

    return f"""你是评分 AI，负责评价一次 coding-assistant 执行轨迹的质量。
{context_block}
## 维度选择

{selection_instruction}
{passrate_block}
## 评分规则

- 分数为 1-5 的整数
- 1分：几乎无有效进展或严重失败
- 2分：仅部分完成，关键要求缺失
- 3分：主路径完成但有明显遗漏
- 4分：大部分完成，仅有轻微问题
- 5分：完整、高质量、有充分验证
- weight 必须使用评分模板中指定的权重值（与架构边界/安全合规相关的维度为 2.0，其余为 1.0）
- description 必须使用上面提供的中文完整描述，保持一行字符串

## 高分校准与封顶规则（必须遵守）

评分以 3 分为基准，根据证据上下调整：
- 3分 = 主路径有推进但有明显缺口
- 4分 = 主路径基本闭环，仅有轻微问题
- 5分 = 罕见高分，需要正面证据闭环、无反面证据、无相关 Bad Pattern

以下情况必须封顶：
- 该维度在轨迹中缺少直接证据 → 最高 3 分；完全无证据 → 通常 2 分以下
- 任务要求改代码/写文件但轨迹中没有修改证据 → delivery/engineering 维度最高 2 分
- 任务要求验证但没有运行测试/构建/lint → testing 维度最高 2 分；delivery 维度最高 4 分
- 仅基于模型最终自述评分，无 tool/file/test 证据 → evidence 维度最高 2 分；相关 delivery 维度最高 3 分
- 工具报错且未有效补救 → tool_usage 维度最高 3 分；若导致任务未完成则最高 2 分
- 命中 Bad Pattern → 相关维度通常最高 3 分；严重的最高 2 分
- 给 5 分时，rationale 必须正面论证为什么没有显著扣分点；如果只能写出模糊夸奖，最高给 4 分

## 轨迹截断公平性（必须遵守）

如果轨迹明显在中途截断（如 followup 未全部执行、对话突然中断、超时终止）：
- 评分必须基于已完成部分的实际质量，不得因截断导致的不完整而额外惩罚
- delivery/completeness 类维度可以因任务未完成而合理扣分，但必须在 rationale 中注明"轨迹截断"
- 其他维度（如 semantic_understanding、tool_usage、context_exploration）应仅评价已执行部分的表现
- 禁止因截断而给所有维度统一低分——截断前的高质量工作仍应获得相应评价
- 如果两个模型的轨迹长度差异显著（如一个有 5 轮对话，另一个只有 2 轮），这可能是截断而非能力差异

## 需求变更归因（必须遵守）

如果用户在对话过程中改变了需求（如重命名属性、调整 API 设计、变更功能范围）：
- 由用户需求变更导致的模型返工/迭代不应视为模型能力不足
- 应区分"模型自身理解错误导致的返工"和"用户主动变更需求导致的返工"
- 评价 stage_progression / planning 类维度时，用户需求变更造成的反复不应扣分
- 模型在需求变更后能快速适应并正确实现新要求，应视为正面表现

## rationale 写作要求（严格遵守）

- 必须使用中文
- 必须引用轨迹中的具体证据：哪些文件被修改、运行了什么命令、出了什么错、遗漏了什么
- 每条 rationale 必须独立且不同，禁止跨维度复制相似句式
- 1-3 句话，简洁具体
- 禁止使用以下套话模式：
  × "整体看，XXX有一些有效推进，但稳定性和完整性不够"
  × "这项给X分比较稳/更贴近实际"
  × "因此给低中档分数" / "所以给中档分"
  × "推进痕迹是有的，只是前后反复比较多"
  × "前面先去看了...后面再回到...补细节"
  × "最终至少落成了...这一类实物"
  × 任何以"我会给这一项X分"开头的句式
- 合格示例：
  ✓ "改了三个文件但漏掉了edge case的单元测试，validate那步直接跳过了"
  ✓ "prompt里要求加日志，它确实加了logging，但log level全用的INFO，没按要求区分WARNING"
- 不要提及 task ID (CT-XXXX) 或与另一个模型做比较

## 分数与理由一致性（红线规则）

- rationale 中描述的证据方向必须与分数方向一致
- 如果 rationale 描述了负面事实（如"没有测试"、"未完成"、"有缺陷"、"不够"、"缺少"），分数不得为 4 或 5
- 如果 rationale 描述了正面事实（如"完整实现"、"覆盖全面"、"准确定位"），分数不得为 1 或 2
- 禁止在 rationale 中写"给 X 分"——分数由 score 字段决定，rationale 只陈述事实

## 输出前自检（必须执行）

输出 TOML 前，逐条检查：
1. 每个 score=5 的维度：rationale 是否有充分正面证据？是否触犯了封顶规则？
2. 每个 score=1 的维度：rationale 是否确实描述了严重失败？
3. rationale 中是否存在与 score 方向矛盾的描述？（如负面描述+高分）
4. 是否有两条以上 rationale 使用了相似句式或模板？如有，必须重写使其独立
5. description 中定义的各档位标准是否与实际给分对应？
如发现矛盾，修正分数使其与证据一致，而非修改理由来匹配分数。

## Bad Pattern 识别

请检查轨迹是否命中以下 Bad Pattern，如有命中请在 TOML 之外单独说明，不要写入 TOML：

{bad_patterns}

## 输出格式

只输出合法 TOML，不要包含 Markdown 代码块、标题或解释文本。
所有字符串字段使用普通双引号 `"..."`，不使用 TOML 多行字符串。
description 和 rationale 都必须写成一行。

单个 criterion 格式：

[[criterion]]
name = "维度名称"
description = "中文完整评分标准（包含1-5分档位定义）"
type = "likert"
points = 5
weight = 1.0
score = 3
rationale = "中文评分理由，引用具体轨迹证据"

如果命中 Bad Pattern，在 TOML 之后另起一行写：
Bad Pattern 命中：
- xxx：具体说明
如果未命中，写：
Bad Pattern 命中：未发现明确命中。
"""


async def call_scoring_ai(
    prompt: str,
    env: dict[str, str],
    model: str = "",
    timeout: int = 300,
) -> str:
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "text",
        "--dangerously-skip-permissions",
        "--setting-sources", "local",
        "--bare",
    ]
    if model:
        cmd += ["--model", model]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return ""

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        print(f"  WARNING: claude -p scoring exited with {proc.returncode}: {err[:300]}")

    return stdout.decode("utf-8", errors="replace")


def extract_toml_section(raw: str) -> str:
    """Extract pure TOML content, stripping Bad Pattern section and markdown fences."""
    cleaned = strip_claude_wrapper(raw)

    # Split at "Bad Pattern" line if present
    for marker in ("Bad Pattern 命中", "Bad Pattern命中", "bad pattern"):
        idx = cleaned.lower().find(marker.lower())
        if idx > 0:
            cleaned = cleaned[:idx].rstrip()
            break

    # Fallback: if cleaned doesn't look like TOML, try to find [[criterion]]
    stripped = cleaned.lstrip()
    if stripped and not stripped.startswith("[[") and "[[criterion]]" in cleaned:
        idx = cleaned.index("[[criterion]]")
        cleaned = cleaned[idx:]

    return cleaned


def extract_bad_pattern(raw: str) -> str:
    """Extract the first matched bad pattern key from AI scoring output.

    Looks for the "Bad Pattern 命中" section after the TOML block and
    matches against the known BAD_PATTERNS list.

    Returns the pattern key (e.g. 'lazy_shortcut') if found, empty string otherwise.
    """
    from ctpipe.config import BAD_PATTERNS

    cleaned = strip_claude_wrapper(raw)

    for marker in ("Bad Pattern 命中", "Bad Pattern命中", "bad pattern"):
        idx = cleaned.lower().find(marker.lower())
        if idx >= 0:
            bp_text = cleaned[idx:]
            # Check for explicit "not found" indicators
            if any(neg in bp_text[:120] for neg in ("未发现", "未命中", "无明确", "无命中")):
                return ""
            # Match known pattern keys
            for pattern in BAD_PATTERNS:
                if pattern in bp_text:
                    return pattern
            return ""
    return ""


def _parse_scored_toml(
    raw: str,
    *,
    expected_names: list[str] | None = None,
    use_custom_descriptions: bool = False,
    custom_criteria: list[Criterion] | None = None,
) -> list[Criterion] | None:
    """Parse AI-generated scored TOML.

    Validates:
    - 6-10 criterion blocks
    - All names are valid snake_case, no duplicates
    - Scores are integers 1-5
    - Rationale is non-empty
    - If expected_names given, the name set must match exactly

    Args:
        use_custom_descriptions: If True, validate that AI-generated
            descriptions contain 1-5 score tier definitions.
        custom_criteria: If provided, override AI-returned descriptions
            and weights with the originals from these criteria (same
            safeguard as rescore.py's _parse_rescore_output).

    Returns parsed Criterion list (with descriptions/weights from TOML) or None.
    """
    cleaned = extract_toml_section(raw)

    try:
        data = tomllib.loads(cleaned)
    except Exception:
        return None

    scored = data.get("criterion", [])
    if not (MIN_CRITERIA_COUNT <= len(scored) <= MAX_CRITERIA_COUNT):
        return None

    # Build lookup map for custom criteria overrides
    custom_map: dict[str, Criterion] = {}
    if custom_criteria:
        custom_map = {c.name: c for c in custom_criteria}

    seen_names: set[str] = set()
    result: list[Criterion] = []

    for item in scored:
        name = item.get("name", "")
        if not is_valid_criterion_name(name):
            return None
        if name in seen_names:
            return None
        seen_names.add(name)

        raw_score = item.get("score", 0)
        if not isinstance(raw_score, int):
            # Reject non-integer scores (e.g. 3.5 from TOML float)
            return None
        score = raw_score
        if score < 1 or score > 5:
            return None

        rationale = item.get("rationale", "")
        if not rationale:
            return None

        # Use original custom descriptions/weights if available (prevents AI drift)
        if name in custom_map:
            ec = custom_map[name]
            description = ec.description
            weight = ec.weight
        else:
            # Read description and weight directly from the TOML item
            description = item.get("description", "")
            if use_custom_descriptions and not has_score_tiers(description):
                return None
            weight = item.get("weight", 1.0)

        result.append(Criterion(
            name=name,
            description=description,
            type="likert",
            points=5,
            weight=weight,
            score=score,
            rationale=rationale,
        ))

    if expected_names is not None:
        if seen_names != set(expected_names):
            return None

    return result


async def _auto_customize_criteria(
    task: TaskConfig,
    config: BatchConfig,
    env: dict[str, str],
) -> list[Criterion] | None:
    """Auto-generate customized criteria when rubric template uses generic descriptions.

    Reuses rescore.py's dimension selection + description customization logic:
    reads task context and both trajectories, calls AI to pick 7-10 dimensions
    with project-specific descriptions, and writes the result as rubric templates.

    Returns customized criteria list, or None on failure (caller falls back to
    free selection mode).
    """
    # Lazy import to avoid circular dependency (rescore imports from score)
    from ctpipe.rescore import (
        build_dimension_prompt,
        build_task_context,
        parse_dimension_output,
    )

    task_id = task.id

    # 1. Read task context from metadata
    task_ctx = build_task_context(task, config)

    # 2. Read collected trajectories for both models
    qwen_jsonl = config.resolve_trajectory_path(task_id, "qwen")
    claude_jsonl = config.resolve_trajectory_path(task_id, "claude")

    if not qwen_jsonl.exists() or not claude_jsonl.exists():
        print(f"  [{task_id}] Auto-customize skipped: missing trajectory JSONL")
        return None

    qwen_text = extract_for_scoring(qwen_jsonl, max_chars=TRAJECTORY_MAX_CHARS)
    claude_text = extract_for_scoring(claude_jsonl, max_chars=TRAJECTORY_MAX_CHARS)

    # 3. Call AI for dimension selection + description customization
    dim_prompt = build_dimension_prompt(task_ctx, qwen_text, claude_text)

    for attempt in range(1, MAX_SCORING_RETRIES + 1):
        raw = await call_scoring_ai(dim_prompt, env, model=config.claude.model, timeout=SCORING_TIMEOUT)
        if not raw:
            continue
        parsed, reason = parse_dimension_output(raw)
        if parsed:
            # 4. Write customized rubric templates for both models
            write_rubric_pair(config.rubrics_dir, task_id, parsed)
            print(f"  [{task_id}] Auto-generated customized criteria ({len(parsed)} dimensions)")
            return parsed
        print(f"  [{task_id}] Auto-customize attempt {attempt} failed: {reason}")

    print(f"  [{task_id}] Auto-customize failed after 3 attempts, falling back to free selection")
    return None


async def score_single(
    task: TaskConfig,
    model_name: str,
    config: BatchConfig,
    state: PipelineState,
    env: dict[str, str],
    fixed_criteria: list[str] | None = None,
    custom_criteria: list[Criterion] | None = None,
    passrate_hint: str | None = None,
) -> tuple[bool, list[str] | None]:
    """Score a single task/model trajectory.

    Args:
        fixed_criteria: If provided, AI must score exactly these dimensions.
        custom_criteria: If provided, use these customized Criterion objects
            (with project-specific descriptions) for scoring.
        passrate_hint: If provided, inject passrate soft constraint into prompt.

    Returns:
        (success, selected_criteria_names) - names is None on failure.
    """
    collect_info = state.get(task.id, "collect", model_name)
    if collect_info.get("status") != "done":
        print(f"  [{task.id}/{model_name}] collect not done, skipping score")
        return False, None

    jsonl_rel = collect_info.get("jsonl_path", "")
    jsonl_path = config.delivery_dir / jsonl_rel
    if not jsonl_path.exists():
        print(f"  [{task.id}/{model_name}] ERROR: JSONL not found: {jsonl_path}")
        state.set(task.id, "score", model=model_name, status="failed", error="JSONL not found")
        return False, None

    print(f"  [{task.id}/{model_name}] Extracting trajectory content...")
    trajectory_text = extract_for_scoring(jsonl_path)
    print(f"  [{task.id}/{model_name}] Trajectory: {len(trajectory_text)} chars")

    # Determine if we're using custom descriptions
    use_custom = custom_criteria is not None
    if use_custom:
        fixed_names = [c.name for c in custom_criteria]
    else:
        fixed_names = fixed_criteria

    # Build task context for scoring prompt
    from ctpipe.rescore import build_task_context
    task_ctx = build_task_context(task, config)

    system_prompt = _build_scoring_prompt(
        fixed_criteria=fixed_names,
        custom_criteria=custom_criteria,
        task_context=task_ctx,
        passrate_hint=passrate_hint,
    )
    user_prompt = (
        f"{system_prompt}\n\n"
        f"---\n\n"
        f"## 待评分轨迹\n\n{trajectory_text}\n\n"
        f"请根据上述规则输出评分 TOML。"
    )

    raw_output = ""
    scored = None
    output_path = config.score_path(task.id, model_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_SCORING_RETRIES + 1):
        retry_hint = "" if attempt == 1 else "\n注意：请只输出合法 TOML，不要用 markdown 代码块包裹。"
        print(f"  [{task.id}/{model_name}] Calling AI for scoring (attempt {attempt}/{MAX_SCORING_RETRIES})...")
        raw_output = await call_scoring_ai(
            user_prompt + retry_hint, env, model=config.claude.model,
            timeout=SCORING_TIMEOUT,
        )

        if not raw_output:
            if attempt < MAX_SCORING_RETRIES:
                print(f"  [{task.id}/{model_name}] Empty response, retrying...")
                continue
            print(f"  [{task.id}/{model_name}] ERROR: empty AI response after {MAX_SCORING_RETRIES} attempts")
            state.set(task.id, "score", model=model_name, status="failed", error="empty response")
            return False, None

        scored = _parse_scored_toml(
            raw_output,
            expected_names=fixed_names,
            use_custom_descriptions=use_custom,
            custom_criteria=custom_criteria,
        )
        if scored is not None:
            break
        if attempt < MAX_SCORING_RETRIES:
            print(f"  [{task.id}/{model_name}] Parse failed, retrying...")

    if scored is None:
        draft_path = output_path.with_suffix(".draft.txt")
        draft_path.write_text(raw_output, encoding="utf-8")
        print(f"  [{task.id}/{model_name}] WARNING: could not parse AI output, saved to {draft_path.name}")
        state.set(task.id, "score", model=model_name, status="draft", draft_path=str(draft_path))
        return False, None

    selected_names = [c.name for c in scored]
    write_quality_toml(output_path, scored)
    passrate = calc_passrate(scored)
    print(f"  [{task.id}/{model_name}] Scored: passrate={passrate:.4f} ({len(scored)} criteria)")

    # Extract bad pattern from AI output
    detected_bp = extract_bad_pattern(raw_output)
    state_kwargs: dict = {"status": "done", "passrate": round(passrate, 4)}
    if detected_bp:
        state_kwargs["bad_pattern"] = detected_bp
        print(f"  [{task.id}/{model_name}] Detected bad pattern: {detected_bp}")
    state.set(task.id, "score", model=model_name, **state_kwargs)
    return True, selected_names


def _read_existing_criteria_names(score_path) -> list[str] | None:
    """Read criterion names from an existing scored TOML file.

    Returns None if the file is invalid, incomplete, or uses malformed names.
    """
    try:
        criteria = read_quality_toml(score_path)
        if not criteria:
            return None
        if not (MIN_CRITERIA_COUNT <= len(criteria) <= MAX_CRITERIA_COUNT):
            return None
        if not all(c.score >= 1 and c.rationale for c in criteria):
            return None
        names = [c.name for c in criteria]
        if not all(is_valid_criterion_name(n) for n in names):
            return None
        return names
    except Exception as exc:
        print(f"  WARNING: could not read existing criteria from {score_path.name}: {exc}")
    return None


async def score_all(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    models: list[str] | None = None,
    auto_rescore: bool = False,
    *,
    dry_run: bool = False,
    as_json: bool = False,
) -> dict | None:
    models = models or ["qwen", "claude"]
    tasks = select_delivery_tasks(config, task_ids)

    if dry_run:
        state = PipelineState(config.state_path)
        tasks_data = []
        will_score = 0
        will_skip = 0

        if not as_json:
            print("=" * 60)
            print("  DRY RUN: score")
            print("=" * 60)
            print("\nCriteria modes:")
            print("  custom_criteria  — rubric template with project-specific descriptions")
            print("  fixed_criteria   — reuse dimensions from another model's scoring")
            print("  free_selection   — AI selects 7-10 from 20 reference dimensions")

        for task in tasks:
            # Step 1: check for custom rubric template
            custom_tpl_criteria = None
            custom_tpl_source = None
            for model_check in ("qwen", "claude"):
                tpl_path = config.rubrics_dir / model_check / f"{task.id}.quality.toml"
                if tpl_path.exists():
                    try:
                        tpl = read_quality_toml(tpl_path)
                        if tpl and has_custom_descriptions(tpl):
                            custom_tpl_criteria = tpl
                            custom_tpl_source = model_check
                            break
                    except Exception:
                        pass

            # Step 2: if no custom template, check if auto-customize would trigger
            auto_customize = False
            if custom_tpl_criteria is None:
                qwen_jsonl = config.resolve_trajectory_path(task.id, "qwen")
                claude_jsonl = config.resolve_trajectory_path(task.id, "claude")
                if qwen_jsonl.exists() and claude_jsonl.exists():
                    auto_customize = True

            # Step 3: check existing qwen score for dimension reuse
            qwen_scored_names: list[str] | None = None
            qwen_score_path = config.resolve_score_path(task.id, "qwen")
            if qwen_score_path.exists():
                try:
                    qc = read_quality_toml(qwen_score_path)
                    if qc and all(c.score >= 1 and c.rationale for c in qc):
                        qwen_scored_names = [c.name for c in qc]
                except Exception:
                    pass

            if not as_json:
                print(f"\n[{task.id}]")

            models_info = {}
            for model_name in models:
                already_done = state.is_done(task.id, "score", model_name)
                score_path = config.resolve_score_path(task.id, model_name)
                traj_path = config.resolve_trajectory_path(task.id, model_name)
                traj_exists = traj_path.exists()

                if already_done and score_path.exists():
                    will_skip += 1
                    # Determine what mode was used
                    if custom_tpl_criteria:
                        mode = "custom_criteria"
                    elif qwen_scored_names and model_name == "claude":
                        mode = "fixed_criteria"
                    elif qwen_scored_names and model_name == "qwen":
                        mode = "custom_criteria" if has_custom_descriptions(
                            read_quality_toml(score_path)) else "free_selection"
                    else:
                        mode = "free_selection"
                    models_info[model_name] = {
                        "skip": True, "mode": mode,
                        "trajectory": str(traj_path), "trajectory_exists": traj_exists,
                    }
                    if not as_json:
                        print(f"  {model_name}: SKIP (already done)  mode={mode}")
                    continue

                will_score += 1
                # Determine criteria mode for this model
                if model_name == "qwen":
                    if custom_tpl_criteria:
                        mode = "custom_criteria"
                        detail = f"from {custom_tpl_source} rubric template, {len(custom_tpl_criteria)} dims"
                    elif auto_customize:
                        mode = "custom_criteria"
                        detail = "auto-customize will generate project-specific descriptions"
                    else:
                        mode = "free_selection"
                        detail = "AI selects 7-10 from 20 reference dimensions"
                else:
                    if custom_tpl_criteria:
                        mode = "custom_criteria"
                        detail = f"from {custom_tpl_source} rubric template, {len(custom_tpl_criteria)} dims"
                    elif qwen_scored_names:
                        mode = "fixed_criteria"
                        detail = f"reuse qwen's {len(qwen_scored_names)} dims: {', '.join(qwen_scored_names)}"
                    else:
                        mode = "free_selection"
                        detail = "no qwen reference, AI selects independently"

                models_info[model_name] = {
                    "skip": False, "mode": mode, "detail": detail,
                    "trajectory": str(traj_path), "trajectory_exists": traj_exists,
                }
                if not as_json:
                    traj_status = "OK" if traj_exists else "MISSING"
                    print(f"  {model_name}: RUN  mode={mode}")
                    print(f"    detail: {detail}")
                    print(f"    trajectory: {traj_path}  [{traj_status}]")

            tasks_data.append({
                "task_id": task.id,
                "auto_customize": auto_customize,
                "custom_template_source": custom_tpl_source,
                "models": models_info,
            })

        if as_json:
            return {
                "criteria_modes": {
                    "custom_criteria": "rubric template with project-specific descriptions",
                    "fixed_criteria": "reuse dimensions from another model's scoring",
                    "free_selection": "AI selects 7-10 from 20 reference dimensions",
                },
                "tasks": tasks_data,
                "summary": {
                    "total_slots": len(tasks) * len(models),
                    "to_score": will_score,
                    "skipped": will_skip,
                },
            }

        print(f"\nTotal: {len(tasks)} task(s) x {len(models)} model(s) = "
              f"{len(tasks) * len(models)} slot(s): "
              f"{will_score} to score, {will_skip} already done")
        return

    state = PipelineState(config.state_path)
    sem = asyncio.Semaphore(config.max_parallel * 2)

    env = build_validated_env(config.claude)

    async def score_task_pair(task: TaskConfig) -> None:
        """Score one task: qwen first (free selection), then claude (same dimensions)."""
        async with sem:
            selected_names: list[str] | None = None
            custom_crit: list[Criterion] | None = None

            # Check if rubric template has customized descriptions
            for model_check in ("qwen", "claude"):
                tpl_path = config.rubrics_dir / model_check / f"{task.id}.quality.toml"
                if tpl_path.exists():
                    try:
                        tpl_criteria = read_quality_toml(tpl_path)
                        if tpl_criteria and has_custom_descriptions(tpl_criteria):
                            custom_crit = tpl_criteria
                            print(f"[{task.id}] Using customized rubric template ({len(tpl_criteria)} criteria)")
                            break
                    except Exception as exc:
                        print(f"  [{task.id}] WARNING: could not read rubric template: {exc}")

            # Auto-customize criteria if no custom template found
            if custom_crit is None:
                custom_crit = await _auto_customize_criteria(task, config, env)

            # Check if qwen already scored — reuse its dimensions
            if "qwen" in models:
                qwen_score_path = config.resolve_score_path(task.id, "qwen")
                if state.is_done(task.id, "score", "qwen") and qwen_score_path.exists():
                    # If custom template exists, check if existing score uses it
                    if custom_crit:
                        try:
                            existing = read_quality_toml(qwen_score_path)
                            if existing and not has_custom_descriptions(existing):
                                print(f"[{task.id}/qwen] score uses generic descriptions but custom template available, re-scoring...")
                                state.reset(task.id, "score", "qwen")
                            else:
                                selected_names = _read_existing_criteria_names(qwen_score_path)
                                if selected_names:
                                    print(f"[{task.id}/qwen] score already done with custom descriptions, reusing")
                        except Exception as exc:
                            print(f"  [{task.id}/qwen] WARNING: could not read existing score: {exc}")
                            state.reset(task.id, "score", "qwen")
                    else:
                        selected_names = _read_existing_criteria_names(qwen_score_path)
                        if selected_names:
                            print(f"[{task.id}/qwen] score already done, reusing {len(selected_names)} criteria")
                        else:
                            print(f"[{task.id}/qwen] score marked done but invalid, re-scoring...")
                            state.reset(task.id, "score", "qwen")

                if not state.is_done(task.id, "score", "qwen"):
                    qwen_hint = (
                        "本模型是 Qwen。期望 passrate 应低于 0.7（即表现有明显不足）。"
                        "请根据轨迹中的实际证据严格评分，关注失败点、遗漏和不足。"
                    )
                    if custom_crit:
                        print(f"[{task.id}/qwen] Scoring trajectory (custom criteria)...")
                        success, names = await score_single(
                            task, "qwen", config, state, env, custom_criteria=custom_crit,
                            passrate_hint=qwen_hint,
                        )
                    else:
                        print(f"[{task.id}/qwen] Scoring trajectory (free selection)...")
                        success, names = await score_single(
                            task, "qwen", config, state, env, passrate_hint=qwen_hint,
                        )
                    if success and names:
                        selected_names = names

            # Score claude with the same dimensions qwen used
            if "claude" in models:
                # Build claude passrate hint (include qwen's passrate if available)
                qwen_pr_str = ""
                qwen_info = state.get(task.id, "score", "qwen")
                if qwen_info.get("passrate"):
                    qwen_pr_str = f"Qwen 的 passrate 为 {qwen_info['passrate']:.4f}。"
                claude_hint = (
                    f"本模型是 Claude。{qwen_pr_str}"
                    f"期望 Claude passrate 应高于 0.7，且与 Qwen 的 passrate 差距应大于 25%。"
                    f"请根据轨迹中的实际证据评分，关注完成度、工程质量和验证充分性。"
                )

                # If qwen wasn't in models but has existing scores, reuse its dimensions
                if selected_names is None:
                    qwen_score_path = config.resolve_score_path(task.id, "qwen")
                    if qwen_score_path.exists():
                        selected_names = _read_existing_criteria_names(qwen_score_path)
                        if selected_names:
                            print(f"[{task.id}] Reusing existing qwen dimensions ({len(selected_names)} criteria)")

                claude_score_path = config.resolve_score_path(task.id, "claude")
                if state.is_done(task.id, "score", "claude") and claude_score_path.exists():
                    # If custom template exists, check if existing score uses it
                    if custom_crit:
                        try:
                            existing_crit = read_quality_toml(claude_score_path)
                            if existing_crit and not has_custom_descriptions(existing_crit):
                                print(f"[{task.id}/claude] score uses generic descriptions but custom template available, re-scoring...")
                                state.reset(task.id, "score", "claude")
                            else:
                                print(f"[{task.id}/claude] score already done with custom descriptions, skipping")
                                return
                        except Exception as exc:
                            print(f"  [{task.id}/claude] WARNING: could not read existing score: {exc}")
                            state.reset(task.id, "score", "claude")
                    else:
                        existing = _read_existing_criteria_names(claude_score_path)
                        if existing:
                            # Check consistency: if qwen was just scored, verify dimensions match
                            if selected_names and set(existing) != set(selected_names):
                                print(f"[{task.id}/claude] dimension mismatch with qwen, re-scoring...")
                                state.reset(task.id, "score", "claude")
                            else:
                                print(f"[{task.id}/claude] score already done, skipping")
                                return
                        else:
                            print(f"[{task.id}/claude] score marked done but invalid, re-scoring...")
                            state.reset(task.id, "score", "claude")

                if custom_crit:
                    print(f"[{task.id}/claude] Scoring trajectory (custom criteria)...")
                    await score_single(
                        task, "claude", config, state, env, custom_criteria=custom_crit,
                        passrate_hint=claude_hint,
                    )
                elif selected_names:
                    # Try to read full criteria (with descriptions) from qwen's scored file
                    # to avoid losing custom descriptions in the fixed_criteria path
                    qwen_full_crit = None
                    qwen_scored = config.resolve_score_path(task.id, "qwen")
                    if qwen_scored.exists():
                        try:
                            qc = read_quality_toml(qwen_scored)
                            if qc and has_custom_descriptions(qc):
                                qwen_full_crit = [
                                    Criterion(name=c.name, description=c.description, type=c.type,
                                              points=c.points, weight=c.weight, score=0, rationale="")
                                    for c in qc
                                ]
                        except Exception:
                            pass
                    if qwen_full_crit:
                        print(f"[{task.id}/claude] Scoring trajectory (using qwen's {len(qwen_full_crit)} criteria with descriptions)...")
                        await score_single(task, "claude", config, state, env,
                                           custom_criteria=qwen_full_crit, passrate_hint=claude_hint)
                    else:
                        print(f"[{task.id}/claude] Scoring trajectory (using qwen's {len(selected_names)} criteria names)...")
                        await score_single(task, "claude", config, state, env,
                                           fixed_criteria=selected_names, passrate_hint=claude_hint)
                else:
                    # Qwen failed or not in models — claude does free selection
                    print(f"[{task.id}/claude] Scoring trajectory (free selection, no qwen reference)...")
                    await score_single(task, "claude", config, state, env,
                                       passrate_hint=claude_hint)

            # Handle non-standard model lists (single model only)
            for model_name in models:
                if model_name not in ("qwen", "claude"):
                    if state.is_done(task.id, "score", model_name):
                        score_path = config.resolve_score_path(task.id, model_name)
                        if score_path.exists():
                            print(f"[{task.id}/{model_name}] score already done, skipping")
                            continue
                        state.reset(task.id, "score", model_name)
                    print(f"[{task.id}/{model_name}] Scoring trajectory...")
                    await score_single(task, model_name, config, state, env)

    coros = [score_task_pair(task) for task in tasks]
    if coros:
        with state.batch():
            results = await asyncio.gather(*coros, return_exceptions=True)
            for task, result in zip(tasks, results):
                if isinstance(result, Exception):
                    print(f"[{task.id}] ERROR: {result}")
                    for model_name in models:
                        if not state.is_done(task.id, "score", model_name):
                            state.set(task.id, "score", model=model_name, status="failed", error=str(result))

    # Auto-rescore: check thresholds and trigger rescore for failing tasks
    if auto_rescore:
        state.reload()
        failed_task_ids: list[str] = []
        for task in tasks:
            qwen_info = state.get(task.id, "score", "qwen")
            claude_info = state.get(task.id, "score", "claude")
            qp = qwen_info.get("passrate", 0)
            cp = claude_info.get("passrate", 0)
            has_q = qwen_info.get("status") == "done"
            has_c = claude_info.get("status") == "done"
            if not has_q or not has_c:
                continue
            threshold_issues = check_passrate_thresholds(task.id, qp, cp, has_q, has_c)
            if threshold_issues:
                failed_task_ids.append(task.id)
                for issue in threshold_issues:
                    print(f"  {issue}")

        if failed_task_ids:
            print(f"\n{len(failed_task_ids)} task(s) failed passrate thresholds, auto-triggering rescore...")
            from ctpipe.rescore import rescore_all
            rescore_result = await rescore_all(config, failed_task_ids, models)
            if rescore_result == 0:
                print("Auto-rescore: all tasks now pass thresholds.")
            else:
                print(f"Auto-rescore: some tasks still fail thresholds.")
        else:
            print("All scored tasks pass passrate thresholds.")

    print("Score complete.")
