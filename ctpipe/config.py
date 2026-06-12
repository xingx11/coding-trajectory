"""Load tasks.toml and .env into typed configuration objects."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


@dataclass
class ModelConfig:
    auth_token: str
    base_url: str
    model: str


@dataclass
class TaskConfig:
    id: str
    project_path: Path
    clone_method: str
    task_type: str
    domain: str
    language: str
    prompt_qwen: str
    prompt_claude: str
    followups_qwen: list[str] = field(default_factory=list)
    followups_claude: list[str] = field(default_factory=list)
    bad_pattern: str = ""
    task_title: str = ""
    task_description: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)

    @property
    def project_subdir(self) -> str:
        return self.project_path.name

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "project_path": str(self.project_path),
            "clone_method": self.clone_method,
            "task_type": self.task_type,
            "domain": self.domain,
            "language": self.language,
            "prompt_qwen": self.prompt_qwen,
            "prompt_claude": self.prompt_claude,
            "followups_qwen": list(self.followups_qwen),
            "followups_claude": list(self.followups_claude),
            "bad_pattern": self.bad_pattern,
            "task_title": self.task_title,
            "task_description": self.task_description,
            "acceptance_criteria": list(self.acceptance_criteria),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TaskConfig:
        return cls(
            id=str(data["id"]),
            project_path=Path(str(data["project_path"])),
            clone_method=str(data.get("clone_method", "git")),
            task_type=str(data["task_type"]),
            domain=str(data["domain"]),
            language=str(data["language"]),
            prompt_qwen=str(data["prompt_qwen"]),
            prompt_claude=str(data["prompt_claude"]),
            followups_qwen=[str(item) for item in data.get("followups_qwen", [])],
            followups_claude=[str(item) for item in data.get("followups_claude", [])],
            bad_pattern=str(data.get("bad_pattern", "")),
            task_title=str(data.get("task_title", "")),
            task_description=str(data.get("task_description", "")),
            acceptance_criteria=[str(item) for item in data.get("acceptance_criteria", [])],
        )


@dataclass
class BatchConfig:
    delivery_date: str
    runs_root: Path
    max_parallel: int
    tasks: list[TaskConfig]
    qwen: ModelConfig
    claude: ModelConfig
    person_id: str = ""
    github_token: str = ""
    gitee_token: str = ""
    http_proxy: str = ""

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def delivery_dir(self) -> Path:
        return self.base_dir / f"delivery_{self.delivery_date}"

    @property
    def rubrics_dir(self) -> Path:
        return self.base_dir / "rubrics_templates"

    @property
    def docs_dir(self) -> Path:
        return self.base_dir / "docs"

    @property
    def task_manifest_path(self) -> Path:
        return self.delivery_dir / "metadata" / "tasks.json"

    @property
    def state_path(self) -> Path:
        return self.delivery_dir / "pipeline_state.json"

    def score_path(self, task_id: str, model: str) -> Path:
        validate_path_component(task_id, "task_id")
        validate_path_component(model, "model")
        return self.delivery_dir / "scores" / model / f"{model_stem(task_id, model)}.quality.toml"

    def trajectory_path(self, task_id: str, model: str) -> Path:
        validate_path_component(task_id, "task_id")
        validate_path_component(model, "model")
        return self.delivery_dir / "trajectories" / model / f"{model_stem(task_id, model)}.jsonl"

    def resolve_score_path(self, task_id: str, model: str) -> Path:
        """读取用：优先新命名 {model}-{编号}.quality.toml，回退旧命名 {task_id}.quality.toml。"""
        new_path = self.score_path(task_id, model)
        if new_path.exists():
            return new_path
        legacy = self.delivery_dir / "scores" / model / f"{task_id}.quality.toml"
        return legacy if legacy.exists() else new_path

    def resolve_trajectory_path(self, task_id: str, model: str) -> Path:
        """读取用：优先新命名 {model}-{编号}.jsonl，回退旧命名 {task_id}.jsonl。"""
        new_path = self.trajectory_path(task_id, model)
        if new_path.exists():
            return new_path
        legacy = self.delivery_dir / "trajectories" / model / f"{task_id}.jsonl"
        return legacy if legacy.exists() else new_path


def model_stem(task_id: str, model: str) -> str:
    """由任务编号与模型名派生文件名主干：CT-0038 + qwen -> 'qwen-0038'。

    编号取 'CT-' 之后的部分，前缀模型名。任务编号本身保持 CT-XXXX 不变。
    若编号不含 '-'（如测试用的裸 id），则整体作为后缀。
    """
    suffix = task_id.split("-", 1)[1] if "-" in task_id else task_id
    return f"{model}-{suffix}"


def _find_git_bash() -> str:
    """Find git-bash on Windows. Returns path or empty string."""
    if sys.platform != "win32":
        return ""
    if os.environ.get("CLAUDE_CODE_GIT_BASH_PATH"):
        return os.environ["CLAUDE_CODE_GIT_BASH_PATH"]
    bash = shutil.which("bash")
    if bash:
        p = Path(bash).resolve()
        if "Git" in str(p):
            # Prefer Git/bin/bash.exe over Git/usr/bin/bash.exe
            if "usr" in p.parts:
                git_bin = p.parent.parent.parent / "bin" / "bash.exe"
                if git_bin.is_file():
                    return str(git_bin)
            return str(p)
    for candidate in [
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/bash.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Git/bin/bash.exe",
    ]:
        if candidate.is_file():
            return str(candidate)
    return ""


def _extract_host(url: str) -> str:
    """Extract hostname from a URL for NO_PROXY."""
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


SUBMISSION_FIELDNAMES = [
    "id",
    "qwen 本地trajectory",
    "qwen session id",
    "qwen rubrics 人工评分",
    "claude 本地trajectory",
    "claude session id",
    "claude rubrics 人工评分",
    "qwen passrate",
    "claude passrate",
    "任务类型",
    "应用领域",
    "编程语言",
    "命中QwenBad Pattern",
]

THRESHOLD_QWEN_MAX = 0.7
THRESHOLD_CLAUDE_MIN = 0.7
THRESHOLD_RELATIVE_GAIN_MIN = 0.25

# ---------------------------------------------------------------------------
# Scoring dimension definitions (from docs/评分规范.md §7)
# ---------------------------------------------------------------------------

MIN_CRITERIA_COUNT = 7
MAX_CRITERIA_COUNT = 10

# Shared pipeline constants
TRAJECTORY_MAX_CHARS = 50_000
TRAJECTORY_SUMMARY_CHARS = 5_000
SCORING_TIMEOUT = 300
MAX_SCORING_RETRIES = 3
MIN_TRAJECTORY_LINES = 10
MIN_TURNS = 2
MAX_TURNS = 8
MIN_SALVAGE_LINES = 3

# Pipeline stage classification
MODEL_SPECIFIC_STAGES = ("run", "collect", "score")

# Reference dimension pool — used as style examples in AI prompts.
# Criterion names are now customized per-task; validation uses
# is_valid_criterion_name() (snake_case format check) instead.
REFERENCE_CRITERION_NAMES: list[str] = [
    "user_experience_and_interaction",
    "task_planning_and_execution_control",
    "semantic_understanding_and_logical_reasoning",
    "instruction_compliance_and_constraint_adherence",
    "engineering_quality_and_completeness",
    "delivery_completeness_and_usability",
    "architecture_boundaries_and_security_compliance",
    "tool_usage_and_failure_recovery",
    "evidence_grounding_and_trace_fidelity",
    "testing_and_verification_rigor",
    "context_exploration_and_code_navigation",
    "requirements_clarification_and_scope_control",
    "environment_and_dependency_handling",
    "attachment_and_artifact_handling",
    "external_research_and_source_use",
    "custom_tool_and_protocol_compliance",
    "parallel_workflow_coordination",
    "security_privacy_and_secret_handling",
    "maintainability_and_change_minimality",
    "final_response_and_handoff_quality",
]

# Reference descriptions — generic templates used as fallback/examples in AI prompts.
REFERENCE_CRITERION_DESCRIPTIONS: dict[str, str] = {
    "user_experience_and_interaction": "交互过程中的用户体验是否顺畅高效？1分：工作流死锁、循环错误、没有有效输出，完全不可用。2分：推理冗长，反复自我纠错，存在明显逻辑偏差，需要用户频繁干预。3分：存在一些重复和冗余推理，偶尔过度复杂化，但用户仍能等待并获得结果。4分：整体体验良好，偶有冗余或自我修正，但不影响任务完成。5分：推理高效，逻辑清晰，响应顺畅，准确抓住核心意图，输出可信且几乎无优化空间。",
    "task_planning_and_execution_control": "模型规划、拆解、跟踪和执行任务的能力如何？1分：没有规划、没有拆解、没有任务列表、没有跟踪、没有反馈，也没有有效工具使用。2分：规划模糊或不合理，任务拆解无效，跟踪不一致，反馈零散。3分：宏观计划可行，但任务列表不完整或未及时更新，工具使用有明显缺口，反馈偏笼统。4分：规划合理，有任务列表但更新略滞后，存在轻微工具误用，阶段反馈清楚。5分：计划完整且有充分依据，能自动生成任务列表，实时同步进度，工具使用得当，阶段反馈充分，并能主动澄清歧义。",
    "semantic_understanding_and_logical_reasoning": "模型理解意图并进行逻辑推理的准确性如何？1分：完全误解意图，无法解析代码结构，即使有指导也难以执行，只能盲目输出。2分：误读关键信息，只能在指导下部分完成，代码定位或修改存在错误。3分：理解主要意图，但遗漏细节或有轻微偏差，需要人工指导才能达成目标。4分：正确理解意图和约束，代码修改准确，无需用户指导，知识满足任务要求。5分：精准整合上下文和代码库信息，以最小且合适的改动完成目标，无需人工指导，知识覆盖充分。",
    "instruction_compliance_and_constraint_adherence": "模型遵循多重约束并在整个过程中保持这些约束的能力如何？1分：完全忽视复杂约束，按自身默认方式输出，人工纠正时陷入循环或争辩，严重破坏原结构。2分：经过两次纠正后仍不能同时满足多重约束，修了一个忘了另一个，最终输出仍不达标。3分：初始输出忽略关键约束，需要两次明确警告后才勉强满足全部约束。4分：初始输出遗漏1-2个非核心约束，用户一次提示后能立即全部修正。5分：完全遵守所有多重约束，在修复、追加和调试过程中不破坏既有要求，不遗忘限制，也不任意扩大范围。",
    "engineering_quality_and_completeness": "代码输出的完整性和工程化质量如何？1分：缺乏工程质量意识，没有提供测试，或只输出无效伪测试。2分：多次提醒后才尝试测试，测试结构、断言和范围存在明显缺陷。3分：在提醒后补充测试，但只覆盖主流程。4分：大多数场景提供有效测试，测试有效但覆盖或风格有轻微偏差。5分：主动新增或更新测试，覆盖关键路径，符合工程规范，并清楚说明验证方法。",
    "delivery_completeness_and_usability": "最终交付的完整性、可运行性和可用性如何？1分：完全不可运行、输出无关或恶意虚假交付。2分：未实现核心目标，代码不能运行，存在较多虚假交付。3分：主功能可运行，但存在明显缺陷，需要人工修复；有轻微虚假交付。4分：核心功能完整且可用，代码可执行，仅有极少细节遗漏。5分：首次运行即可成功，功能闭环完整，覆盖显式与隐式需求，没有虚假交付。",
    "architecture_boundaries_and_security_compliance": "模型尊重架构边界和安全合规要求的程度如何？1分：任意删除或破坏核心文件，执行未授权高风险操作，多次警告后仍拒绝纠正。2分：频繁越界修改，纠正后仍反复，影响项目安全和进展。3分：做出无关修改或新增冗余依赖，需要强干预但仍可恢复。4分：偶有轻微边界问题，但被提醒后立即停止，未触碰核心资产。5分：严格尊重架构边界，没有未授权操作，对高风险动作会主动确认授权。",
    "tool_usage_and_failure_recovery": "模型选择、调用工具并从失败中恢复的能力如何？1分：几乎不用工具或持续误用工具，失败后无处理。2分：多次使用错误工具或错误参数，重复失败且缺少有效fallback。3分：能使用部分正确工具，但对错误信息利用不足，失败恢复不完整。4分：工具选择基本正确，轻微失败后能及时调整，主流程不受明显影响。5分：工具选择精准，参数符合要求，能根据tool_result快速修正并完成有效验证。",
    "evidence_grounding_and_trace_fidelity": "模型结论是否忠实于真实轨迹证据？1分：大量编造文件、命令、结果或完成状态，与轨迹明显不符。2分：多处依赖最终自述，缺少工具或文件证据支撑。3分：主结论大体有证据，但存在夸大完成度或忽略反证。4分：大多数结论能对应轨迹证据，仅有轻微表述过满。5分：结论、评分和交付说明都能被JSONL、工具结果或文件变更清楚支撑。",
    "testing_and_verification_rigor": "模型验证改动的严谨程度如何？1分：没有任何有效验证，或声称验证但无证据。2分：只做表面命令或伪验证，失败后未处理。3分：运行了部分相关检查，但覆盖不足或未覆盖关键场景。4分：运行了主要测试、构建、lint或smoke test，验证基本可信。5分：验证覆盖关键路径和边界场景，测试结果明确，失败处理和限制说明充分。",
    "context_exploration_and_code_navigation": "模型探索项目上下文和定位代码的能力如何？1分：几乎不读项目上下文，盲目修改或回答。2分：读取方向明显错误，遗漏关键文件或入口。3分：能找到主路径文件，但探索不够聚焦或遗漏相关模块。4分：上下文读取较完整，能定位关键调用链，仅有轻微冗余。5分：高效定位关键文件、调用关系和约束，探索范围充分且不过度。",
    "requirements_clarification_and_scope_control": "模型澄清需求并控制任务范围的能力如何？1分：需求明显不清却直接臆测，或任意扩大范围。2分：只问无关技术细节，不确认关键业务口径。3分：能部分识别不确定点，但澄清不充分或范围控制一般。4分：能识别主要歧义并适度澄清，范围基本可控。5分：准确判断何时该问、何时该做，澄清关键口径并保持改动边界清晰。",
    "environment_and_dependency_handling": "模型处理运行环境、依赖和版本约束的能力如何？1分：忽略环境说明，使用错误解释器或破坏依赖。2分：失败后反复安装或乱改配置，未理解lockfile、README或环境要求。3分：能发现部分环境问题，但诊断和修复不完整。4分：能读取环境说明并选择正确命令或依赖处理方式。5分：准确诊断版本、依赖、解释器、平台差异，并完成可复现验证。",
    "attachment_and_artifact_handling": "模型处理附件、二进制文件和外部产物的能力如何？1分：无视附件或要求用户手动转写。2分：识别附件但不尝试可行解析路径。3分：能处理部分附件，但结果覆盖不完整或缺少fallback。4分：能识别类型并使用合适工具解析，输出基本覆盖关键内容。5分：主动读取、转换、解析附件，失败时有备选路径，最终结构化结果完整可靠。",
    "external_research_and_source_use": "模型使用外部资料和上游信息的能力如何？1分：需要外部资料却完全不查，或编造来源。2分：搜索方向错误，引用不可靠或未落回本地问题。3分：找到部分相关资料，但和本地证据结合不足。4分：能结合官方文档、GitHub issue、PR或release note支撑修复。5分：先收集本地证据，再检索可靠来源，并把外部结论准确转化为本地改动和验证。",
    "custom_tool_and_protocol_compliance": "模型遵守自定义工具、内部协议和skill流程的能力如何？1分：完全忽略schema、skill或协议，使用通用工具硬套。2分：只加载说明但不执行，或参数明显不符合schema。3分：能部分使用正确工具，但流程缺步骤或结果未闭环。4分：基本按schema、skill或协议执行，轻微遗漏不影响主路径。5分：准确读取规则、构造参数、执行真实调用，并汇总结果和后续动作。",
    "parallel_workflow_coordination": "模型拆分并协调并行任务的能力如何？1分：并行边界混乱，导致冲突、遗漏或无汇总。2分：不适合并行却强行并行，或适合并行时完全串行导致明显遗漏。3分：有拆分但边界、写入范围或汇总机制不足。4分：能合理拆分多个模块或信息源，并完成基本汇总。5分：先定义共享规范和写入边界，再并行推进，最后统一整合、冲突检查和验证。",
    "security_privacy_and_secret_handling": "模型处理安全、隐私和敏感信息的能力如何？1分：泄露、写入或滥用密钥，绕过安全校验，造成严重风险。2分：多次触碰敏感数据或削弱权限逻辑，纠正后仍不稳定。3分：能避免明显泄露，但安全边界、日志脱敏或权限处理仍有缺口。4分：基本遵守安全要求，未引入新风险，轻微问题可修正。5分：主动识别凭据、权限、隐私、审计和高风险操作，采取最小权限和安全处理方式。",
    "maintainability_and_change_minimality": "模型改动的可维护性和最小必要性如何？1分：大范围无关改动、重复代码严重或破坏结构。2分：改动过度、抽象混乱、引入不必要依赖。3分：主功能能工作，但结构、命名、重复或边界仍有明显维护成本。4分：改动范围较克制，结构基本清晰，仅有轻微维护性问题。5分：改动最小且贴合现有架构，命名清晰，复用合理，长期维护成本低。",
    "final_response_and_handoff_quality": "模型最终回复和交接质量如何？1分：最终回复无关、误导或缺失关键结果。2分：只说完成但不说明改动、验证或限制。3分：能总结主要改动，但遗漏验证结果、失败原因或后续风险。4分：清楚说明改动、验证和限制，轻微遗漏不影响用户接手。5分：最终回复准确、简洁、可复查，明确列出改动、验证命令、剩余风险和下一步。",
}


def is_valid_criterion_name(name: str) -> bool:
    """Validate criterion name is a well-formed snake_case identifier."""
    return bool(name) and bool(re.match(r'^[a-z][a-z0-9]*(_[a-z0-9]+)*$', name))


# ---------------------------------------------------------------------------
# Input sanitization helpers
# ---------------------------------------------------------------------------

_TASK_ID_RE = re.compile(r'^CT-\d{4}$')
_DELIVERY_DATE_RE = re.compile(r'^\d{8}$')
_WINDOWS_RESERVED = frozenset({
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
})


def validate_task_id(task_id: str) -> str:
    """Validate task_id format (CT-XXXX). Raises ValueError on bad input."""
    if not _TASK_ID_RE.match(task_id):
        raise ValueError(f"Invalid task_id format: {task_id!r} (expected CT-XXXX)")
    return task_id


def validate_delivery_date(date_str: str) -> str:
    """Validate delivery_date format (YYYYMMDD digits only). Raises ValueError on bad input."""
    if not _DELIVERY_DATE_RE.match(date_str):
        raise ValueError(f"Invalid delivery_date format: {date_str!r} (expected YYYYMMDD)")
    return date_str


def validate_session_id(session_id: str) -> str:
    """Validate session_id has no path traversal characters.

    Raises ValueError on bad input. Empty string is allowed (means unknown).
    """
    if not session_id:
        return session_id
    return validate_path_component(session_id, "session_id")


def validate_path_component(value: str, label: str) -> str:
    """Reject path traversal characters in a value used as a path component."""
    if not value:
        return value
    if '..' in value or '/' in value or '\\' in value or '\0' in value:
        raise ValueError(f"Invalid {label}: {value!r} contains path traversal characters")
    # Block Windows reserved device names
    stem = value.split('.')[0].upper()
    if stem in _WINDOWS_RESERVED:
        raise ValueError(f"Invalid {label}: {value!r} is a Windows reserved name")
    return value


def is_safe_clone_url(url: str) -> bool:
    """Check that a git clone URL uses an allowed scheme (https, http, git)."""
    parsed = urlparse(url)
    return parsed.scheme in ('https', 'http', 'git') and bool(parsed.hostname)


def _validate_runs_root(path: Path) -> Path:
    """Validate runs_root is not a system-sensitive directory."""
    resolved = path.resolve()
    _BLOCKED = [Path("/"), Path("C:/"), Path("C:/Windows"), Path("C:/Windows/System32")]
    if sys.platform == "win32":
        _BLOCKED.extend([
            Path(os.environ.get("SYSTEMROOT", "C:/Windows")),
            Path(os.environ.get("WINDIR", "C:/Windows")),
        ])
    else:
        _BLOCKED.extend([Path("/etc"), Path("/usr"), Path("/var"), Path("/root")])
    for blocked in _BLOCKED:
        try:
            if resolved == blocked.resolve():
                raise ValueError(f"runs_root must not be a system directory: {path}")
        except (OSError, ValueError):
            pass
    return path


def build_reference_dimension_table(names: list[str] | None = None) -> str:
    """Build a formatted reference dimension table for AI prompts.

    Args:
        names: If provided, only include these dimensions. Otherwise use
            all REFERENCE_CRITERION_NAMES.
    """
    target = names if names else REFERENCE_CRITERION_NAMES
    lines: list[str] = []
    for name in target:
        desc = REFERENCE_CRITERION_DESCRIPTIONS.get(name, "")
        lines.append(f"- `{name}`: {desc}")
    return "\n".join(lines)

def build_bad_pattern_table() -> str:
    """Build formatted bad pattern reference table for scoring prompts."""
    lines: list[str] = []
    for key, desc in BAD_PATTERN_DESCRIPTIONS.items():
        lines.append(f"- `{key}`: {desc}")
    return "\n".join(lines)


def check_passrate_thresholds(
    task_id: str,
    qwen_pr: float,
    claude_pr: float,
    has_qwen: bool,
    has_claude: bool,
) -> list[str]:
    """Check passrate thresholds and return list of issue strings."""
    issues: list[str] = []
    if has_qwen and qwen_pr >= THRESHOLD_QWEN_MAX:
        issues.append(f"[{task_id}] qwen passrate {qwen_pr:.4f} >= {THRESHOLD_QWEN_MAX}")
    if has_claude and claude_pr <= THRESHOLD_CLAUDE_MIN:
        issues.append(f"[{task_id}] claude passrate {claude_pr:.4f} <= {THRESHOLD_CLAUDE_MIN}")
    if has_claude and has_qwen and claude_pr <= qwen_pr:
        issues.append(f"[{task_id}] claude passrate {claude_pr:.4f} <= qwen {qwen_pr:.4f}")
    if has_claude and has_qwen:
        if qwen_pr > 0:
            relative_gain = (claude_pr - qwen_pr) / qwen_pr
            if relative_gain <= THRESHOLD_RELATIVE_GAIN_MIN:
                issues.append(
                    f"[{task_id}] relative gain {relative_gain:.2%} <= {THRESHOLD_RELATIVE_GAIN_MIN:.0%} "
                    f"(claude={claude_pr:.4f}, qwen={qwen_pr:.4f})"
                )
        else:
            if claude_pr < THRESHOLD_RELATIVE_GAIN_MIN:
                issues.append(
                    f"[{task_id}] qwen=0, claude passrate {claude_pr:.4f} too low "
                    f"(need >= {THRESHOLD_RELATIVE_GAIN_MIN} when qwen=0)"
                )
    return issues


CLAUDE_CODE_VERSION = "2.1.86"

VALID_TASK_TYPES = [
    "bug-fix",
    "feature",
    "enhancement",
    "from_scratch",
    "testing-quality",
    "refactor-maintenance",
    "build-release-config",
    "documentation",
    "code-explanation",
    "security-compliance",
]

SUBMISSION_KEY_MAP: dict[str, str] = {
    "qwen 本地trajectory": "qwen_trajectory",
    "qwen session id": "qwen_session_id",
    "qwen rubrics 人工评分": "qwen_score_path",
    "claude 本地trajectory": "claude_trajectory",
    "claude session id": "claude_session_id",
    "claude rubrics 人工评分": "claude_score_path",
    "qwen passrate": "qwen_passrate",
    "claude passrate": "claude_passrate",
    "任务类型": "task_type",
    "应用领域": "domain",
    "编程语言": "language",
    "命中QwenBad Pattern": "bad_pattern",
}

BAD_PATTERNS = [
    "lazy_shortcut",
    "poor_interaction",
    "github_based",
    "environment_dependency",
    "instruction_follow",
    "attachment_binary",
    "planning_only",
    "macos_development",
    "parallel_tool_usage",
]

BAD_PATTERN_DESCRIPTIONS: dict[str, str] = {
    "lazy_shortcut": "偷懒 - 模型只做核心功能，忽略隐式质量要求",
    "poor_interaction": "交互不通畅 - 模型不与用户确认就直接执行",
    "github_based": "基于GitHub的题目 - 需要web search定位旧版本bug",
    "environment_dependency": "环境依赖 - venv/cuda/python版本等环境陷阱",
    "instruction_follow": "指令follow - 不遵循CLAUDE.md或项目约束",
    "attachment_binary": "附件处理不足 - 不主动处理PDF/zip/图片等附件",
    "planning_only": "只做准备/只写计划 - 不执行实质动作或不使用自定义工具",
    "macos_development": "macOS开发能力不足 - 套用Linux方案",
    "parallel_tool_usage": "并行能力不足 - 串行处理可并行子任务",
}


def build_claude_env(model_config: ModelConfig) -> dict[str, str]:
    """Build environment dict for running `claude -p` subprocesses.

    Sets ANTHROPIC_AUTH_TOKEN (Claude Code's auth), ANTHROPIC_BASE_URL,
    model overrides, and git-bash path. Only passes through a safe subset
    of environment variables to avoid leaking sensitive tokens.
    """
    # Start with a minimal safe subset of the parent environment
    _SAFE_ENV_PREFIXES = (
        "PATH", "SYSTEMROOT", "COMSPEC", "WINDIR",
        "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
        "TEMP", "TMP", "TMPDIR",
        "LANG", "LC_", "LANGUAGE",
        "NO_PROXY", "no_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
        "NODE_", "NPM_",
        "GIT_AUTHOR_", "GIT_COMMITTER_", "GIT_DIR", "GIT_WORK_TREE",
        "PROGRAMFILES", "LOCALAPPDATA", "APPDATA",
        "OTEL_",
        "CLAUDE_CODE_", "CLAUDE_TELEMETRY_",
        "DISABLE_AUTOUPDATER",
    )
    env: dict[str, str] = {}
    for key, val in os.environ.items():
        if any(key == prefix or key.startswith(prefix) for prefix in _SAFE_ENV_PREFIXES):
            env[key] = val
    if model_config.auth_token:
        env["ANTHROPIC_AUTH_TOKEN"] = model_config.auth_token
    env.update({
        "ANTHROPIC_BASE_URL": model_config.base_url,
        "ANTHROPIC_MODEL": model_config.model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_config.model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model_config.model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model_config.model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model_config.model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
    })
    api_host = _extract_host(model_config.base_url)
    no_proxy = env.get("NO_PROXY", env.get("no_proxy", ""))
    if api_host and api_host not in no_proxy:
        entries = [e.strip() for e in no_proxy.split(",") if e.strip()]
        entries.append(api_host)
        env["NO_PROXY"] = ",".join(entries)
        env["no_proxy"] = env["NO_PROXY"]
    git_bash = _find_git_bash()
    if git_bash:
        env.setdefault("CLAUDE_CODE_GIT_BASH_PATH", git_bash)
    return env


def build_validated_env(model_config: ModelConfig) -> dict[str, str]:
    """Validate model config completeness and build environment dict.

    Raises ValueError if auth_token, base_url, or model is missing.
    """
    missing = []
    if not model_config.auth_token:
        missing.append("auth_token")
    if not model_config.base_url:
        missing.append("base_url")
    if not model_config.model:
        missing.append("model")
    if missing:
        raise ValueError(f"Model config incomplete: missing {', '.join(missing)}")
    return build_claude_env(model_config)


def load_env(env_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            env[key] = value
    return env


def load_config(tasks_toml: Path, env_path: Path) -> BatchConfig:
    data = tomllib.loads(tasks_toml.read_text(encoding="utf-8"))
    env = load_env(env_path)

    for key, value in env.items():
        if key.startswith("OTEL_") or key in (
            "NODE_OPTIONS", "CLAUDE_CODE_ENABLE_TELEMETRY", "CLAUDE_TELEMETRY_DEBUG",
        ):
            os.environ.setdefault(key, value)

    batch = data.get("batch", {})

    qwen = ModelConfig(
        auth_token=env.get("QWEN_AUTH_TOKEN", ""),
        base_url=env.get("QWEN_BASE_URL", ""),
        model=env.get("QWEN_MODEL", "qwen3.7-max"),
    )
    claude = ModelConfig(
        auth_token=env.get("CLAUDE_AUTH_TOKEN", ""),
        base_url=env.get("CLAUDE_BASE_URL", ""),
        model=env.get("CLAUDE_MODEL", "claude-opus-4-6-20260205"),
    )

    tasks: list[TaskConfig] = []
    for t in data.get("task", []):
        tasks.append(TaskConfig(
            id=t["id"],
            project_path=Path(t["project_path"]),
            clone_method=t.get("clone_method", "git"),
            task_type=t["task_type"],
            domain=t["domain"],
            language=t["language"],
            prompt_qwen=t["prompt_qwen"],
            prompt_claude=t["prompt_claude"],
            followups_qwen=t.get("followups_qwen", []),
            followups_claude=t.get("followups_claude", []),
            bad_pattern=t.get("bad_pattern", ""),
            task_title=t.get("task_title", ""),
            task_description=t.get("task_description", ""),
            acceptance_criteria=t.get("acceptance_criteria", []),
        ))

    return BatchConfig(
        delivery_date=validate_delivery_date(batch.get("delivery_date", "")),
        runs_root=_validate_runs_root(Path(batch.get("runs_root", str(Path(__file__).resolve().parent.parent / "runs")))),
        max_parallel=int(batch.get("max_parallel", 3)),
        tasks=tasks,
        qwen=qwen,
        claude=claude,
        person_id=batch.get("person_id", ""),
        github_token=env.get("GITHUB_TOKEN", ""),
        gitee_token=env.get("GITEE_TOKEN", ""),
        http_proxy=env.get("HTTP_PROXY", ""),
    )


def select_tasks(tasks: Iterable[TaskConfig], task_ids: list[str] | None = None) -> list[TaskConfig]:
    task_list = list(tasks)
    if not task_ids:
        return task_list

    task_map = {task.id: task for task in task_list}
    missing = [task_id for task_id in task_ids if task_id not in task_map]
    if missing:
        raise ValueError(f"Unknown task IDs: {', '.join(missing)}")
    return [task_map[task_id] for task_id in task_ids]


def load_task_manifest(path: Path) -> list[TaskConfig]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TaskConfig.from_dict(item) for item in data.get("tasks", [])]


def write_task_manifest(path: Path, tasks: Iterable[TaskConfig]) -> None:
    payload = {
        "tasks": [task.to_dict() for task in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def select_delivery_tasks(config: BatchConfig, task_ids: list[str] | None = None) -> list[TaskConfig]:
    manifest_tasks = load_task_manifest(config.task_manifest_path)
    if manifest_tasks:
        return select_tasks(manifest_tasks, task_ids)
    return select_tasks(config.tasks, task_ids)
