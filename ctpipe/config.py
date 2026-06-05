"""Load tasks.toml and .env into typed configuration objects."""

from __future__ import annotations

import json
import os
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
        )


@dataclass
class BatchConfig:
    delivery_date: str
    runs_root: Path
    max_parallel: int
    tasks: list[TaskConfig]
    qwen: ModelConfig
    claude: ModelConfig
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
}


def build_claude_env(model_config: ModelConfig) -> dict[str, str]:
    """Build environment dict for running `claude -p` subprocesses.

    Sets ANTHROPIC_AUTH_TOKEN (Claude Code's auth), ANTHROPIC_BASE_URL,
    model overrides, and git-bash path. Proxy vars are removed so the
    API endpoint is accessed directly.
    """
    env = os.environ.copy()
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
    })
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(key, None)
    git_bash = _find_git_bash()
    if git_bash:
        env.setdefault("CLAUDE_CODE_GIT_BASH_PATH", git_bash)
    return env


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
        ))

    return BatchConfig(
        delivery_date=batch.get("delivery_date", ""),
        runs_root=Path(batch.get("runs_root", str(Path(__file__).resolve().parent.parent / "runs"))),
        max_parallel=int(batch.get("max_parallel", 3)),
        tasks=tasks,
        qwen=qwen,
        claude=claude,
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
