"""Task distribution data and weighted sampling.

Contains the 225-row distribution table and sampling functions for
auto-generating tasks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class TaskSlot:
    task_type: str
    domain: str
    language: str
    weight: float


DISTRIBUTION: list[TaskSlot] = [
    TaskSlot("bug-fix", "web_frontend", "ts", 10.0),
    TaskSlot("bug-fix", "mobile_app", "other", 9.7),
    TaskSlot("bug-fix", "web_frontend", "js", 9.5),
    TaskSlot("bug-fix", "backend_service", "java", 9.4),
    TaskSlot("bug-fix", "data_engineering", "java", 9.4),
    TaskSlot("bug-fix", "devtools_test", "python", 9.4),
    TaskSlot("feature", "backend_service", "java", 9.2),
    TaskSlot("bug-fix", "game_dev", "other", 9.1),
    TaskSlot("bug-fix", "mobile_app", "java", 9.0),
    TaskSlot("bug-fix", "web_frontend", "html/css", 8.7),
    TaskSlot("feature", "web_frontend", "js", 8.6),
    TaskSlot("bug-fix", "backend_service", "c", 8.2),
    TaskSlot("bug-fix", "database_storage", "python", 8.2),
    TaskSlot("bug-fix", "backend_service", "go", 8.0),
    TaskSlot("bug-fix", "backend_service", "python", 7.9),
    TaskSlot("feature", "data_engineering", "python", 7.9),
    TaskSlot("refactor-maintenance", "ai_ml", "python", 7.9),
    TaskSlot("bug-fix", "data_engineering", "python", 7.8),
    TaskSlot("feature", "web_frontend", "ts", 7.7),
    TaskSlot("bug-fix", "ai_ml", "python", 7.6),
    TaskSlot("code-explanation", "mobile_app", "other", 7.5),
    TaskSlot("feature", "devops_infrastructure", "python", 7.2),
    TaskSlot("feature", "mobile_app", "other", 7.2),
    TaskSlot("testing-quality", "backend_service", "java", 7.1),
    TaskSlot("bug-fix", "game_dev", "lua", 7.0),
    TaskSlot("code-explanation", "data_engineering", "python", 7.0),
    TaskSlot("code-explanation", "backend_service", "java", 7.0),
    TaskSlot("feature", "web_frontend", "html/css", 7.0),
    TaskSlot("build-release-config", "devops_infrastructure", "shell", 6.9),
    TaskSlot("feature", "ai_ml", "python", 6.8),
    TaskSlot("bug-fix", "backend_service", "rust", 6.8),
    TaskSlot("refactor-maintenance", "backend_service", "java", 6.7),
    TaskSlot("enhancement", "graphics_media", "python", 6.6),
    TaskSlot("bug-fix", "devops_infrastructure", "shell", 6.6),
    TaskSlot("from_scratch", "backend_service", "java", 6.5),
    TaskSlot("enhancement", "mobile_app", "other", 6.4),
    TaskSlot("bug-fix", "desktop_gui", "ts", 6.4),
    TaskSlot("feature", "backend_service", "python", 6.4),
    TaskSlot("code-explanation", "data_engineering", "sql", 6.2),
    TaskSlot("bug-fix", "devops_infrastructure", "python", 6.2),
    TaskSlot("enhancement", "web_frontend", "ts", 6.2),
    TaskSlot("feature", "backend_service", "go", 6.2),
    TaskSlot("enhancement", "web_frontend", "js", 6.1),
    TaskSlot("build-release-config", "devops_infrastructure", "c", 6.1),
    TaskSlot("bug-fix", "graphics_media", "python", 6.1),
    TaskSlot("documentation", "backend_service", "java", 6.1),
    TaskSlot("enhancement", "backend_service", "java", 6.1),
    TaskSlot("testing-quality", "backend_service", "c++", 6.0),
    TaskSlot("bug-fix", "devtools_test", "java", 6.0),
    TaskSlot("feature", "business_logic", "python", 6.0),
    TaskSlot("feature", "devops_infrastructure", "shell", 5.9),
    TaskSlot("feature", "mobile_app", "ts", 5.9),
    TaskSlot("enhancement", "backend_service", "python", 5.9),
    TaskSlot("bug-fix", "embedded_system", "c++", 5.9),
    TaskSlot("bug-fix", "data_engineering", "sql", 5.8),
    TaskSlot("feature", "backend_service", "c", 5.8),
    TaskSlot("enhancement", "data_engineering", "python", 5.8),
    TaskSlot("testing-quality", "web_frontend", "js", 5.7),
    TaskSlot("documentation", "web_frontend", "js", 5.7),
    TaskSlot("refactor-maintenance", "backend_service", "c", 5.5),
    TaskSlot("testing-quality", "backend_service", "go", 5.5),
    TaskSlot("refactor-maintenance", "web_frontend", "ts", 5.5),
    TaskSlot("bug-fix", "desktop_gui", "lua", 5.5),
    TaskSlot("feature", "mobile_app", "java", 5.5),
    TaskSlot("bug-fix", "backend_service", "ts", 5.5),
    TaskSlot("code-explanation", "backend_service", "c++", 5.5),
    TaskSlot("refactor-maintenance", "desktop_gui", "ts", 5.5),
    TaskSlot("feature", "web_frontend", "python", 5.5),
    TaskSlot("feature", "game_dev", "shell", 5.5),
    TaskSlot("bug-fix", "web_frontend", "python", 5.4),
    TaskSlot("refactor-maintenance", "backend_service", "python", 5.4),
    TaskSlot("from_scratch", "web_frontend", "html/css", 5.4),
    TaskSlot("feature", "data_engineering", "sql", 5.4),
    TaskSlot("bug-fix", "graphics_media", "c++", 5.4),
    TaskSlot("enhancement", "mobile_app", "dart", 5.3),
    TaskSlot("feature", "graphics_media", "python", 5.3),
    TaskSlot("bug-fix", "mobile_app", "c", 5.3),
    TaskSlot("feature", "docs_knowledge", "python", 5.2),
    TaskSlot("feature", "backend_service", "c++", 5.1),
    TaskSlot("testing-quality", "devtools_test", "c", 5.1),
    TaskSlot("bug-fix", "mobile_app", "swift", 5.0),
    TaskSlot("code-explanation", "web_frontend", "ts", 5.0),
    TaskSlot("feature", "devtools_test", "python", 4.9),
    TaskSlot("from_scratch", "ai_ml", "python", 4.9),
    TaskSlot("testing-quality", "ai_ml", "python", 4.8),
    TaskSlot("refactor-maintenance", "web_frontend", "js", 4.8),
    TaskSlot("testing-quality", "mobile_app", "java", 4.8),
    TaskSlot("bug-fix", "pkg_manager_cli", "ts", 4.7),
    TaskSlot("testing-quality", "backend_service", "c", 4.7),
    TaskSlot("from_scratch", "web_frontend", "ts", 4.7),
    TaskSlot("bug-fix", "devtools_test", "ts", 4.7),
    TaskSlot("testing-quality", "web_frontend", "ts", 4.7),
    TaskSlot("testing-quality", "devtools_test", "python", 4.7),
    TaskSlot("refactor-maintenance", "backend_service", "go", 4.7),
    TaskSlot("enhancement", "ai_ml", "python", 4.6),
    TaskSlot("refactor-maintenance", "mobile_app", "java", 4.6),
    TaskSlot("testing-quality", "data_engineering", "python", 4.5),
    TaskSlot("bug-fix", "devtools_test", "js", 4.5),
    TaskSlot("code-explanation", "database_storage", "sql", 4.5),
    TaskSlot("feature", "web_frontend", "shell", 4.5),
    TaskSlot("testing-quality", "backend_service", "python", 4.4),
    TaskSlot("feature", "devtools_test", "ts", 4.4),
    TaskSlot("documentation", "ai_ml", "python", 4.4),
    TaskSlot("bug-fix", "backend_service", "c++", 4.4),
    TaskSlot("bug-fix", "mobile_app", "c++", 4.3),
    TaskSlot("enhancement", "data_engineering", "sql", 4.3),
    TaskSlot("feature", "backend_service", "js", 4.3),
    TaskSlot("code-explanation", "web_frontend", "js", 4.3),
    TaskSlot("code-explanation", "embedded_system", "c", 4.2),
    TaskSlot("feature", "game_dev", "lua", 4.2),
    TaskSlot("feature", "backend_service", "other", 4.2),
    TaskSlot("code-explanation", "ai_ml", "python", 4.2),
    TaskSlot("from_scratch", "web_frontend", "js", 4.2),
    TaskSlot("code-explanation", "devops_infrastructure", "python", 4.1),
    TaskSlot("bug-fix", "mobile_app", "dart", 4.1),
    TaskSlot("refactor-maintenance", "mobile_app", "kotlin", 4.1),
    TaskSlot("feature", "devops_infrastructure", "js", 4.1),
    TaskSlot("bug-fix", "devops_infrastructure", "js", 4.1),
    TaskSlot("testing-quality", "devops_infrastructure", "python", 4.1),
    TaskSlot("testing-quality", "devtools_test", "ts", 4.1),
    TaskSlot("from_scratch", "backend_service", "python", 4.1),
    TaskSlot("feature", "backend_service", "shell", 4.0),
    TaskSlot("enhancement", "web_frontend", "html/css", 4.0),
    TaskSlot("feature", "mobile_app", "kotlin", 4.0),
    TaskSlot("enhancement", "backend_service", "go", 4.0),
    TaskSlot("code-explanation", "backend_service", "python", 4.0),
    TaskSlot("documentation", "docs_knowledge", "java", 3.9),
    TaskSlot("enhancement", "backend_service", "c", 3.9),
    TaskSlot("bug-fix", "docs_knowledge", "python", 3.9),
    TaskSlot("bug-fix", "game_dev", "ts", 3.9),
    TaskSlot("feature", "data_engineering", "js", 3.9),
    TaskSlot("build-release-config", "devops_infrastructure", "python", 3.9),
    TaskSlot("enhancement", "devtools_test", "python", 3.8),
    TaskSlot("code-explanation", "backend_service", "go", 3.8),
    TaskSlot("code-explanation", "mobile_app", "ts", 3.8),
    TaskSlot("code-explanation", "game_dev", "lua", 3.8),
    TaskSlot("documentation", "docs_knowledge", "python", 3.8),
    TaskSlot("enhancement", "devops_infrastructure", "python", 3.8),
    TaskSlot("enhancement", "business_logic", "python", 3.8),
    TaskSlot("code-explanation", "ai_ml", "c++", 3.7),
    TaskSlot("bug-fix", "mobile_app", "kotlin", 3.7),
    TaskSlot("refactor-maintenance", "devops_infrastructure", "shell", 3.7),
    TaskSlot("code-explanation", "backend_service", "ts", 3.6),
    TaskSlot("code-explanation", "devops_infrastructure", "shell", 3.6),
    TaskSlot("security-compliance", "backend_service", "java", 3.6),
    TaskSlot("refactor-maintenance", "devtools_test", "python", 3.6),
    TaskSlot("feature", "backend_service", "ts", 3.6),
    TaskSlot("refactor-maintenance", "mobile_app", "other", 3.5),
    TaskSlot("feature", "pkg_manager_cli", "ts", 3.5),
    TaskSlot("from_scratch", "devtools_test", "ts", 3.5),
    TaskSlot("testing-quality", "data_engineering", "sql", 3.5),
    TaskSlot("build-release-config", "backend_service", "java", 3.4),
    TaskSlot("code-explanation", "mobile_app", "java", 3.4),
    TaskSlot("code-explanation", "data_engineering", "shell", 3.4),
    TaskSlot("documentation", "data_engineering", "sql", 3.4),
    TaskSlot("bug-fix", "business_logic", "python", 3.4),
    TaskSlot("code-explanation", "devtools_test", "ts", 3.4),
    TaskSlot("testing-quality", "devops_infrastructure", "shell", 3.4),
    TaskSlot("bug-fix", "data_engineering", "shell", 3.4),
    TaskSlot("feature", "graphics_media", "shell", 3.3),
    TaskSlot("documentation", "docs_knowledge", "html/css", 3.3),
    TaskSlot("enhancement", "devops_infrastructure", "shell", 3.3),
    TaskSlot("code-explanation", "web_frontend", "html/css", 3.3),
    TaskSlot("feature", "desktop_gui", "shell", 3.3),
    TaskSlot("enhancement", "mobile_app", "java", 3.2),
    TaskSlot("feature", "data_engineering", "shell", 3.2),
    TaskSlot("feature", "pkg_manager_cli", "shell", 3.2),
    TaskSlot("from_scratch", "data_engineering", "python", 3.2),
    TaskSlot("code-explanation", "devops_infrastructure", "java", 3.2),
    TaskSlot("enhancement", "web_frontend", "python", 3.2),
    TaskSlot("bug-fix", "backend_service", "js", 3.2),
    TaskSlot("code-explanation", "desktop_gui", "c++", 3.1),
    TaskSlot("code-explanation", "pkg_manager_cli", "ts", 3.1),
    TaskSlot("code-explanation", "operating_system", "c", 3.1),
    TaskSlot("enhancement", "devtools_test", "js", 3.0),
    TaskSlot("documentation", "data_engineering", "python", 3.0),
    TaskSlot("documentation", "backend_service", "c++", 3.0),
    TaskSlot("refactor-maintenance", "devops_infrastructure", "python", 3.0),
    TaskSlot("code-explanation", "backend_service", "js", 3.0),
    TaskSlot("enhancement", "docs_knowledge", "html/css", 3.0),
    TaskSlot("code-explanation", "database_storage", "c++", 3.0),
    TaskSlot("code-explanation", "devtools_test", "shell", 3.0),
    TaskSlot("code-explanation", "backend_service", "kotlin", 2.9),
    TaskSlot("refactor-maintenance", "backend_service", "c++", 2.9),
    TaskSlot("feature", "embedded_system", "c", 2.9),
    TaskSlot("code-explanation", "backend_service", "sql", 2.8),
    TaskSlot("code-explanation", "mobile_app", "kotlin", 2.8),
    TaskSlot("testing-quality", "devops_infrastructure", "go", 2.8),
    TaskSlot("from_scratch", "devtools_test", "python", 2.8),
    TaskSlot("documentation", "backend_service", "python", 2.7),
    TaskSlot("code-explanation", "business_logic", "python", 2.7),
    TaskSlot("build-release-config", "devops_infrastructure", "java", 2.7),
    TaskSlot("feature", "database_storage", "sql", 2.7),
    TaskSlot("feature", "backend_service", "sql", 2.6),
    TaskSlot("refactor-maintenance", "data_engineering", "java", 2.6),
    TaskSlot("enhancement", "backend_service", "c++", 2.5),
    TaskSlot("code-explanation", "devtools_test", "python", 2.5),
    TaskSlot("feature", "devtools_test", "js", 2.5),
    TaskSlot("build-release-config", "devtools_test", "shell", 2.5),
    TaskSlot("code-explanation", "backend_service", "other", 2.5),
    TaskSlot("code-explanation", "devops_infrastructure", "sql", 2.4),
    TaskSlot("refactor-maintenance", "data_engineering", "python", 2.4),
    TaskSlot("build-release-config", "devops_infrastructure", "c++", 2.4),
    TaskSlot("feature", "devtools_test", "shell", 2.3),
    TaskSlot("documentation", "web_frontend", "html/css", 2.3),
    TaskSlot("code-explanation", "backend_service", "c", 2.3),
    TaskSlot("code-explanation", "operating_system", "shell", 2.2),
    TaskSlot("code-explanation", "devops_infrastructure", "go", 2.2),
    TaskSlot("documentation", "devops_infrastructure", "shell", 2.2),
    TaskSlot("documentation", "devops_infrastructure", "python", 2.1),
    TaskSlot("feature", "mobile_app", "shell", 2.1),
    TaskSlot("refactor-maintenance", "mobile_app", "dart", 2.1),
    TaskSlot("documentation", "web_frontend", "ts", 2.0),
    TaskSlot("code-explanation", "backend_service", "shell", 1.9),
    TaskSlot("documentation", "mobile_app", "java", 1.8),
    TaskSlot("enhancement", "database_storage", "sql", 1.7),
    TaskSlot("code-explanation", "data_engineering", "java", 1.7),
    TaskSlot("build-release-config", "web_frontend", "ts", 1.6),
    TaskSlot("enhancement", "data_engineering", "java", 1.6),
    TaskSlot("code-explanation", "ai_ml", "java", 1.5),
    TaskSlot("code-explanation", "devtools_test", "js", 1.5),
    TaskSlot("build-release-config", "pkg_manager_cli", "shell", 1.3),
    TaskSlot("code-explanation", "mobile_app", "shell", 1.2),
    TaskSlot("build-release-config", "devops_infrastructure", "js", 1.1),
    TaskSlot("build-release-config", "devops_infrastructure", "ts", 1.0),
]


def sample_slots(
    count: int,
    domain: str | None = None,
    language: str | None = None,
    task_type: str | None = None,
    exclude_indices: set[int] | None = None,
) -> list[tuple[int, TaskSlot]]:
    """Weighted random sample without replacement from the distribution table.

    Returns list of (original_index, TaskSlot) tuples.
    """
    exclude = exclude_indices or set()
    candidates: list[tuple[int, TaskSlot]] = []
    for i, slot in enumerate(DISTRIBUTION):
        if i in exclude:
            continue
        if domain and slot.domain != domain:
            continue
        if language and slot.language != language:
            continue
        if task_type and slot.task_type != task_type:
            continue
        candidates.append((i, slot))

    if not candidates:
        return []

    actual_count = min(count, len(candidates))
    remaining = list(range(len(candidates)))
    weights = [slot.weight for _, slot in candidates]
    result: list[tuple[int, TaskSlot]] = []

    for _ in range(actual_count):
        chosen = random.choices(remaining, weights=[weights[j] for j in remaining], k=1)[0]
        result.append(candidates[chosen])
        remaining.remove(chosen)

    return result


def expand_slot_for_batch(
    slot: TaskSlot,
    batch_size: int,
    exclude_indices: set[int] | None = None,
) -> list[TaskSlot]:
    """Expand a slot into batch_size slots with distinct task_types.

    Returns a list starting with the original slot's task_type, followed by
    other task_types available for the same (domain, language). If not enough
    distinct types exist, duplicates the original type to fill.
    """
    exclude = exclude_indices or set()
    same_dl = [
        s for i, s in enumerate(DISTRIBUTION)
        if i not in exclude
        and s.domain == slot.domain
        and s.language == slot.language
        and s.task_type != slot.task_type
    ]

    seen_types: set[str] = {slot.task_type}
    extras: list[TaskSlot] = []
    for s in same_dl:
        if s.task_type not in seen_types:
            extras.append(TaskSlot(s.task_type, slot.domain, slot.language, s.weight))
            seen_types.add(s.task_type)
        if len(extras) >= batch_size - 1:
            break

    result = [slot] + extras
    while len(result) < batch_size:
        result.append(slot)
    return result[:batch_size]
