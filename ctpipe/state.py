"""Pipeline state tracking via JSON ledger.

Stored at delivery_YYYYMMDD/pipeline_state.json.
Supports idempotent re-runs: each subcommand skips completed tasks.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class PipelineState:
    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, dict[str, Any]] = {}
        self._batch_depth = 0
        self._lock = threading.Lock()
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    def reload(self) -> None:
        """Re-read state from disk (useful after other stages have written to it)."""
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

    @contextmanager
    def batch(self):
        with self._lock:
            self._batch_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._batch_depth -= 1
                if self._batch_depth == 0:
                    self.save()

    def _task(self, task_id: str) -> dict[str, Any]:
        if task_id not in self._data:
            self._data[task_id] = {}
        return self._data[task_id]

    def get(self, task_id: str, stage: str, model: str | None = None) -> dict[str, Any]:
        with self._lock:
            task = self._data.get(task_id, {})
            stage_data = task.get(stage, {})
            if model:
                return stage_data.get(model, {})
            return stage_data

    def set(self, task_id: str, stage: str, model: str | None = None, **data: Any) -> None:
        with self._lock:
            task = self._task(task_id)
            if model:
                if stage not in task:
                    task[stage] = {}
                if "status" in data:
                    task[stage][model] = data
                else:
                    task[stage][model] = {**task[stage].get(model, {}), **data}
            else:
                if "status" in data:
                    task[stage] = data
                else:
                    task[stage] = {**task.get(stage, {}), **data}
            if self._batch_depth == 0:
                self.save()

    def is_done(self, task_id: str, stage: str, model: str | None = None) -> bool:
        with self._lock:
            task = self._data.get(task_id, {})
            stage_data = task.get(stage, {})
            info = stage_data.get(model, {}) if model else stage_data
            return info.get("status") == "done"

    def reset(self, task_id: str, stage: str, model: str | None = None) -> bool:
        with self._lock:
            task = self._data.get(task_id)
            if not task or stage not in task:
                return False
            if model:
                stage_data = task[stage]
                if isinstance(stage_data, dict) and model in stage_data:
                    del stage_data[model]
                    if self._batch_depth == 0:
                        self.save()
                    return True
                return False
            del task[stage]
            if self._batch_depth == 0:
                self.save()
            return True

    @property
    def all_task_ids(self) -> list[str]:
        return [k for k in self._data.keys() if not k.startswith("_")]
