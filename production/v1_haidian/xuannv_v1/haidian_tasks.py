from __future__ import annotations

from typing import Any


def run_task(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError("run_task 将在 Task 3 实现。")


def run_all_tasks(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
    raise NotImplementedError("run_all_tasks 将在 Task 3 实现。")
