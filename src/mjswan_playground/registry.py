"""Task ID -> the module whose ``setup_builder()`` builds it. Imported lazily: a task
drags in mjlab, torch and whatever its upstream needs."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mjswan

_TASKS: dict[str, str] = {
    "husky-skater": "mjswan_playground.husky_skater.main",
    "wbc-g1": "mjswan_playground.wbc_g1.main",
}

ALL_TASKS: tuple[str, ...] = tuple(_TASKS)


def load(task_id: str) -> "mjswan.Builder":
    """Return the configured builder for ``task_id``, fetching its assets if needed."""
    if task_id not in _TASKS:
        raise KeyError(f"Unknown task {task_id!r}. Available: {', '.join(ALL_TASKS)}")
    return importlib.import_module(_TASKS[task_id]).setup_builder()
