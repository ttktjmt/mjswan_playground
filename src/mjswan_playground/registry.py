"""Task ID -> the module whose ``setup_builder()`` builds it. Imported lazily: a task
drags in mjlab, torch and whatever its upstream needs."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mjswan

_TASKS: dict[str, str] = {
    "husky": "mjswan_playground.husky.main",
    "microduck": "mjswan_playground.microduck.main",
    "musclemimic": "mjswan_playground.musclemimic.main",
    "pacman": "mjswan_playground.pacman.main",
    "wbc": "mjswan_playground.wbc.main",
}

ALL_TASKS: tuple[str, ...] = tuple(_TASKS)


def load(task_id: str) -> "mjswan.Builder":
    """Return the configured builder for ``task_id``, fetching its assets if needed."""
    if task_id not in _TASKS:
        raise KeyError(f"Unknown task {task_id!r}. Available: {', '.join(ALL_TASKS)}")
    return importlib.import_module(_TASKS[task_id]).setup_builder()
