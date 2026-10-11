"""Task ID -> the module whose ``setup_builder()`` builds it. Imported lazily: a task
drags in mjlab, torch and whatever its upstream needs."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mjswan

_TASKS: dict[str, str] = {
    "bipedhrl": "mjswan_playground.bipedhrl.main",
    "duet": "mjswan_playground.duet.main",
    "husky": "mjswan_playground.husky.main",
    "jumper": "mjswan_playground.jumper.main",
    "microduck": "mjswan_playground.microduck.main",
    "microduckpg": "mjswan_playground.microduckpg.main",
    "musclemimic": "mjswan_playground.musclemimic.main",
    "pacman": "mjswan_playground.pacman.main",
    "spinkick": "mjswan_playground.spinkick.main",
    "unitreerl": "mjswan_playground.unitreerl.main",
    "upkie": "mjswan_playground.upkie.main",
    "wbc": "mjswan_playground.wbc.main",
}

ALL_TASKS: tuple[str, ...] = tuple(_TASKS)

#: Part ID -> the task module whose ``setup_builder(part)`` builds it: a subset of the
#: task small enough to upload alone. Not tasks: the README, the site and the previews
#: show the whole.
_PARTS: dict[str, str] = dict.fromkeys(
    ("microduck-moves", "microduck-parkour", "microduck-stilts"),
    "mjswan_playground.microduckpg.main",
)

ALL_PARTS: tuple[str, ...] = tuple(_PARTS)


def load(task_id: str) -> "mjswan.Builder":
    """Return the configured builder for ``task_id``, a task or a part of one, fetching
    its assets if needed."""
    if task_id in _TASKS:
        return importlib.import_module(_TASKS[task_id]).setup_builder()
    if task_id in _PARTS:
        return importlib.import_module(_PARTS[task_id]).setup_builder(task_id)
    known = ", ".join(ALL_TASKS + ALL_PARTS)
    raise KeyError(f"Unknown task {task_id!r}. Available: {known}")
