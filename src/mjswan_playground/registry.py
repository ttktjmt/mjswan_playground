"""Task ID -> the module whose ``setup_builder()`` builds it. Imported lazily: a task
drags in mjlab, torch and whatever its upstream needs."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mjswan

_TASKS: dict[str, str] = {
    "asimov": "mjswan_playground.asimov.main",
    "bipedhrl": "mjswan_playground.bipedhrl.main",
    "booster": "mjswan_playground.booster.main",
    "duet": "mjswan_playground.duet.main",
    "husky": "mjswan_playground.husky.main",
    "jumper": "mjswan_playground.jumper.main",
    "leap": "mjswan_playground.leap.main",
    "luwu": "mjswan_playground.luwu.main",
    "microduck": "mjswan_playground.microduck.main",
    "microduckpg": "mjswan_playground.microduckpg.main",
    "msm": "mjswan_playground.msm.main",
    "musclemimic": "mjswan_playground.musclemimic.main",
    "orcs": "mjswan_playground.orcs.main",
    "pacman": "mjswan_playground.pacman.main",
    "parkour": "mjswan_playground.parkour.main",
    "spinkick": "mjswan_playground.spinkick.main",
    "terra": "mjswan_playground.terra.main",
    "unitreerl": "mjswan_playground.unitreerl.main",
    "upkie": "mjswan_playground.upkie.main",
    "wbc": "mjswan_playground.wbc.main",
    "wuji": "mjswan_playground.wuji.main",
}

ALL_TASKS: tuple[str, ...] = tuple(_TASKS)

#: Part ID -> the task module whose ``setup_builder(part)`` builds it: a subset of the
#: task to upload alone. Not tasks: the README and the previews show the whole, and so
#: does the site, but for a task's ``<task>-open`` part, the scenes no non-commercial
#: license covers, which the site and mjswan Cloud carry in the whole's place.
_PARTS: dict[str, str] = {
    **dict.fromkeys(
        ("microduckpg-moves", "microduckpg-parkour", "microduckpg-stilts"),
        "mjswan_playground.microduckpg.main",
    ),
    **dict.fromkeys(("wuji-reorient", "wuji-penspin"), "mjswan_playground.wuji.main"),
    "orcs-open": "mjswan_playground.orcs.main",
}

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
