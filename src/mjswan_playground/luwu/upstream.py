"""The luwu_mjlab checkout: pinned clone and task ids registered."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from mjswan_playground._compat import (
    collision_defaults,
    legacy_update_assets,
    legacy_xml_actuators,
)
from mjswan_playground._deps import ensure_repo

REPO_URL = "https://github.com/LuwuDynamics/luwu_mjlab.git"
REPO_COMMIT = "447945fd40fb5585271dce4ea8cc82567916032a"

#: robomotic's export of one policy per task, trained with luwu_mjlab at REPO_COMMIT.
HF_REPO_ID = "robomotic/XGO"
HF_REVISION = "1646bc65ed37feff3698a43524a65cb3cd5b6981"

#: Importing it registers every task id.
TASK_PACKAGE = "src.tasks"


def resolve_root() -> Path:
    return ensure_repo(
        name="luwu_mjlab",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker="src/tasks/velocity/velocity_env_cfg.py",
        root_env_var="MJSWAN_LUWU_ROOT",
    )


def register_tasks(root: Path) -> None:
    """Import upstream's task package so mjlab's registry knows its task ids.

    Upstream runs from source as top-level ``src``, so its root must come *first* on
    ``sys.path``: this repo has a ``src/`` of its own that would shadow it.
    """
    if TASK_PACKAGE in sys.modules:
        return
    imported = sys.modules.get("src")
    if imported is not None:
        paths = [str(path) for path in getattr(imported, "__path__", ()) or ()]
        if not any(path.startswith(str(root)) for path in paths):
            raise RuntimeError(
                f"A different top-level 'src' package is already imported ({paths}), "
                f"so upstream's own 'src.*' modules under {root} cannot be reached. "
                "Build this task in a fresh interpreter."
            )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # ponytail: upstream is on mjlab 1.2.0; drop the shims once REPO_COMMIT is on 1.6.
    with collision_defaults(), legacy_update_assets(), legacy_xml_actuators():
        importlib.import_module(TASK_PACKAGE)
