"""The PAC-MAN checkout: pinned clone, task ids registered, deployed contract."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from mjswan_playground._deps import ensure_repo

REPO_URL = "https://github.com/lzyang2000/perceptive_cbf_rl.git"
REPO_COMMIT = "55c47eb32d8602f078fe0761e179e6b65d3656ac"

#: Importing it registers every task the repo defines with mjlab's registry.
TASK_PACKAGE = "src.tasks.amp_loco.config.g1"
#: Frozen articulation (joint order, rest pose, action scale, PD gains) the ONNX
#: checkpoints were exported against.
DEPLOY_CONSTANTS = "deploy/common/g1_deploy_constants.py"
MARKER = "deploy/ckpts/walk_policy.onnx"


def resolve_root() -> Path:
    return ensure_repo(
        name="perceptive_cbf_rl",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=MARKER,
        root_env_var="MJSWAN_PACMAN_ROOT",
    )


def register_tasks(root: Path) -> None:
    """Import upstream's task package so mjlab's registry knows its task ids.

    Upstream runs from source as top-level ``src``, so its root must come *first* on
    ``sys.path``: this repo has a ``src/`` of its own that would shadow it.
    """
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
    importlib.import_module(TASK_PACKAGE)


def deployed_contract(root: Path) -> ModuleType:
    """``DEPLOY_CONSTANTS``, loaded by path: its directory is not a package."""
    path = root / DEPLOY_CONSTANTS
    spec = importlib.util.spec_from_file_location("_pacman_deploy_constants", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load the deployed contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
