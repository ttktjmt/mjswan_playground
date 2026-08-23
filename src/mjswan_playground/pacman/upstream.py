"""Access to the PAC-MAN checkout both of this task's scenes build from.

Three things come out of it: a pinned checkout, its mjlab task ids registered, and the
frozen articulation contract its ONNX checkpoints were exported against.
"""

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
#: The frozen, mjlab-free copy of the trained articulation — joint order, rest pose,
#: action scale and PD gains — that the ONNX checkpoints were exported against, and that
#: upstream's own test re-derives from ``g1_constants.py`` on every run.
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

    The repo runs from source under the top-level name ``src`` — it is never installed —
    so its root has to reach ``sys.path``, and reach it *first*: this repository has a
    ``src/`` of its own, and a namespace package bound to that one answers every
    ``src.*`` import upstream makes with nothing.
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
    """``DEPLOY_CONSTANTS``, loaded by path rather than imported.

    It lives in a second uv project (Python 3.8, no mjlab) whose directory is not a
    package, so there is no import path to it — and its bare module name would collide
    with whatever else claims it.
    """
    path = root / DEPLOY_CONSTANTS
    spec = importlib.util.spec_from_file_location("_pacman_deploy_constants", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load the deployed contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
