"""The unitree_rl_mjlab checkout: pinned clone and task ids registered."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml

from mjswan_playground._compat import collision_defaults, legacy_update_assets
from mjswan_playground._deps import ensure_repo

REPO_URL = "https://github.com/unitreerobotics/unitree_rl_mjlab.git"
REPO_COMMIT = "1425b15f73bd4095f0df53709d7c389c3eb9e790"

#: Importing it imports upstream's ``src.tasks``, which registers every task id.
TASK_PACKAGE = "src.tasks.velocity.config.g1"
VELOCITY_DIR = "deploy/robots/g1/config/policy/velocity/v0"
DANCE_DIR = "deploy/robots/g1/config/policy/mimic/dance1_subject2"
POLICY_ONNX = f"{VELOCITY_DIR}/exported/policy.onnx"
DANCE_ONNX = f"{DANCE_DIR}/exported/policy.onnx"
DANCE_CLIP = f"{DANCE_DIR}/params/dance1_subject2.npz"


def resolve_root() -> Path:
    return ensure_repo(
        name="unitree_rl_mjlab",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=POLICY_ONNX,
        root_env_var="MJSWAN_UNITREERL_ROOT",
    )


def deploy_contract(root: Path, policy_dir: str) -> dict:
    """The ``deploy.yaml`` Unitree's controller reads beside a policy."""
    return yaml.safe_load((root / policy_dir / "params" / "deploy.yaml").read_text())


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
    with collision_defaults(), legacy_update_assets():
        importlib.import_module(TASK_PACKAGE)
