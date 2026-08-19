"""WBC-Mjlab whole-body motion-tracking demo.

Nothing here is hand-assembled: ``add_scene_mjlab`` defaults every term off the task's
own env config, and this module supplies only the trained policy and the clips. See
``README.md``.
"""

from __future__ import annotations

from pathlib import Path

import mjlab.tasks  # noqa: F401  (registers the mjlab task ids)
import mjswan
import onnx
import yaml
from mjlab.tasks.registry import load_env_cfg

from mjswan_playground._deps import ensure_repo

DEPLOY_REPO_URL = "https://github.com/wbc-mjlab/wbc-g1-deploy.git"
DEPLOY_REPO_COMMIT = "6dabf86fddc2b7b429b09e74999732fcde3441f9"

TASK_ID = "Wbc-G1"
POLICY_ONNX = "config/policy/wbc/params/policy.onnx"
POLICY_CONFIG = "config/policy/wbc/params/config.yaml"
CLIP_DIR = "config/clips"
CLIP_MANIFEST = "config/clips/manifest.yaml"


def _resolve_deploy_root() -> Path:
    return ensure_repo(
        name="wbc-g1-deploy",
        url=DEPLOY_REPO_URL,
        commit=DEPLOY_REPO_COMMIT,
        marker=POLICY_ONNX,
        root_env_var="MJSWAN_WBC_DEPLOY_ROOT",
    )


def _clip_names(root: Path) -> list[tuple[str, Path]]:
    """The clips the deploy manifest enables, in its order, plus its default first."""
    manifest = yaml.safe_load((root / CLIP_MANIFEST).read_text())
    default = manifest.get("default")
    clips = [
        (entry["name"], root / CLIP_DIR / entry["file"]) for entry in manifest["clips"]
    ]
    clips.sort(key=lambda item: item[0] != default)
    return clips


def setup_builder() -> mjswan.Builder:
    deploy = _resolve_deploy_root()
    contract = yaml.safe_load((deploy / POLICY_CONFIG).read_text())

    env_cfg = load_env_cfg(TASK_ID, play=True)
    # Training-only events the deploy runtime does not apply either — see README.
    for training_only in ("assistive_wrench", "pull_robot"):
        env_cfg.events.pop(training_only, None)

    builder = mjswan.Builder()
    project = builder.add_project(name="WBC-Mjlab G1")
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)

    joint_names = [f"robot/{name}" for name in contract["joint_names"]]
    policy = scene.add_policy(
        name="WBC Tracking",
        policy=onnx.load(str(deploy / POLICY_ONNX)),
        policy_joint_names=joint_names,
        default_joint_pos=[float(v) for v in contract["default_joint_pos"]],
        default=True,
    )

    anchor = contract["tracking"]["anchor_body_name"]
    # ponytail: `add_motion` takes the tracked bodies explicitly; drop this line once
    # mjswan resolves them off the scene's env cfg (0.9.1 does so only for W&B motions).
    body_names = env_cfg.commands["motion"].body_names
    for index, (name, path) in enumerate(_clip_names(deploy)):
        policy.add_motion(
            name=name,
            source=str(path),
            fps=50.0,
            anchor_body_name=anchor,
            body_names=body_names,
            dataset_joint_names=list(contract["joint_names"]),
            default=index == 0,
        )

    return builder
