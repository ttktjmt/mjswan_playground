"""Unitree G1 walking to velocity commands and dancing a LAFAN1 clip. See ``README.md``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mjswan
import onnx
from mjlab.tasks.registry import load_env_cfg
from mjswan.mjlab.onnx_meta import read_mjlab_metadata

from mjswan_playground._gait_clock import clock_phase

from . import upstream

TASK_ID = "Unitree-G1-Flat"
DANCE_TASK_ID = "Unitree-G1-Tracking-No-State-Estimation"


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    builder = mjswan.Builder()
    project = builder.add_project(name="unitree_rl_mjlab", license=root / "LICENCE")
    joint_names = _add_walk(project, root)
    _add_dance(project, root, joint_names)
    return builder


def _add_walk(project: Any, root: Path) -> list[str]:
    policy = onnx.load(str(root / upstream.POLICY_ONNX))
    contract = read_mjlab_metadata(policy)
    env_cfg = load_env_cfg(TASK_ID, play=True)
    clock_phase(env_cfg)
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
    joint_names = [f"robot/{name}" for name in contract.joint_names]
    scene.add_policy(
        name="v0",
        policy=policy,
        policy_joint_names=joint_names,
        default_joint_pos=contract.default_joint_pos,
    )
    return joint_names


def _add_dance(project: Any, root: Path, joint_names: list[str]) -> None:
    # The export carries no metadata; both deploy.yaml files map the motors alike, so the
    # dance takes the walk's joint order.
    contract = upstream.deploy_contract(root, upstream.DANCE_DIR)
    walk = upstream.deploy_contract(root, upstream.VELOCITY_DIR)
    if contract["joint_ids_map"] != walk["joint_ids_map"]:
        raise ValueError("The dance and the walk map their joints differently.")
    clip = root / upstream.DANCE_CLIP
    env_cfg = load_env_cfg(DANCE_TASK_ID, play=True)
    motion = env_cfg.commands["motion"]
    motion.motion_file = str(clip)
    scene = project.add_scene_mjlab(DANCE_TASK_ID, env_cfg=env_cfg)
    handle = scene.add_policy(
        name="dance1_subject2",
        policy=onnx.load(str(root / upstream.DANCE_ONNX)),
        policy_joint_names=joint_names,
        default_joint_pos=contract["default_joint_pos"],
    )
    handle.add_motion(
        name="Dance 1, subject 2",
        source=str(clip),
        fps=1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation),
        anchor_body_name=motion.anchor_body_name,
        body_names=motion.body_names,
        default=True,
    )
