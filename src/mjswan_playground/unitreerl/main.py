"""Unitree G1 walking to velocity commands. See ``README.md``."""

from __future__ import annotations

import mjswan
import onnx
from mjlab.tasks.registry import load_env_cfg
from mjswan.mjlab.onnx_meta import read_mjlab_metadata

from mjswan_playground._gait_clock import clock_phase

from . import upstream

TASK_ID = "Unitree-G1-Flat"


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    policy = onnx.load(str(root / upstream.POLICY_ONNX))
    contract = read_mjlab_metadata(policy)
    env_cfg = load_env_cfg(TASK_ID, play=True)
    clock_phase(env_cfg)

    builder = mjswan.Builder()
    project = builder.add_project(name="G1 Velocity", license=root / "LICENCE")
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
    scene.add_policy(
        name="v0",
        policy=policy,
        policy_joint_names=[f"robot/{name}" for name in contract.joint_names],
        default_joint_pos=contract.default_joint_pos,
    )
    return builder
