"""XGO Mini2S, Mini2SW and Lite3 walking to velocity commands. See ``README.md``."""

from __future__ import annotations

import dataclasses
from typing import Any

import mjswan
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjswan_playground._gait_clock import clock_phase

from . import upstream

#: Each scene's task id and the folder of its policy in the Hub repo, in upstream's order.
WALKS = {
    "Mini2S-Walk-Flat": "mini2s_walk",
    "Mini2SW-Walk-Flat": "mini2sw_walk",
    "Lite3-Walk-Flat": "lite3_walk",
}


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    builder = mjswan.Builder()
    project = builder.add_project(name="luwu_mjlab", license=root / "LICENCE")
    for task_id, folder in WALKS.items():
        scene = project.add_scene_mjlab(task_id, env_cfg=_env_cfg(task_id))
        scene.add_policy_hf(
            upstream.HF_REPO_ID,
            filename=f"{folder}/policy.onnx",
            revision=upstream.HF_REVISION,
            name=folder,
        )
    return builder


def _env_cfg(task_id: str) -> Any:
    env_cfg = load_env_cfg(task_id, play=True)
    clock_phase(env_cfg)
    env_cfg.commands["twist"] = _mjlab_twist(env_cfg.commands["twist"])
    # Play re-picks a terrain tile on reset; the flat walks have a plane and no tiles.
    env_cfg.events.pop("randomize_terrain")
    return env_cfg


def _mjlab_twist(twist: Any) -> UniformVelocityCommandCfg:
    """Upstream's copy of mjlab 1.2's twist command, as the 1.6 class mjswan binds."""
    fields = {field.name for field in dataclasses.fields(UniformVelocityCommandCfg)}
    values = {
        field.name: getattr(twist, field.name)
        for field in dataclasses.fields(twist)
        if field.name in fields and field.name not in ("ranges", "viz")
    }
    return UniformVelocityCommandCfg(
        **values,
        ranges=UniformVelocityCommandCfg.Ranges(**dataclasses.asdict(twist.ranges)),
    )
