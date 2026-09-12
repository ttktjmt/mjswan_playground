"""MuscleMimic: a 354-muscle full body tracking a walking clip, driven by the public
checkpoint. See ``README.md``."""

from __future__ import annotations

import mjswan
import onnx
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mjswan.envs.mdp.actions import MuscleActivationActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg

from . import terms, upstream

TASK_ID = "myoMimicFullbody-v0"


def setup_builder() -> mjswan.Builder:
    from myosuite.core.trajectory_io import load_motion_clip
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        bootstrap_myosuite_mjlab_registry,
    )

    clip_path = upstream.fetch_clip()
    onnx_path, params = upstream.ensure_policy()
    # myosuite registers the clip-tracking variant of the task only when handed a clip.
    bootstrap_myosuite_mjlab_registry(clip_path=clip_path)

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.events["rsi"] = EventTermCfg(func=terms.make_reset(clip_path), mode="reset")

    model = Scene(env_cfg.scene, device="cpu").spec.compile()
    clip = load_motion_clip(clip_path, expected_nq=model.nq, expected_nv=model.nv)
    ctrl_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    contract = upstream.build_terms(model, clip, params, ctrl_dt)

    builder = mjswan.Builder()
    project = builder.add_project(name="MuscleMimic")
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)

    muscles = env_cfg.actions["muscles"]
    scene.add_policy(
        "mm-10m-2",
        onnx.load(onnx_path),
        observations=ObservationGroupCfg(
            terms={"upstream": ObservationTermCfg(func=contract.observation)}
        ),
        actions={
            "muscles": MuscleActivationActionCfg(
                entity_name=muscles.entity_name,
                actuator_names=tuple(
                    f"{muscles.entity_name}/{n}" for n in muscles.actuator_names
                ),
                # Upstream writes the output straight to `ctrl`, clipped to the actuator
                # range ([-1, 1] here); ~190 of the 354 outputs are negative every step.
                action_mode="direct",
            )
        },
        terminations={
            "time_out": env_cfg.terminations["time_out"],
            "deviation": TerminationTermCfg(func=contract.termination),
        },
    )
    return builder
