"""TERRA: a 354-muscle body walking myosuite's terrain course and a flat route with the
TERRA-4B checkpoint. See ``README.md``."""

from __future__ import annotations

import re
from typing import Any

import mjswan
import mujoco
import onnx
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import Scene
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.registry import load_env_cfg
from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
    MyoMuscleActivationActionCfg,
)
from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import mujoco_cfg_from_model
from myosuite.integrations.musclemimic.fullbody_model import TERRAIN_GROUP

from mjswan_playground._deps import license_file

from . import terms, upstream
from .terms import ENTITY

#: myosuite's mjlab twin of its CPU waypoint task, registered through mjlab's entry point.
TASK_ID = "myoFullBodyWaypoint-v0"

#: The checkpoint states no license, so the scene carries where it comes from instead.
WEIGHTS_NOTICE = """\
TERRA-4B: https://huggingface.co/merc-s/TERRA-4B at b89a604, checkpoint_24416.
Merkourios Simos, Chengkun Li, Bianca Ziliotto, Alexander Mathis. TERRA: Terrain-Aware
Reconstruction, Retargeting and Control for Musculoskeletal Locomotion.
arXiv:2609.38653, 2026. The model card states no license.
"""


def _entity_spec(route: upstream.Route) -> mujoco.MjSpec:
    spec = route.actor_spec()
    # An unnamed keyframe clashes on attach; mjlab writes its own from `init_state`.
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def play_env_cfg(route: upstream.Route, obs_cfg: Any) -> Any:
    """The twin's play config with TERRA's actor on ``route``, and the terms TERRA reads
    in place of the twin's own policy's."""
    env_cfg = load_env_cfg(TASK_ID, play=True)
    actor = route.actor_spec().compile()
    clip = upstream.reference(route, actor)
    start = route.initial_qpos(actor)

    tendons = [actor.tendon(actor.actuator_trnid[i, 0]).name for i in range(actor.nu)]
    joints = {
        f"^{re.escape(actor.joint(j).name)}$": float(start[actor.jnt_qposadr[j]])
        for j in range(1, actor.njnt)
    }
    env_cfg.scene.entities[ENTITY] = EntityCfg(
        spec_fn=lambda: _entity_spec(route),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=tuple(f"^{re.escape(t)}$" for t in tendons),
                    transmission_type=TransmissionType.TENDON,
                ),
            )
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(start[:3]),
            rot=tuple(start[3:7]),
            joint_pos=joints,
            joint_vel={".*": 0.0},
        ),
    )
    rows, cols = obs_cfg.grid
    if obs_cfg.forward_offset != 0.0:
        raise ValueError("A grid pattern is centred on its frame")
    # From the root body, which sits where TERRA's pelvis does but, unlike the pelvis
    # body, is not turned 90 degrees from the root's yaw.
    env_cfg.scene.sensors = (
        RayCastSensorCfg(
            name=terms.SCAN,
            frame=ObjRef(
                type="body", name=actor.body(actor.jnt_bodyid[0]).name, entity=ENTITY
            ),
            pattern=GridPatternCfg(
                size=((rows - 1) * obs_cfg.resolution, (cols - 1) * obs_cfg.resolution),
                resolution=obs_cfg.resolution,
            ),
            ray_alignment="yaw",
            include_geom_groups=(TERRAIN_GROUP,),
        ),
    )
    env_cfg.sim.mujoco = mujoco_cfg_from_model(actor)
    env_cfg.episode_length_s = (clip.num_frames - 1.5) * upstream.CTRL_DT

    # Raw controls clipped to `ctrlrange`, as WaypointEnv.step, in a term mjswan runs.
    env_cfg.actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=ENTITY,
            actuator_names=tuple(actor.actuator(i).name for i in range(actor.nu)),
            tendon_names=tuple(tendons),
            action_mode="direct",
        )
    }
    # The twin's route lives in a command its own policy reads; TERRA reads none of it.
    env_cfg.commands = {}
    env_cfg.rewards = {}
    env_cfg.metrics = {}

    model = Scene(env_cfg.scene, device="cpu").spec.compile()
    env_cfg.observations = {
        "actor": ObservationGroupCfg(
            {"terra": ObservationTermCfg(func=terms.observation(model, clip, obs_cfg))}
        )
    }
    env_cfg.terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "fall": TerminationTermCfg(func=terms.fall(model)),
    }
    env_cfg.events = {
        "reset_to_start": EventTermCfg(func=terms.reset_to_start(start), mode="reset")
    }
    return env_cfg


def setup_builder() -> mjswan.Builder:
    onnx_path, obs_cfg = upstream.ensure_policy()
    policy = onnx.load(str(onnx_path))
    builder = mjswan.Builder()
    project = builder.add_project(
        name="TERRA", license=license_file("myosuite", "LICENSE")
    )
    for route in upstream.routes():
        scene = project.add_scene_mjlab(
            route.name, env_cfg=play_env_cfg(route, obs_cfg)
        )
        scene.add_policy("TERRA-4B", policy)
        scene.add_attribution(
            "musclemimic_models",
            license=license_file("musclemimic_models", "LICENSE"),
            notice=license_file("musclemimic_models", "NOTICE"),
        )
        scene.add_attribution("terra-4b", notice=WEIGHTS_NOTICE)
    return builder
