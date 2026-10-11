"""Stilts: microduck walking on stilts from 10 cm to 2 m, one scene and policy per
height."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import mjswan
import mujoco
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjswan.managers.event_manager import EventTermCfg
from mjswan.managers.observation_manager import ObservationGroupCfg
from mjswan.mjlab import apply_mjlab_sim_options, build_single_entity_trace_env

from mjswan_playground._trace import CommandValues
from mjswan_playground.microduck.main import (
    CONTROL_DT,
    ENTITY,
    STAND_HEIGHT,
    TRACKED_BODY,
    VELOCITY_ENV,
    _driven,
    _padded,
    _stand_pose,
)

from . import _common

POLICY_REPO_ID = "HannesVonEssen/microduck-stilts"
POLICY_REVISION = "371b7e0a8843d215afa768439b9e6ceab6248979"

STILTS_PY = "src/mjlab_microduck/robot/stilt_constants.py"
#: Every released height, each ``<h>cm/policy.onnx`` on the Hub.
HEIGHTS_CM = (10, 15, 20, 25, 50, 100, 140, 200)
#: ``MICRODUCK_STILT_BLEND`` of every release: halfway from platform to peg.
BLEND = 0.5

#: ``make_microduck_stilt_env_cfg``'s trained ranges, Forward at its play speed.
TWIST = mjswan.ui_command(
    [
        mjswan.SliderConfig(
            name="lin_vel_x",
            label="Forward (m/s)",
            range=(-0.12, 0.25),
            default=0.15,
            step=0.01,
        ),
        mjswan.SliderConfig(
            name="lin_vel_y",
            label="Sideways (m/s)",
            range=(-0.06, 0.06),
            default=0.0,
            step=0.01,
        ),
        mjswan.SliderConfig(
            name="ang_vel_z",
            label="Turn (rad/s)",
            range=(-0.35, 0.35),
            default=0.0,
            step=0.01,
        ),
    ]
)

#: Shorter than ``_common.SERVO_FILTER_S``: every height walks at 2.5 to 30 ms, and the
#: 1.4 m policy falls from 35 ms.
SERVO_FILTER_S = 0.025
#: BAM's ``BamActuator`` (bam/mjlab.py) stiffens each servo's friction constraint, as
#: MuJoCo Warp has no noslip solver.
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)
#: The XL330 friction model upstream trains with, ``params/xl330/m6.json`` in
#: Rhoban/bam at 62bd8ce (the commit upstream's ``uv.lock`` pins).
FRICTION_BASE = 0.004771183165566
FRICTION_STRIBECK = 0.004676345799486616
LOAD_FRICTION_MOTOR = 0.2667860954283698
DTHETA_STRIBECK = 2.890372094130307
ALPHA = 8.683259907618984


def servo_load_friction(
    env: Any, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg
) -> None:
    """BAM's gearbox friction on each servo's own torque, as its ``frictionloss``.

    Training's ``BamActuator`` rewrites it every physics step, this every control step.
    Its terms on the external load are left out: they read ``qfrc_bias`` and
    ``qfrc_constraint``, which neither mjlab's entity data nor the browser serves."""
    asset = env.scene[asset_cfg.name]
    joints = asset_cfg.joint_ids
    speed = asset.data.joint_vel[:, joints].abs()
    friction = (
        FRICTION_BASE
        + FRICTION_STRIBECK * torch.exp(-((speed / DTHETA_STRIBECK) ** ALPHA))
        + LOAD_FRICTION_MOTOR * asset.data.qfrc_actuator[:, joints].abs()
    )
    dofs = asset.indexing.joint_v_adr[joints]
    env.sim.model.dof_frictionloss[env_ids[:, None], dofs] = friction[env_ids]


def _scene_spec(
    root: Path, stand_pose: dict[str, float], height_cm: int, *, tracing: bool = False
) -> mujoco.MjSpec:
    """``get_stilt_walk_spec`` on ``scene_walk.xml`` under ``FULL_COLLISION``, with
    mjlab's velocity sim settings, the servo filters and BAM's stiff servo friction;
    ``tracing`` leaves out the last three and the mesh copies, as nothing traced reads
    them."""
    stilts = _common.load_upstream(root, STILTS_PY)
    spec = stilts.get_stilt_walk_spec(height_cm=height_cm, blend=BLEND)
    _common.full_collision(spec)
    if not tracing:
        apply_mjlab_sim_options(spec, VELOCITY_ENV.sim)
        _common.filter_servos(spec, SERVO_FILTER_S)
        for name in _common.servo_joints(spec):
            spec.joint(name).solref_friction = STIFF_SOLREF_FRICTION
            spec.joint(name).solimp_friction = STIFF_SOLIMP_FRICTION
        _common.own_collision_meshes(spec)
    # Upstream raises its reset by the stilt length.
    root_pose = (0.0, 0.0, STAND_HEIGHT + height_cm / 100, 1.0, 0.0, 0.0, 0.0)
    _common.set_only_keyframe(spec, stand_pose, root_pose)
    return spec


def add_scenes(
    project: mjswan.ProjectHandle, root: Path, heights_cm: tuple[int, ...] = HEIGHTS_CM
) -> None:
    stand_pose = _stand_pose(root / _common.WALK_SCENE_XML)
    # One trace env for every height: the traced terms read only the joints and the
    # root, which the stilts leave alone, and each env holds a compiled mujoco_warp model.
    trace_env = build_single_entity_trace_env(
        partial(_scene_spec, root, stand_pose, HEIGHTS_CM[0], tracing=True),
        entity_name=ENTITY,
        commands={"twist": CommandValues(3)},
    )
    for height_cm in heights_cm:
        spec = _scene_spec(root, stand_pose, height_cm)
        joint_names = _common.servo_joints(spec)
        joints = _common.joints_cfg(joint_names)
        height = f"{height_cm} cm" if height_cm < 100 else f"{height_cm / 100:.1f} m"

        scene = project.add_scene(
            name=f"Stilts {height}", spec=spec, control_dt=CONTROL_DT
        )
        scene.add_attribution("3d-models", license=root / "LICENSE-HARDWARE")
        # Aimed halfway down the stilts, from just far enough to fit the duck and its
        # stilts in the 45 degree view.
        scene.set_viewer(
            mjswan.ViewerConfig(
                origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
                body_name=TRACKED_BODY,
                lookat=(0.0, 0.0, -height_cm / 200),
                distance=max(0.8, 1.7 * (height_cm / 100 + 0.25)),
                elevation=-10.0,
                azimuth=40.0,
            )
        )
        scene.set_trace_env(trace_env)
        scene.add_policy(
            name="Walk",
            policy=_common.hub_policy(
                POLICY_REPO_ID, f"{height_cm}cm/policy.onnx", POLICY_REVISION
            ),
            commands={"twist": TWIST},
            # The Hub's contract: 48 of proprioception, twist(3), head(4), body(6).
            observations=ObservationGroupCfg(
                terms={
                    **_common.proprioception(joints),
                    "twist": _driven("twist"),
                    "head_command": _padded(4),
                    "body_command": _padded(6),
                }
            ),
            actions=_common.servo_action(),
            terminations={"fell_over": VELOCITY_ENV.terminations["fell_over"]},
            # Without it the policies from 1.0 m up fall within 10 s.
            events={
                "servo_friction": EventTermCfg(
                    func=servo_load_friction,
                    mode="interval",
                    interval_range_s=(CONTROL_DT, CONTROL_DT),
                    params={"asset_cfg": joints},
                    label="Servo gearbox friction",
                )
            },
            policy_joint_names=joint_names,
            default_joint_pos=[stand_pose[name] for name in joint_names],
        )
