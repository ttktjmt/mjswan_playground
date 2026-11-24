"""Playground's ``tasks/manipulation/pick_cube`` as an mjlab task: the cube, the target
that moves, and the SO-101's state config the vision one starts from."""

from __future__ import annotations

from typing import Any

import mjswan
import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as lift_mdp
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.viewer import ViewerConfig

from . import _so101
from ._common import add_floor

#: Playground's `TransformActions` scale: 12.5 times the torque limit over each joint's
#: stiffness, 25 times for the gripper.
ACTION_SCALE = {
    joint: (25.0 if joint == "gripper" else 12.5) * _so101.FORCE_LIMIT / kp
    for joint, (kp, *_) in _so101.DYNAMICS.items()
}


def _cube() -> mujoco.MjSpec:
    """Playground's `primitive.cube`: 3 cm, 27 g, on a free joint."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_freejoint")
    body.add_geom(
        name="cube",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015, 0.015, 0.015],
        density=1000.0,
        condim=3,
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.0, 1.0, 0.0, 1.0],
        contype=2,
        conaffinity=1,
        solref=[0.01, 1.0],
    )
    return spec


def world(spec: mujoco.MjSpec) -> None:
    add_floor(spec)
    spec.stat.center = [0.25, 0.0, 0.15]


def env_cfg() -> ManagerBasedRlEnvCfg:
    """`SO101PickCube` as mjlab's lift task states it.

    The observations are mjlab's lift-cube terms, which compute Playground's: the arm's
    base is the world frame here. Playground reads the cube and gripper positions from
    `mj_step`'s kinematics, one physics step older than these, up to 4 mm apart on its
    reference rollout, which the policy does not notice. The target moves every 4 to 7 s
    as Playground's resetter moves it, without teleporting the cube.
    """
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={
                "robot": EntityCfg(
                    spec_fn=_so101.spec,
                    init_state=EntityCfg.InitialStateCfg(joint_pos=dict(_so101.HOME)),
                    # The arm's own actuators: Playground's servo model.
                    articulation=EntityArticulationInfoCfg(
                        actuators=(
                            XmlActuatorCfg(target_names_expr=tuple(_so101.HOME)),
                        )
                    ),
                ),
                "cube": EntityCfg(
                    spec_fn=_cube,
                    init_state=EntityCfg.InitialStateCfg(pos=(0.275, 0.0, 0.015)),
                ),
            },
            extent=0.6,
            spec_fn=world,
        ),
        observations={
            "actor": ObservationGroupCfg(
                {
                    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
                    "ee_to_object": ObservationTermCfg(
                        func=lift_mdp.ee_to_object_distance,
                        params={
                            "object_name": "cube",
                            "asset_cfg": SceneEntityCfg(
                                "robot", site_names=("gripperframe",)
                            ),
                        },
                    ),
                    "object_to_goal": ObservationTermCfg(
                        func=lift_mdp.object_to_goal_distance,
                        params={"object_name": "cube", "command_name": "target"},
                    ),
                    "last_action": ObservationTermCfg(func=mdp.last_action),
                }
            )
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=ACTION_SCALE,
                use_default_offset=True,
            )
        },
        commands={
            "target": LiftingCommandCfg(
                entity_name="cube",
                resampling_time_range=(4.0, 7.0),
                difficulty="dynamic",
                target_position_range=LiftingCommandCfg.TargetPositionRangeCfg(
                    x=(0.20, 0.30), y=(-0.1, 0.1), z=(0.08, 0.2)
                ),
                object_pose_range=None,
                debug_vis=True,
                viz=LiftingCommandCfg.VizCfg(target_color=(1.0, 0.0, 0.0, 0.3)),
            )
        },
        events={
            "reset_robot_joints": EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.1, 0.1),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                },
            ),
            "reset_cube": EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.075, 0.075),
                        "y": (-0.1, 0.1),
                        "yaw": (-np.pi, np.pi),
                    },
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("cube"),
                },
            ),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.2, 0.0, 0.1),
            distance=0.8,
            azimuth=138.0,
            elevation=-26.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.005, impratio=10, cone="elliptic")
        ),
        decimation=4,
    )


def _target_viz(cfg: LiftingCommandCfg) -> list[dict[str, Any]]:
    """The target, a sphere around where the cube's centre should go."""
    return [
        {
            "shape": "sphere",
            "radius": 0.02,
            "color": list(cfg.viz.target_color),
            "origin": {"state": "target_pos"},
        }
    ]


# mjswan does not bind mjlab's `LiftingCommandCfg` itself; the port registers it.
mjswan.register_command(
    "LiftingCommandCfg",
    mjswan.CommandBinding(
        state_fields=["target_pos"], command_field="target_pos", viz=_target_viz
    ),
)


def add_scene(
    project: mjswan.ProjectHandle,
    env_id: str,
    *,
    env_cfg: ManagerBasedRlEnvCfg,
    policy: tuple[str, str],
    policy_name: str,
    in_keys: list[str] | None = None,
) -> mjswan.SceneHandle:
    """A scene named after its Playground environment, driven by the policy trained on
    it with `mjswan/examples/train/so101_pick_cube.py`, from the Hub at a pinned
    revision."""
    scene = project.add_scene_mjlab(env_id, name=env_id, env_cfg=env_cfg)
    repo, revision = policy
    scene.add_policy_hf(repo, revision=revision, name=policy_name, in_keys=in_keys)
    scene.add_attribution("robotstudio_so101", license=_so101.license_file())
    return scene
