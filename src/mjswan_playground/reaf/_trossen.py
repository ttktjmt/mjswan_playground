"""Playground's Trossen WXAI (`robots/trossen_wxai/sim`) on its table, and its PickCube
(`tasks/manipulation/pick_cube/trossen_wxai`) for the right arm."""

from __future__ import annotations

import re

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
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.viewer import ViewerConfig

from . import _pick_cube
from ._common import MENAGERIE, add_floor, fetch, load, name_unnamed_geoms

TABLE_HEIGHT = 0.75
JOINTS = (*(f"joint_{i}" for i in range(6)), "right_carriage_joint")
#: Playground's `constant`: each joint's position-servo stiffness, damping, force limit
#: and control range.
SERVO = {
    "joint_0": (65.092177, 14.17367, 27.0, (-3.05433, 3.05433)),
    "joint_1": (83.698740, 8.0188, 27.0, (0.0, 3.14159)),
    "joint_2": (115.808333, 0.725663, 27.0, (0.0, 2.35619)),
    "joint_3": (45.972562, 2.315707, 7.0, (-1.5708, 1.5708)),
    "joint_4": (11.364627, 2.368174, 7.0, (-1.5708, 1.5708)),
    "joint_5": (4.388846, 0.684856, 7.0, (-3.14159, 3.14159)),
    "right_carriage_joint": (339.018069, 2.373963, 35.0, (0.0, 0.044)),
}
#: The pose a pick starts from: the arm, then both gripper carriages.
PICKUP = dict(
    zip(
        (*(f"right_joint_{i}" for i in range(6)), "right_right_carriage_joint"),
        (0.0, 0.785, 1.25, -1.25, 0.0, 0.0, 0.044),
    )
) | {"right_left_carriage_joint": 0.044}
#: Playground's `TransformActions`: radians (metres for the gripper) per unit of action,
#: about each joint's pickup pose, and the gripper about the middle of its range.
ACTION_SCALE = dict(zip(PICKUP, (0.5, 0.85, 0.5, 0.5, 0.8, 0.8, 0.022)))
ACTION_OFFSET = {joint: PICKUP[joint] for joint in ACTION_SCALE} | {
    "right_right_carriage_joint": 0.022
}
ARM = SceneEntityCfg("robot", joint_names=(r"right_joint_\d",))
GRIPPER = SceneEntityCfg(
    "robot", joint_names=("right_right_carriage_joint", "right_left_carriage_joint")
)
EE = SceneEntityCfg("robot", site_names=("right_ee_gripper_link",))
#: The middle of the half sizes Playground trains the cube on, 2 to 2.5 cm.
CUBE_HALF_SIZE = 0.0225

#: Playground's `furniture.aloha_table`, around Menagerie's ALOHA table meshes.
TABLE_XML = """
<mujoco model="aloha_table">
  <asset>
    <mesh name="tabletop" file="tabletop.obj"/>
    <mesh name="tablelegs" file="tablelegs.obj"/>
    <texture name="small_meta_table_diffuse" type="2d" file="small_meta_table_diffuse.png"/>
    <material name="table" texture="small_meta_table_diffuse"/>
  </asset>
  <worldbody>
    <body name="aloha_table" pos="0 0 0.75">
      <geom name="tabletop" type="mesh" mesh="tabletop" material="table" pos="0 0 -0.75"
        quat="1 0 0 1" contype="0" conaffinity="0" group="2"/>
      <geom name="tablelegs" type="mesh" mesh="tablelegs" material="table" pos="0 0 -0.75"
        quat="1 0 0 1" contype="0" conaffinity="0" group="2"/>
      <geom name="table_collision" type="box" size="0.61 0.37 0.1" pos="0 0 -0.1" group="3"
        condim="3" friction="1 0.005 0.0001" solref="0.01 1" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _flat_names(xml: str) -> str:
    """`left/joint_0` as `left_joint_0`: mjlab keeps only what follows an entity's last
    `/`, which would give both arms the same names."""
    return re.sub(r'"(left|right)/', r'"\1_', xml)


def robot_spec() -> mujoco.MjSpec:
    """Menagerie's bimanual WXAI with Playground's servos. The left arm is passive, as
    Playground leaves it: no actuators."""
    spec = name_unnamed_geoms(
        load(MENAGERIE, "trossen_wxai/wxai_follower_biarm.xml", edit=_flat_names)
    )
    for side in ("left", "right"):
        for joint, (kp, damping, force, ctrlrange) in SERVO.items():
            name = f"{side}_{joint}"
            spec.joint(name).damping[0] = damping
            actuator = spec.actuator(name)
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.gainprm[0] = kp
            actuator.biasprm[1] = -kp
            actuator.biasprm[2] = 0.0
            actuator.forcelimited = True
            actuator.forcerange = [-force, force]
            actuator.ctrllimited = True
            actuator.ctrlrange = list(ctrlrange)
    for actuator in [a for a in spec.actuators if a.name.startswith("left_")]:
        spec.delete(actuator)
    while spec.keys:
        spec.delete(spec.keys[0])
    spec.nkey = 0
    # mjlab attaches the arm to a scene on default options; others would warn.
    spec.option = mujoco.MjSpec().option
    return spec


def license_file():
    return fetch(MENAGERIE, "trossen_wxai/LICENSE")


def table_license_file():
    return fetch(MENAGERIE, "aloha/LICENSE")


def world(spec: mujoco.MjSpec) -> None:
    """Playground's plane, the ALOHA table, and the two side lights on the workspace."""
    add_floor(spec)
    table = mujoco.MjSpec.from_string(TABLE_XML)
    # Absolute: once attached, an asset keeps only its file name.
    for asset in (*table.meshes, *table.textures):
        asset.file = str(fetch(MENAGERIE, f"aloha/assets/{asset.file}"))
    spec.attach(table, prefix="table/", frame=spec.worldbody.add_frame())
    for name, x in (("left_side_light", -0.8), ("right_side_light", 0.8)):
        light = spec.worldbody.add_light(name=name, pos=[x, -0.2, 1.2])
        light.dir = [-x, 0.2, -0.45]
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.diffuse = [0.35, 0.35, 0.35]
        light.specular = [0.1, 0.1, 0.1]
        light.castshadow = False
    spec.stat.center = [0.15, 0.0, TABLE_HEIGHT + 0.15]


def env_cfg() -> ManagerBasedRlEnvCfg:
    """`TrossenPickCube`: the right arm lifts a cube on the table to a target that moves
    every 4 to 7 s, reading its joints, the gripper's carriages, where the cube is and
    the target, at 50 Hz. The observations are Playground's, in the world frame."""
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={
                "robot": EntityCfg(
                    spec_fn=robot_spec,
                    init_state=EntityCfg.InitialStateCfg(
                        pos=(0.0, 0.0, TABLE_HEIGHT), joint_pos=dict(PICKUP)
                    ),
                    articulation=EntityArticulationInfoCfg(
                        actuators=(
                            XmlActuatorCfg(target_names_expr=tuple(ACTION_SCALE)),
                        )
                    ),
                ),
                "cube": EntityCfg(
                    spec_fn=lambda: _pick_cube.cube(CUBE_HALF_SIZE),
                    init_state=EntityCfg.InitialStateCfg(
                        pos=(0.175, 0.0, TABLE_HEIGHT + 0.025)
                    ),
                ),
            },
            extent=0.8,
            spec_fn=world,
        ),
        observations={
            "actor": ObservationGroupCfg(
                {
                    "arm_pos": ObservationTermCfg(
                        func=mdp.joint_pos_rel, params={"asset_cfg": ARM}
                    ),
                    "arm_vel": ObservationTermCfg(
                        func=mdp.joint_vel_rel, params={"asset_cfg": ARM}
                    ),
                    "gripper_pos": ObservationTermCfg(
                        func=mdp.joint_pos_rel, params={"asset_cfg": GRIPPER}
                    ),
                    "gripper_vel": ObservationTermCfg(
                        func=mdp.joint_vel_rel, params={"asset_cfg": GRIPPER}
                    ),
                    "ee_to_object": ObservationTermCfg(
                        func=_pick_cube.ee_to_object,
                        params={"object_name": "cube", "asset_cfg": EE},
                    ),
                    "object_to_goal": ObservationTermCfg(
                        func=_pick_cube.object_to_goal,
                        params={"object_name": "cube", "command_name": "target"},
                    ),
                    "last_action": ObservationTermCfg(func=mdp.last_action),
                }
            )
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=tuple(ACTION_SCALE),
                scale=ACTION_SCALE,
                offset=ACTION_OFFSET,
                use_default_offset=False,
            )
        },
        commands={
            "target": LiftingCommandCfg(
                entity_name="cube",
                resampling_time_range=(4.0, 7.0),
                difficulty="dynamic",
                target_position_range=LiftingCommandCfg.TargetPositionRangeCfg(
                    x=(0.10, 0.20), y=(-0.1, 0.1), z=(0.85, 1.0)
                ),
                object_pose_range=None,
                debug_vis=True,
                viz=LiftingCommandCfg.VizCfg(target_color=(1.0, 0.0, 0.0, 0.3)),
            )
        },
        events={
            "reset_arm": EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.1, 0.1),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": ARM,
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
            lookat=(0.2, 0.0, TABLE_HEIGHT + 0.1),
            distance=1.0,
            azimuth=100.0,
            elevation=-25.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.005, impratio=10, cone="elliptic"),
            njmax=450,
            nconmax=64,
        ),
        decimation=4,
    )


def add_scene(
    project: mjswan.ProjectHandle,
    env_id: str,
    *,
    env_cfg: ManagerBasedRlEnvCfg,
    policy: tuple[str, str] | None,
    policy_name: str,
    in_keys: list[str] | None = None,
) -> mjswan.SceneHandle:
    """A scene named after its Playground environment, driven by the policy trained on
    it with `train.py`, from the Hub at a pinned revision, once there is one."""
    scene = project.add_scene_mjlab(env_id, name=env_id, env_cfg=env_cfg)
    if policy is not None:
        repo, revision = policy
        scene.add_policy_hf(repo, revision=revision, name=policy_name, in_keys=in_keys)
    scene.add_attribution("trossen_wxai", license=license_file())
    scene.add_attribution("aloha", license=table_license_file())
    return scene
