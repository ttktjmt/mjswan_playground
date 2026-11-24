"""`SO101PickCubeVision`: the SO-101 finding the cube through its wrist camera, the task
Playground's sim-to-real demo deploys."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mjswan
import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as lift_mdp
from mjlab.utils.lab_api.math import matrix_from_quat

from . import _pick_cube, _so101

#: Trained on Playground 1.0.1's `SO101PickCubeVision`: 1800 of its 3000 iterations.
POLICY = ("ttktjmt/so101-pick-cube-vision", "c1b4d112454275b54425b69f9e6c1e244a770a96")

#: Playground's `DeltaActions` for the SO-101: radians per unit of action, and the range
#: each joint's target is held in.
DELTA_SCALE = {**{joint: 0.05 for joint in _so101.DYNAMICS}, "gripper": 0.25}
DELTA_LIMITS = {
    "shoulder_pan": (-3.14159, 3.14159),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-0.17453, 2.79253),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.79253, 2.79253),
    "gripper": (-0.174533, 1.7453292),
}


@dataclass(kw_only=True)
class DeltaJointPositionActionCfg(JointPositionActionCfg):
    """Playground's `DeltaActions`: each action, clipped to ±1 and scaled, moves the
    joint targets on from where they were, smoothed with the step before. The targets
    stay within `limits` and start each episode at the default pose."""

    smoothing: float = 0.5
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    def build(self, env: Any) -> DeltaJointPositionAction:
        return DeltaJointPositionAction(self, env)


class DeltaJointPositionAction(JointPositionAction):
    cfg: DeltaJointPositionActionCfg

    def __init__(self, cfg: DeltaJointPositionActionCfg, env: Any):
        super().__init__(cfg, env)
        bounds = torch.tensor(
            [cfg.limits[name] for name in self.target_names], device=self.device
        )
        self._low, self._high = bounds[:, 0], bounds[:, 1]
        self._target = self._offset.clone()
        self._step = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        step = actions.clamp(-1.0, 1.0) * self._scale
        smoothing = self.cfg.smoothing
        self._step = (1.0 - smoothing) * step + smoothing * self._step
        self._target = torch.clamp(self._target + self._step, self._low, self._high)
        self._processed_actions = self._target.clone()

    def reset(self, env_ids: Any = None) -> None:
        super().reset(env_ids)
        self._target[env_ids] = self._offset[env_ids]
        self._step[env_ids] = 0.0


def _site_pos(env: Any, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """A site's position in the world: Playground's `ee/xpos`."""
    return env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def _site_xmat(env: Any, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """A site's orientation, its rotation matrix row by row: Playground's `ee/xmat`."""
    quat = env.scene[asset_cfg.name].data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
    return matrix_from_quat(quat).reshape(-1, 9)


def _wrist_image(env: Any, sensor_name: str, size: int) -> torch.Tensor:
    """Playground's `obs_camera`: the camera's square frame, averaged down to `size`."""
    rgb = lift_mdp.camera_rgb(env, sensor_name)
    return torch.nn.functional.interpolate(rgb, size=(size, size), mode="area")


def _world(spec: mujoco.MjSpec) -> None:
    """Playground's vision scene: a dim headlight, shadows from a key light on the
    workspace and from the front fill, two room lights, and the arm, the cube and the
    floor drawn in flat colour (`override_flat_materials`), the floor the SO-101's grey."""
    _pick_cube.world(spec)
    center = spec.worldbody.add_body(name="workspace_center", pos=[0.25, 0.0, 0.0])
    key = spec.light("light")
    key.pos = [0.2, 0.0, 1.4]
    key.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    key.targetbody = center.name
    key.diffuse = [0.6, 0.6, 0.6]
    key.castshadow = True
    for name, x in (("left_room_light", -0.7), ("right_room_light", 0.7)):
        light = spec.worldbody.add_light(name=name, pos=[x, 0.1, 1.2])
        light.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        light.targetbody = center.name
        light.diffuse = [0.25, 0.25, 0.25]
        light.specular = [0.05, 0.05, 0.05]
        light.castshadow = False
    spec.visual.headlight.diffuse = [0.1, 0.1, 0.1]
    spec.visual.headlight.ambient = [0.15, 0.15, 0.15]
    fill = spec.light("front_fill")
    fill.pos = [0.1, -1.0, 1.2]
    fill.diffuse = [0.35, 0.35, 0.35]
    fill.castshadow = True
    spec.geom("floor").rgba = [0.25, 0.25, 0.25, 1.0]
    for geom in spec.geoms:
        if geom.name == "floor" or geom.name.startswith(("robot/", "cube/")):
            geom.material = ""


def env_cfg() -> ManagerBasedRlEnvCfg:
    """The policy sees the wrist camera's 64 x 64 frame averaged to 16 x 16, beside the
    joints, the gripper's pose and the target in the gripper's frame, and moves the arm
    by deltas at 10 Hz. The rest is `SO101PickCube`'s. three.js draws the frame, not
    Playground's ray tracer, so its shading differs from what the policy trained on."""
    cfg = _pick_cube.env_cfg()
    gripper = SceneEntityCfg("robot", site_names=("gripperframe",))
    cfg.scene.spec_fn = _world
    cfg.scene.sensors = (
        CameraSensorCfg(
            name="wrist_cam",
            camera_name="robot/wrist_cam",
            width=64,
            height=64,
            data_types=("rgb",),
            enabled_geom_groups=(0, 2),
            use_shadows=True,
        ),
    )
    cfg.observations = {
        "actor": ObservationGroupCfg(
            {
                "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
                "ee_pos": ObservationTermCfg(
                    func=_site_pos, params={"asset_cfg": gripper}
                ),
                "ee_orientation": ObservationTermCfg(
                    func=_site_xmat, params={"asset_cfg": gripper}
                ),
                "goal_position": ObservationTermCfg(
                    func=lift_mdp.target_position,
                    params={"command_name": "target", "asset_cfg": gripper},
                ),
                "last_action": ObservationTermCfg(func=mdp.last_action),
            }
        ),
        "camera": ObservationGroupCfg(
            {
                "camera": ObservationTermCfg(
                    func=_wrist_image, params={"sensor_name": "wrist_cam", "size": 16}
                )
            }
        ),
    }
    cfg.actions = {
        "joint_pos": DeltaJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=DELTA_SCALE,
            limits=DELTA_LIMITS,
        )
    }
    cfg.sim.mujoco.ccd_iterations = 300
    cfg.decimation = 20
    return cfg


def add_scenes(project: mjswan.ProjectHandle) -> None:
    _pick_cube.add_scene(
        project,
        "SO101PickCubeVision",
        env_cfg=env_cfg(),
        policy=POLICY,
        policy_name="PickCube Vision",
        in_keys=["actor", "camera"],
    )
