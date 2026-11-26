"""`TrossenPickCubeVision`: the same, finding the cube through the right wrist camera."""

from __future__ import annotations

import mjswan
import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as lift_mdp

from . import _pick_cube, _trossen

#: Trained by `train.py` on Playground 1.1.0's `TrossenPickCubeVision`, all 3000
#: iterations.
POLICY: tuple[str, str] | None = (
    "ttktjmt/trossen-pick-cube-vision",
    "8ca63bea76fc56ebc89d39da462e0754acd42a66",
)


def _world(spec: mujoco.MjSpec) -> None:
    """`TrossenPickCube`'s, with the cube, the tabletop and the floor drawn in flat colour
    (Playground's `override_flat_materials`)."""
    _trossen.world(spec)
    for name in ("floor", "table/tabletop"):
        spec.geom(name).material = ""


def env_cfg(*, colors: bool = False) -> ManagerBasedRlEnvCfg:
    """The policy sees the wrist camera's 32 x 32 frame, collision geometry and the table
    without shadows or textures, beside the joints, the gripper's pose and the target in
    the gripper's frame. The rest is `TrossenPickCube`'s. three.js draws the frame, not
    Playground's ray tracer, so its shading differs from what the policy trained on."""
    cfg = _trossen.env_cfg()
    # In colour, the tabletop and the floor keep their materials.
    cfg.scene.spec_fn = _trossen.world if colors else _world
    cfg.scene.sensors = (
        CameraSensorCfg(
            name="wrist_cam",
            camera_name="robot/right_wrist_cam",
            width=32,
            height=32,
            data_types=("rgb",),
            enabled_geom_groups=(0, 2, 3),
            use_shadows=False,
            use_textures=False,
        ),
    )
    actor = cfg.observations["actor"].terms
    cfg.observations = {
        "actor": ObservationGroupCfg(
            {
                **{
                    k: actor[k]
                    for k in ("arm_pos", "arm_vel", "gripper_pos", "gripper_vel")
                },
                "ee_pos": ObservationTermCfg(
                    func=_pick_cube.site_pos, params={"asset_cfg": _trossen.EE}
                ),
                "ee_orientation": ObservationTermCfg(
                    func=_pick_cube.site_xmat, params={"asset_cfg": _trossen.EE}
                ),
                "goal_position": ObservationTermCfg(
                    func=lift_mdp.target_position,
                    params={"command_name": "target", "asset_cfg": _trossen.EE},
                ),
                "last_action": ObservationTermCfg(func=mdp.last_action),
            }
        ),
        "camera": ObservationGroupCfg(
            {
                "camera": ObservationTermCfg(
                    func=lift_mdp.camera_rgb, params={"sensor_name": "wrist_cam"}
                )
            }
        ),
    }
    return cfg


def add_scenes(project: mjswan.ProjectHandle, *, colors: bool = False) -> None:
    """In gray, as published: in colour the policy, trained on flat colours, fails."""
    _trossen.add_scene(
        project,
        "TrossenPickCubeVision",
        env_cfg=env_cfg(colors=colors),
        policy=POLICY,
        policy_name="PickCube Vision",
        in_keys=["actor", "camera"],
    )
