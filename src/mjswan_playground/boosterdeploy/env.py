"""mjlab's tracking env without state estimation, around a Booster robot as
booster_deploy's MuJoCo controller runs it: its own MJCF, timestep and PD gains."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sim import MujocoCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

ANCHOR = "trunk"
#: The root first: mjlab resets the robot to the first tracked body's pose.
BODY_NAMES = (
    "trunk",
    "left_hip_roll_link",
    "left_knee_pitch_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_pitch_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
)
END_EFFECTORS = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
)


def _robot_spec(xml: Path) -> mujoco.MjSpec:
    """The robot and its sky: no floor, lights or motors, and its gyro named as mjlab's
    tracking observation reads it."""
    spec = mujoco.MjSpec.from_file(str(xml))
    floor = [t for t in spec.textures if t.type != mujoco.mjtTexture.mjTEXTURE_SKYBOX]
    for element in [
        *spec.worldbody.geoms,
        *spec.worldbody.lights,
        *spec.actuators,
        *spec.materials,
        *floor,
    ]:
        spec.delete(element)
    for sensor in list(spec.sensors):
        if sensor.type == mujoco.mjtSensor.mjSENS_GYRO:
            sensor.name = "imu_ang_vel"
            sensor.noise = 0.0
        else:
            spec.delete(sensor)
    return spec


def _robot_cfg(xml: Path, task: Any) -> EntityCfg:
    robot = task.robot
    gains = zip(
        robot.joint_names,
        robot.joint_stiffness,
        robot.joint_damping,
        robot.effort_limit,
    )
    return EntityCfg(
        spec_fn=lambda: _robot_spec(xml),
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(task.mujoco.init_pos),
            joint_pos=dict(zip(robot.joint_names, robot.default_joint_pos)),
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            # booster_deploy's MuJoCo controller: the PD torque each physics step,
            # clipped to the task's effort limit.
            actuators=tuple(
                IdealPdActuatorCfg(
                    target_names_expr=(name,),
                    stiffness=kp,
                    damping=kd,
                    effort_limit=limit,
                )
                for name, kp, kd, limit in gains
            ),
            soft_joint_pos_limit_factor=1.0,
        ),
    )


def tracking_env_cfg(task: Any, assets: Path) -> ManagerBasedRlEnvCfg:
    robot = task.robot
    xml = Path(robot.mjcf_path.replace("{BOOSTER_ASSETS_DIR}", str(assets)))
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": _robot_cfg(xml, task)}

    actor = cfg.observations["actor"].terms
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={
                k: v
                for k, v in actor.items()
                if k not in ("motion_anchor_pos_b", "base_lin_vel")
            },
            concatenate_terms=True,
            enable_corruption=False,
        )
    }
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    # `BeyondMimicPolicy`'s default, which both K1 tasks keep.
    assert task.policy.action_scale_factors is None
    assert task.policy.fixed_action_scale is None
    action.scale = {
        name: 0.25 * limit / kp
        for name, limit, kp in zip(
            robot.joint_names, robot.effort_limit, robot.joint_stiffness
        )
    }

    motion = cfg.commands["motion"]
    assert isinstance(motion, MotionCommandCfg)
    motion.anchor_body_name = ANCHOR
    motion.body_names = BODY_NAMES
    motion.sampling_mode = "start"
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.0, 0.0)

    cfg.terminations["ee_body_pos"].params["body_names"] = END_EFFECTORS
    # booster_deploy runs the nominal robot: no pushes or randomization.
    cfg.events = {}
    cfg.rewards = {}
    cfg.viewer.body_name = ANCHOR

    cfg.decimation = task.mujoco.decimation
    cfg.sim.mujoco = MujocoCfg(
        timestep=task.mujoco.physics_dt,
        # The MJCF's own solver options, which booster_deploy keeps.
        integrator="euler",
        impratio=10.0,
        tolerance=1e-6,
    )
    cfg.episode_length_s = int(1e9)
    return cfg
