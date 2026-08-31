"""Microduck: the nine policies the robot ships, each in the scene it runs in.

The robot XMLs already carry the position actuators the real servos run, so the scenes
compile straight from them rather than from upstream's training envs. See ``README.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mjswan
import mujoco
import numpy as np
import onnx
from mjlab.envs.mdp import observations as obs_fns
from mjlab.envs.mdp import terminations as term_fns
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.trace_env import build_single_entity_trace_env

from mjswan_playground._deps import ensure_repo

from . import terms

#: The training repo: robot XMLs, meshes and the env configs the numbers below come from.
RL_REPO_URL = "https://github.com/pollen-robotics/microduck_rl.git"
RL_REPO_COMMIT = "c690e5599866893b11b3ab2764883a8b0800cbe7"
#: The robot's own repo: the ONNX policies its daemon runs.
DEPLOY_REPO_URL = "https://github.com/pollen-robotics/microduck.git"
DEPLOY_REPO_COMMIT = "590b986bd8c0d50ae02cb3ea2f59c463b6828168"

_ROBOT_DIR = "src/mjlab_microduck/robot/microduck"
SCENE_XML = f"{_ROBOT_DIR}/scene.xml"
SCENE_ROLLERS_XML = f"{_ROBOT_DIR}/scene_rollers.xml"
SCENE_BALL_XML = f"{_ROBOT_DIR}/scene_ball.xml"
POLICY_DIR = "policies"

ENTITY = "robot"
ROOT_JOINT = "trunk_base_freejoint"
TRACKED_BODY = "trunk_base"
BALL_JOINT = "ball_free"

#: ``sim.mujoco.timestep`` (0.005) * ``decimation`` (4) — 50 Hz. The XMLs carry no
#: ``<option>``, so both halves travel in the spec.
CONTROL_DT = 0.02
TIMESTEP = 0.005
#: The rest of mjlab's ``MujocoCfg``; MuJoCo's XML defaults are Euler and 100/50.
INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
SOLVER_ITERATIONS = 10
SOLVER_LS_ITERATIONS = 20

#: The keyframe every scene resets to: upstream's STAND2 / ``HOME_FRAME``.
STAND_KEY = "STAND"
#: ``infer_policy.py``'s spawn heights: 5 mm above the keyframe's, plus wheel on rollers.
STAND_HEIGHT = 0.125
ROLLERS_STAND_HEIGHT = 0.1385
#: Wheel-bearing friction, which upstream sets at inference time rather than in the XML
#: ("non-zero frictionloss in the XML breaks training").
WHEEL_FRICTIONLOSS = 0.003
#: Where the kick ball sits, from ``reset_ball_in_front_of_foot``'s docstring: "the toe
#: tip at x~=0.034, so (0.08, -0.042) puts a 35mm-radius ball ~1cm in front of the toe".
#: Mirrored per foot. Not that function's 0.09 default, which is only the centre of a
#: +/-15 mm draw: a scene bakes one placement, and at 0.09 the right foot swings past.
BALL_OFFSET_X = 0.08
BALL_OFFSET_ABS_Y = 0.042
BALL_RADIUS = 0.035

#: ``make_velocity_env_cfg``'s fall termination, which upstream drops on the policies
#: that start or end on the ground.
FELL_OVER_ANGLE_DEG = 70.0

#: Final command ranges the curricula reach — what a finished policy has seen.
TWIST_RANGES = ((-0.4, 0.4), (-0.3, 0.3), (-1.0, 1.0))
ROLLER_THROTTLE_RANGE = (-0.5, 0.6)
#: ``infer_policy.py``'s deploy range for the heading slot; see README.
ROLLER_HEADING_RANGE = (-1.0, 1.0)
HEAD_RANGES = ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))
HEAD_LABELS = ("Neck Pitch", "Head Pitch", "Head Yaw", "Head Roll")
#: Only z / roll / pitch are steerable; x, y and yaw are alive-range noise upstream never
#: trains as a command, so they are padded with the zero they were centred on.
BODY_Z_RANGE = (-0.04, 0.030)
BODY_ANGLE_RANGE = (-math.radians(15), math.radians(15))


@dataclass(frozen=True)
class _Demo:
    """One shipped checkpoint, the scene it runs in, and what drove each command slot.

    One policy per scene, not one per model file: mjswan 0.9.3 writes a scene's fused
    observation graph to ``obs/<group>.onnx``, one path for every policy on the scene, so
    policies reading different command slots would overwrite each other's. The nine need
    five layouts, hence nine scenes off four specs: upstream's three XMLs, kick mirrored.
    """

    name: str
    onnx: str
    xml: str
    twist: str
    """Which of five shapes upstream put in this policy's twist slot: ``"velocity"``,
    ``"heading"``, ``"posture"``, ``"phase"`` or ``"zero"``."""
    head: bool = False
    body: bool = False
    fell_over: bool = True
    root_height: float = STAND_HEIGHT
    wheels: bool = False
    ball_y: float | None = None
    trace_xml: str | None = None
    """The spec tracing runs against, when it cannot be the scene's own. Only the kick
    scenes need it: an mjlab ``Entity`` is one freejoint and theirs holds a second for the
    ball, which no policy reads anyway."""


DEMOS = (
    _Demo("Walk", "alpha_walking.onnx", SCENE_XML, "velocity", head=True),
    _Demo(
        "Stand & Pose",
        "alpha_stand.onnx",
        SCENE_XML,
        "zero",
        head=True,
        body=True,
        fell_over=False,
    ),
    _Demo(
        "Sit / Stand",
        "alpha_sitstand.onnx",
        SCENE_XML,
        "posture",
        head=True,
        fell_over=False,
    ),
    _Demo("Ground Pick", "alpha_ground_pick.onnx", SCENE_XML, "phase"),
    _Demo("Roulade", "roulade.onnx", SCENE_XML, "zero", fell_over=False),
    _Demo(
        "Ball Kick (right)",
        "ball_kick_right.onnx",
        SCENE_BALL_XML,
        "zero",
        ball_y=-BALL_OFFSET_ABS_Y,
        trace_xml=SCENE_XML,
    ),
    _Demo(
        "Ball Kick (left)",
        "ball_kick_left.onnx",
        SCENE_BALL_XML,
        "zero",
        ball_y=BALL_OFFSET_ABS_Y,
        trace_xml=SCENE_XML,
    ),
    _Demo(
        "Roller Skate",
        "roller.onnx",
        SCENE_ROLLERS_XML,
        "heading",
        root_height=ROLLERS_STAND_HEIGHT,
        wheels=True,
    ),
    _Demo(
        "Roller Crouch",
        "roller_crouch.onnx",
        SCENE_ROLLERS_XML,
        "phase",
        root_height=ROLLERS_STAND_HEIGHT,
        wheels=True,
    ),
)


def _resolve_rl_root() -> Path:
    return ensure_repo(
        name="microduck_rl",
        url=RL_REPO_URL,
        commit=RL_REPO_COMMIT,
        marker=SCENE_XML,
        root_env_var="MJSWAN_MICRODUCK_RL_ROOT",
    )


def _resolve_deploy_root() -> Path:
    return ensure_repo(
        name="microduck",
        url=DEPLOY_REPO_URL,
        commit=DEPLOY_REPO_COMMIT,
        marker=f"{POLICY_DIR}/alpha_walking.onnx",
        root_env_var="MJSWAN_MICRODUCK_ROOT",
    )


def _servo_joints(model: mujoco.MjModel) -> list[str]:
    """The 14 servos in actuator order, the order every policy reads and writes — the
    same way upstream's runtime indexes (``jnt_qposadr[actuator_trnid]``), which drops the
    rollers model's four passive wheel hinges on its own."""
    names: list[str] = []
    for actuator in range(model.nu):
        if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        names.append(model.joint(int(model.actuator_trnid[actuator, 0])).name)
    return names


def _stand_pose(scene_xml: Path) -> dict[str, float]:
    """The STAND keyframe's joint positions by name: what actions offset from and every
    ``*_rel`` observation subtracts (upstream's ``DEFAULT_POSE``).

    Read from ``scene.xml`` for every scene: the ball scene carries no keyframe and the
    rollers scene's is the same pose at a different height.
    """
    model = mujoco.MjSpec.from_file(str(scene_xml)).compile()
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, STAND_KEY)
    if key < 0:
        raise ValueError(f"{scene_xml} has no {STAND_KEY!r} keyframe to pose from.")
    qpos = model.key_qpos[key]
    return {
        model.joint(joint).name: float(qpos[model.jnt_qposadr[joint]])
        for joint in range(model.njnt)
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
    }


def _scene_spec(
    scene_xml: Path,
    stand_pose: dict[str, float],
    *,
    root_height: float,
    wheels: bool,
    ball_y: float | None,
    sim_options: bool = True,
) -> mujoco.MjSpec:
    """The scene as the browser compiles it: upstream's solver settings (the XMLs carry
    no ``<option>``), the wheel friction its inference script writes by hand, and one
    keyframe.

    STAND has to be the *only* keyframe: the browser resets to the first one, and so does
    the tracing env, whose ``default_joint_pos`` ``joint_pos_rel`` bakes in. Upstream's
    INIT first would pose every policy against a zero it never trained on.
    """
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    if sim_options:
        # Left off the tracing spec, whose own scene's options win the conflict anyway;
        # nothing traced depends on them.
        spec.option.timestep = TIMESTEP
        spec.option.integrator = INTEGRATOR
        spec.option.iterations = SOLVER_ITERATIONS
        spec.option.ls_iterations = SOLVER_LS_ITERATIONS
    if wheels:
        for joint in spec.joints:
            if joint.name.startswith("passive_"):
                joint.frictionloss = WHEEL_FRICTIONLOSS

    model = spec.compile()
    qpos = np.array(model.qpos0, dtype=float)
    qpos[_free_joint_adr(model, ROOT_JOINT)] = [
        0.0,
        0.0,
        root_height,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    for joint in range(model.njnt):
        name = model.joint(joint).name
        if name in stand_pose:
            qpos[model.jnt_qposadr[joint]] = stand_pose[name]
    if ball_y is not None:
        qpos[_free_joint_adr(model, BALL_JOINT)] = [
            BALL_OFFSET_X,
            ball_y,
            BALL_RADIUS,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    ctrl = [stand_pose[name] for name in _servo_joints(model)]

    for key in list(spec.keys):
        spec.delete(key)
    spec.add_key(name=STAND_KEY, qpos=qpos.tolist(), ctrl=ctrl)
    return spec


def _free_joint_adr(model: mujoco.MjModel, name: str) -> slice:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise ValueError(f"No joint named {name!r} in the compiled scene.")
    start = int(model.jnt_qposadr[joint])
    return slice(start, start + 7)


def _sliders(specs: list[tuple[str, str, tuple[float, float]]]) -> mjswan.CommandInput:
    return mjswan.ui_command(
        [
            mjswan.SliderConfig(
                name=name,
                label=label,
                range=value_range,
                default=0.0,
                step=round((value_range[1] - value_range[0]) / 40.0, 4),
            )
            for name, label, value_range in specs
        ]
    )


def _twist_commands(kind: str) -> dict[str, mjswan.CommandInput]:
    """The browser-side command terms this policy's twist slot needs, if any. A slot
    upstream pinned to zero gets no command — it is padded in the observation instead."""
    if kind == "velocity":
        return {
            "twist": _sliders(
                [
                    ("lin_vel_x", "Forward (m/s)", TWIST_RANGES[0]),
                    ("lin_vel_y", "Sideways (m/s)", TWIST_RANGES[1]),
                    ("ang_vel_z", "Turn (rad/s)", TWIST_RANGES[2]),
                ]
            )
        }
    if kind == "heading":
        return {
            # Two commands, not one: the middle slot is padding — see `_twist_terms`.
            "throttle": _sliders(
                [("lin_vel_x", "Forward (m/s)", ROLLER_THROTTLE_RANGE)]
            ),
            "heading": _sliders(
                [("heading_error", "Error (rad)", ROLLER_HEADING_RANGE)]
            ),
        }
    if kind == "posture":
        return {
            "posture": mjswan.ui_command(
                [mjswan.CheckboxConfig(name="sit", label="Sit", default=False)]
            )
        }
    if kind == "phase":
        return {"twist": terms.GroundPickPhaseCommandCfg(control_dt=CONTROL_DT)}
    if kind != "zero":
        raise ValueError(f"Unknown twist kind {kind!r}")
    return {}


def _twist_terms(kind: str) -> dict[str, ObservationTermCfg]:
    """The three twist values, in order, from whatever drives them."""
    generated = obs_fns.generated_commands
    if kind in ("velocity", "phase"):
        return {
            "twist": ObservationTermCfg(
                func=generated, params={"command_name": "twist"}
            )
        }
    if kind == "heading":
        # `lin_vel_y` is the middle slot, pinned to (0, 0) by upstream's roller env.
        return {
            "throttle": ObservationTermCfg(
                func=generated, params={"command_name": "throttle"}
            ),
            "lin_vel_y": ObservationTermCfg(func=terms.zeros, params={"dim": 1}),
            "heading": ObservationTermCfg(
                func=generated, params={"command_name": "heading"}
            ),
        }
    if kind == "posture":
        return {
            "posture": ObservationTermCfg(
                func=generated, params={"command_name": "posture"}
            ),
            "twist_pad": ObservationTermCfg(func=terms.zeros, params={"dim": 2}),
        }
    return {"twist": ObservationTermCfg(func=terms.zeros, params={"dim": 3})}


def _pose_commands(policy: _Demo) -> dict[str, mjswan.CommandInput]:
    commands: dict[str, mjswan.CommandInput] = {}
    if policy.head:
        commands["head_pose"] = _sliders(
            [
                (name, label, value_range)
                for name, label, value_range in zip(
                    ("neck_pitch", "head_pitch", "head_yaw", "head_roll"),
                    HEAD_LABELS,
                    HEAD_RANGES,
                )
            ]
        )
    if policy.body:
        commands["body_pose"] = _sliders(
            [
                ("z", "Trunk Height (m)", BODY_Z_RANGE),
                ("roll", "Trunk Roll (rad)", BODY_ANGLE_RANGE),
                ("pitch", "Trunk Pitch (rad)", BODY_ANGLE_RANGE),
            ]
        )
    return commands


def _pose_terms(policy: _Demo) -> dict[str, ObservationTermCfg]:
    generated = obs_fns.generated_commands
    pose: dict[str, ObservationTermCfg] = {}
    if policy.head:
        pose["head_command"] = ObservationTermCfg(
            func=generated, params={"command_name": "head_pose"}
        )
    else:
        pose["head_command"] = ObservationTermCfg(func=terms.zeros, params={"dim": 4})
    if policy.body:
        # [x, y, z, roll, pitch, yaw]: x, y and yaw are not commands upstream trains, so
        # they stay at the zero their alive-range is centred on.
        pose["body_xy"] = ObservationTermCfg(func=terms.zeros, params={"dim": 2})
        pose["body_command"] = ObservationTermCfg(
            func=generated, params={"command_name": "body_pose"}
        )
        pose["body_yaw"] = ObservationTermCfg(func=terms.zeros, params={"dim": 1})
    else:
        pose["body_command"] = ObservationTermCfg(func=terms.zeros, params={"dim": 6})
    return pose


def _observations(policy: _Demo, joints: SceneEntityCfg) -> ObservationGroupCfg:
    """The 61 values ``robotd`` builds, in its order: 3 gyro + 3 gravity + 14 joint
    positions + 14 joint velocities + 14 last actions + a 13-wide command block."""
    group: dict[str, ObservationTermCfg] = {
        # Upstream's gyro sits on an identity-oriented site on the root body, so it
        # reads exactly this.
        "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=obs_fns.projected_gravity),
        "joint_pos": ObservationTermCfg(
            func=obs_fns.joint_pos_rel, params={"asset_cfg": joints}
        ),
        "joint_vel": ObservationTermCfg(
            func=obs_fns.joint_vel_rel, params={"asset_cfg": joints}
        ),
        "actions": ObservationTermCfg(func=obs_fns.last_action),
    }
    group.update(_twist_terms(policy.twist))
    group.update(_pose_terms(policy))
    return ObservationGroupCfg(terms=group)


def _trace_commands(policy: _Demo) -> dict[str, object]:
    """Trace-time widths for the commands the browser owns."""
    widths = {
        "velocity": {"twist": 3},
        # The clock is traced for real, but the term reading it still resolves its name
        # against the trace env — as husky's does.
        "phase": {"twist": 3},
        "heading": {"throttle": 1, "heading": 1},
    }
    stand_ins: dict[str, object] = {
        name: terms.CommandValues(width)
        for name, width in widths.get(policy.twist, {}).items()
    }
    if policy.twist == "posture":
        stand_ins["posture"] = terms.CommandValues(1)
    if policy.head:
        stand_ins["head_pose"] = terms.CommandValues(4)
    if policy.body:
        stand_ins["body_pose"] = terms.CommandValues(3)
    return stand_ins


def setup_builder() -> mjswan.Builder:
    rl_root = _resolve_rl_root()
    deploy_root = _resolve_deploy_root()
    stand_pose = _stand_pose(rl_root / SCENE_XML)

    builder = mjswan.Builder()
    project = builder.add_project(name="Microduck")

    for index, demo in enumerate(DEMOS):

        def spec_fn(demo: _Demo = demo) -> mujoco.MjSpec:
            return _scene_spec(
                rl_root / demo.xml,
                stand_pose,
                root_height=demo.root_height,
                wheels=demo.wheels,
                ball_y=demo.ball_y,
            )

        def trace_spec_fn(demo: _Demo = demo) -> mujoco.MjSpec:
            return _scene_spec(
                rl_root / (demo.trace_xml or demo.xml),
                stand_pose,
                root_height=demo.root_height,
                wheels=demo.wheels,
                ball_y=None if demo.trace_xml else demo.ball_y,
                sim_options=False,
            )

        spec = spec_fn()
        joint_names = _servo_joints(spec.compile())
        joints = SceneEntityCfg(
            name=ENTITY, joint_names=tuple(joint_names), preserve_order=True
        )

        scene = project.add_scene(name=demo.name, spec=spec, control_dt=CONTROL_DT)
        scene.set_viewer(
            mjswan.ViewerConfig(
                # Three-quarter view from the front: 25 cm of robot, and its face is
                # the half worth watching.
                origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
                body_name=TRACKED_BODY,
                distance=0.8,
                elevation=-12.0,
                azimuth=40.0,
            )
        )
        scene.set_trace_env(
            build_single_entity_trace_env(
                trace_spec_fn, entity_name=ENTITY, commands=_trace_commands(demo)
            )
        )

        terminations = {}
        if demo.fell_over:
            # `make_velocity_env_cfg`'s own, which upstream keeps for this task.
            terminations["fell_over"] = TerminationTermCfg(
                func=term_fns.bad_orientation,
                params={"limit_angle": math.radians(FELL_OVER_ANGLE_DEG)},
            )
        scene.add_policy(
            name=demo.name,
            policy=onnx.load(str(deploy_root / POLICY_DIR / demo.onnx)),
            commands={**_twist_commands(demo.twist), **_pose_commands(demo)},
            observations=_observations(demo, joints),
            actions={
                # The XML's own `<position>` actuators: the browser reads their gains
                # off the compiled model, so there is no PD to configure here.
                "joint_pos": JointPositionActionCfg(
                    entity_name="",
                    actuator_names=(".*",),
                    scale=1.0,
                    use_default_offset=True,
                )
            },
            terminations=terminations,
            policy_joint_names=joint_names,
            default_joint_pos=[stand_pose[name] for name in joint_names],
            default=index == 0,
        )

    return builder
