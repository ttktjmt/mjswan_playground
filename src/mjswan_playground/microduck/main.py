"""Microduck: the nine policies the robot ships, on the four scenes they run in.

The robot XMLs already carry the position actuators the real servos run, so the scenes
compile straight from them rather than from upstream's training envs. See ``README.md``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
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

#: The final command ranges the curricula reach — what a finished policy has seen — as
#: the sliders that drive them: ``(name, label, range)``.
TWIST_SLIDERS = (
    ("lin_vel_x", "Forward (m/s)", (-0.4, 0.4)),
    ("lin_vel_y", "Sideways (m/s)", (-0.3, 0.3)),
    ("ang_vel_z", "Turn (rad/s)", (-1.0, 1.0)),
)
THROTTLE_SLIDERS = (("lin_vel_x", "Forward (m/s)", (-0.5, 0.6)),)
#: ``infer_policy.py``'s deploy range for the heading slot; see README.
HEADING_SLIDERS = (("heading_error", "Error (rad)", (-1.0, 1.0)),)
HEAD_SLIDERS = (
    ("neck_pitch", "Neck Pitch", (-1.10, 1.10)),
    ("head_pitch", "Head Pitch", (-1.10, 1.10)),
    ("head_yaw", "Head Yaw", (-1.40, 1.40)),
    ("head_roll", "Head Roll", (-0.31, 0.31)),
)
_TILT = math.radians(15)
#: Only z / roll / pitch are steerable; the body slot's x, y and yaw are alive-range noise
#: upstream never trains as a command, so they are padded with the zero they centre on.
BODY_SLIDERS = (
    ("z", "Trunk Height (m)", (-0.04, 0.030)),
    ("roll", "Trunk Roll (rad)", (-_TILT, _TILT)),
    ("pitch", "Trunk Pitch (rad)", (-_TILT, _TILT)),
)

#: Trace-time widths for the commands the browser owns. One table for every scene: the
#: tracer only resolves the names a policy's own terms read, so a spare entry costs
#: nothing. The phase clock is traced for real, but the term reading it still resolves
#: its name against the trace env — as husky's does.
TRACE_COMMAND_WIDTHS = {
    "twist": 3,
    "throttle": 1,
    "heading": 1,
    "posture": 1,
    "head_pose": 4,
    "body_pose": 3,
}


@dataclass(frozen=True)
class _Policy:
    """One shipped checkpoint, and what drives each of its command slots."""

    name: str
    onnx: str
    twist: str
    """Which of five shapes upstream put in this policy's twist slot: ``"velocity"``,
    ``"heading"``, ``"posture"``, ``"phase"`` or ``"zero"``."""
    head: bool = False
    body: bool = False
    fell_over: bool = True


@dataclass(frozen=True)
class _Scene:
    """One upstream XML, and the policies that run in it.

    Nine policies over four scenes: mjswan traces each policy's terms into its own
    ``mdp/<policy>/``, so they only need splitting where the *scene* differs — wheels,
    or a ball baked in at one of two placements.
    """

    name: str
    xml: str
    policies: tuple[_Policy, ...]
    root_height: float = STAND_HEIGHT
    wheels: bool = False
    ball_y: float | None = None


SCENES = (
    _Scene(
        "Duck",
        SCENE_XML,
        (
            _Policy("Walk", "alpha_walking.onnx", "velocity", head=True),
            _Policy(
                "Stand & Pose",
                "alpha_stand.onnx",
                "zero",
                head=True,
                body=True,
                fell_over=False,
            ),
            _Policy(
                "Sit / Stand",
                "alpha_sitstand.onnx",
                "posture",
                head=True,
                fell_over=False,
            ),
            _Policy("Ground Pick", "alpha_ground_pick.onnx", "phase"),
            _Policy("Roulade", "roulade.onnx", "zero", fell_over=False),
        ),
    ),
    _Scene(
        "Ball (right foot)",
        SCENE_BALL_XML,
        (_Policy("Kick", "ball_kick_right.onnx", "zero"),),
        ball_y=-BALL_OFFSET_ABS_Y,
    ),
    _Scene(
        "Ball (left foot)",
        SCENE_BALL_XML,
        (_Policy("Kick", "ball_kick_left.onnx", "zero"),),
        ball_y=BALL_OFFSET_ABS_Y,
    ),
    _Scene(
        "Rollers",
        SCENE_ROLLERS_XML,
        (
            _Policy("Skate", "roller.onnx", "heading"),
            _Policy("Crouch", "roller_crouch.onnx", "phase"),
        ),
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
    )


def _resolve_deploy_root() -> Path:
    return ensure_repo(
        name="microduck",
        url=DEPLOY_REPO_URL,
        commit=DEPLOY_REPO_COMMIT,
        marker=f"{POLICY_DIR}/alpha_walking.onnx",
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


def _free_joint_adr(model: mujoco.MjModel, name: str) -> slice:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise ValueError(f"No joint named {name!r} in the compiled scene.")
    start = int(model.jnt_qposadr[joint])
    return slice(start, start + 7)


def _scene_spec(
    scene: _Scene,
    rl_root: Path,
    stand_pose: dict[str, float],
    *,
    tracing: bool = False,
) -> mujoco.MjSpec:
    """The scene as the browser compiles it: upstream's solver settings (the XMLs carry
    no ``<option>``), the wheel friction its inference script writes by hand, and one
    keyframe.

    STAND has to be the *only* keyframe: the browser resets to the first one, and so does
    the tracing env, whose ``default_joint_pos`` ``joint_pos_rel`` bakes in. Upstream's
    INIT first would pose every policy against a zero it never trained on.

    ``tracing`` builds the variant the ONNX tracer runs against instead: no sim options,
    since the scene's own win that conflict and nothing traced reads them, and no ball,
    since an mjlab ``Entity`` is one freejoint and a ball scene holds a second.
    """
    xml, ball_y = scene.xml, scene.ball_y
    if tracing and ball_y is not None:
        xml, ball_y = SCENE_XML, None

    spec = mujoco.MjSpec.from_file(str(rl_root / xml))
    if not tracing:
        spec.option.timestep = TIMESTEP
        spec.option.integrator = INTEGRATOR
        spec.option.iterations = SOLVER_ITERATIONS
        spec.option.ls_iterations = SOLVER_LS_ITERATIONS
    if scene.wheels:
        for joint in spec.joints:
            if joint.name.startswith("passive_"):
                joint.frictionloss = WHEEL_FRICTIONLOSS

    model = spec.compile()
    qpos = np.array(model.qpos0, dtype=float)
    qpos[_free_joint_adr(model, ROOT_JOINT)] = [
        0.0,
        0.0,
        scene.root_height,
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


def _sliders(
    specs: Iterable[tuple[str, str, tuple[float, float]]],
) -> mjswan.CommandInput:
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


def _driven(command_name: str) -> ObservationTermCfg:
    """Slots the browser's command fills."""
    return ObservationTermCfg(
        func=obs_fns.generated_commands, params={"command_name": command_name}
    )


def _padded(dim: int) -> ObservationTermCfg:
    """Slots nothing drives: upstream's ``zero_command_padding``, which is what keeps one
    runtime able to swap any of these policies in."""
    return ObservationTermCfg(func=terms.zeros, params={"dim": dim})


def _twist(
    kind: str,
) -> tuple[dict[str, mjswan.CommandInput], dict[str, ObservationTermCfg]]:
    """The commands this policy's twist slot needs and the three values they fill."""
    if kind == "velocity":
        return {"twist": _sliders(TWIST_SLIDERS)}, {"twist": _driven("twist")}
    if kind == "heading":
        # Two commands, not one: `lin_vel_y` is the middle slot, which upstream's roller
        # env pins to (0, 0).
        return (
            {
                "throttle": _sliders(THROTTLE_SLIDERS),
                "heading": _sliders(HEADING_SLIDERS),
            },
            {
                "throttle": _driven("throttle"),
                "lin_vel_y": _padded(1),
                "heading": _driven("heading"),
            },
        )
    if kind == "posture":
        return (
            {
                "posture": mjswan.ui_command(
                    [mjswan.CheckboxConfig(name="sit", label="Sit", default=False)]
                )
            },
            {"posture": _driven("posture"), "twist_pad": _padded(2)},
        )
    if kind == "phase":
        return (
            {"twist": terms.GroundPickPhaseCommandCfg(control_dt=CONTROL_DT)},
            {"twist": _driven("twist")},
        )
    if kind != "zero":
        raise ValueError(f"Unknown twist kind {kind!r}")
    return {}, {"twist": _padded(3)}


def _pose(
    policy: _Policy,
) -> tuple[dict[str, mjswan.CommandInput], dict[str, ObservationTermCfg]]:
    """The head (4) and body (6) slots, driven where this policy steers them."""
    commands: dict[str, mjswan.CommandInput] = {}
    if policy.head:
        commands["head_pose"] = _sliders(HEAD_SLIDERS)
    obs = {"head_command": _driven("head_pose") if policy.head else _padded(4)}
    if policy.body:
        commands["body_pose"] = _sliders(BODY_SLIDERS)
        # [x, y, z, roll, pitch, yaw] — see `BODY_SLIDERS`.
        obs["body_xy"] = _padded(2)
        obs["body_command"] = _driven("body_pose")
        obs["body_yaw"] = _padded(1)
    else:
        obs["body_command"] = _padded(6)
    return commands, obs


def _mdp(
    policy: _Policy, joints: SceneEntityCfg
) -> tuple[dict[str, mjswan.CommandInput], ObservationGroupCfg]:
    """The policy's control panel, and the 61 values ``robotd`` builds in its order: 3
    gyro + 3 gravity + 14 joint positions + 14 joint velocities + 14 last actions + a
    13-wide command block."""
    twist_commands, twist_obs = _twist(policy.twist)
    pose_commands, pose_obs = _pose(policy)
    return {**twist_commands, **pose_commands}, ObservationGroupCfg(
        terms={
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
            **twist_obs,
            **pose_obs,
        }
    )


def setup_builder() -> mjswan.Builder:
    rl_root = _resolve_rl_root()
    deploy_root = _resolve_deploy_root()
    stand_pose = _stand_pose(rl_root / SCENE_XML)

    builder = mjswan.Builder()
    project = builder.add_project(name="Microduck")

    for entry in SCENES:
        spec = _scene_spec(entry, rl_root, stand_pose)
        joint_names = _servo_joints(spec.compile())
        joints = SceneEntityCfg(
            name=ENTITY, joint_names=tuple(joint_names), preserve_order=True
        )

        scene = project.add_scene(name=entry.name, spec=spec, control_dt=CONTROL_DT)
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
                partial(_scene_spec, entry, rl_root, stand_pose, tracing=True),
                entity_name=ENTITY,
                commands={
                    name: terms.CommandValues(width)
                    for name, width in TRACE_COMMAND_WIDTHS.items()
                },
            )
        )

        for policy in entry.policies:
            terminations = {}
            if policy.fell_over:
                # `make_velocity_env_cfg`'s own, which upstream keeps for this task.
                terminations["fell_over"] = TerminationTermCfg(
                    func=term_fns.bad_orientation,
                    params={"limit_angle": math.radians(FELL_OVER_ANGLE_DEG)},
                )
            commands, observations = _mdp(policy, joints)
            scene.add_policy(
                name=policy.name,
                policy=onnx.load(str(deploy_root / POLICY_DIR / policy.onnx)),
                commands=commands,
                observations=observations,
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
            )

    return builder
