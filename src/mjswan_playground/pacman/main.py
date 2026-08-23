"""PAC-MAN dodgeball demo, in two scenes. See ``README.md``."""

from __future__ import annotations

from typing import Any

import mjswan
import onnx
from mjlab.tasks.registry import load_env_cfg
from mjswan.adapters import DEFAULT_OBS_GROUP_KEY, adapt_observations
from mjswan.managers.event_manager import EventTermCfg
from mjswan.managers.observation_manager import ObservationTermCfg

from . import terms, upstream

DODGE_TASK_ID = "Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat"
DODGE_POLICY_ONNX = "deploy/ckpts/dodge_link_cbf.onnx"
WALK_TASK_ID = "Unitree-G1-AMP-Flat"
WALK_POLICY_ONNX = "deploy/ckpts/walk_policy.onnx"

CAMERA = "head_camera_single"
COMMAND_NAME = "twist"
#: The group a policy reads, and the group the dodge task renders its image into.
ACTOR_GROUP = "actor"
DEPTH_GROUP = "depth"
DEPTH_TERM = "head_depth"
#: The rendering camera sensor, which is what the browser cannot serve.
CAMERA_SENSOR = "head_depth_single"
BALL_GEOM = "ball_collision"

#: Walking, not standing, is what the walk policy is here to show. Well inside the play
#: config's own `lin_vel_x` range, which its sliders take verbatim.
DEFAULT_FORWARD_SPEED = 0.5

#: Reference-state initialization from the AMP clips — see README.
MOTION_EVENTS = ("init_motion_loader", "reset_from_motion")
#: Dodge events with nothing to trace, or nothing to trace them from — see README.
DROPPED_DODGE_EVENTS = MOTION_EVENTS + ("throw_ball_on_dwell", "randomize_ball_size")
#: A dwell counter mjswan has no state for; `bad_base_height` still catches the fall.
DROPPED_DODGE_TERMINATIONS = ("collapsed_crouch",)

#: Upstream's throw geometry, under this task's names for it.
_THROW_PARAMS = {
    "dist_range": "dist_range",
    "height_range": "height_range",
    "angle_deg": "angle_deg",
    "flight_time_range": "flight_time_range",
    "high_throw_fraction": "high_fraction",
    "high_launch_height_range": "high_launch_height_range",
    "high_target_z_range": "high_target_z_range",
    "aim_noise_scale": "aim_noise",
    "lead_target": "lead_target",
}
#: Depth-term params this task reproduces itself; anything else upstream sets is domain
#: randomization, which the browser's clean image does not have.
_DEPTH_KEYS = (
    "sensor_name",
    "near",
    "far",
    "flatten",
    "ball_geom_name",
    "update_period",
)


def _depth_geometry(params: dict[str, Any]) -> dict[str, Any]:
    """``near`` / ``far`` from upstream's own depth term.

    Everything else it can carry is either already true here (``flatten``) or a
    training-time perturbation of the image, and a silently ignored perturbation is a
    policy fed something it was not shown. So refuse instead.
    """
    extra = {key: value for key, value in params.items() if key not in _DEPTH_KEYS}
    if any(extra.values()):
        raise ValueError(
            f"Upstream's depth term carries domain randomization this task cannot "
            f"reproduce ({sorted(k for k, v in extra.items() if v)}). The browser image "
            "is the clean one its play config builds — unset BALLONLY_AUG / "
            "BALLONLY_DR_SCALE, or extend `terms.ball_depth`."
        )
    if int(params.get("update_period", 1)) != 1:
        raise ValueError(
            "Upstream's depth term samples the camera every "
            f"{params['update_period']} control steps and holds it in between; this "
            "task computes a fresh image every step. Set DEPTH_UPDATE_PERIOD=1."
        )
    return {"near": float(params["near"]), "far": float(params["far"])}


def _model_geometry(env_cfg: Any) -> dict[str, float]:
    """The camera's field of view and the ball's radius, off the specs they live in.

    A traced term is handed the simulation state, not the model behind it, so anything
    model-derived has to be resolved here and baked into the graph as a constant.
    """
    robot_spec = env_cfg.scene.entities["robot"].spec_fn()
    ball_spec = env_cfg.scene.entities["ball"].spec_fn()
    return {
        "fovy": float(robot_spec.camera(CAMERA).fovy),
        "ball_radius": float(ball_spec.geom(BALL_GEOM).size[0]),
    }


def _strip_untraceable(env_cfg: Any) -> None:
    """Drop what the browser has no counterpart for, and defang what it can keep."""
    for name in DROPPED_DODGE_EVENTS:
        env_cfg.events.pop(name, None)
    for name in DROPPED_DODGE_TERMINATIONS:
        env_cfg.terminations.pop(name, None)
    # The reset keeps parking the ball aside; its optional throw countdown is a
    # `torch.randint` the tracer cannot record, and the interval event owns the timing now.
    env_cfg.events["reset_dodge_state"].params["throw_interval_range"] = None
    # The hit is the contact sensor's alone: the velocity-discontinuity fallback compares
    # against the previous step's ball velocity, which it keeps on the env.
    env_cfg.terminations["ball_hit"].params["delta_v_threshold"] = 0.0
    # The rendering camera goes, and with it every group that reads it. Only the actor's
    # is exported anyway, and the critic's belief gate asks the camera sensor for its
    # index — so leaving those groups in fails the tracing env on a sensor that is gone.
    env_cfg.scene.sensors = tuple(
        sensor
        for sensor in (env_cfg.scene.sensors or ())
        if sensor.name != CAMERA_SENSOR
    )
    env_cfg.observations = {ACTOR_GROUP: env_cfg.observations[ACTOR_GROUP]}


def _add_dodge_scene(project: mjswan.ProjectHandle, root, contract) -> None:
    """The paper's regime: a ball every 1–4 s, seen only as a depth image."""
    env_cfg = load_env_cfg(DODGE_TASK_ID, play=True)
    upstream_throw = env_cfg.events["throw_ball_on_dwell"].params
    throw_params = {
        ours: upstream_throw[theirs]
        for theirs, ours in _THROW_PARAMS.items()
        if theirs in upstream_throw
    }
    throw_interval = tuple(upstream_throw["throw_interval_range"])
    depth_params = {
        **_depth_geometry(env_cfg.observations[DEPTH_GROUP].terms[DEPTH_TERM].params),
        **_model_geometry(env_cfg),
    }

    _strip_untraceable(env_cfg)
    terms.add_camera_pose_sensors(env_cfg.scene.entities["robot"], CAMERA)
    env_cfg.events["throw_ball"] = EventTermCfg(
        func=terms.throw_ball,
        mode="interval",
        interval_range_s=throw_interval,
        params=throw_params,
    )

    # One group, not two: the actor reads `("actor", "depth")` concatenated, and mjswan
    # feeds one vector per ONNX input. Adapting upstream's own group keeps its terms and
    # their order rather than restating them here; the image goes last, as the
    # checkpoint's 384 + 576 layout has it.
    observations = adapt_observations(env_cfg.observations[ACTOR_GROUP])[
        DEFAULT_OBS_GROUP_KEY
    ]
    observations.terms[DEPTH_TERM] = ObservationTermCfg(
        func=terms.ball_depth,
        params=depth_params,
        history_steps=tuple(contract.DEFAULT_FRAME_OFFSETS),
    )

    scene = project.add_scene_mjlab(DODGE_TASK_ID, env_cfg=env_cfg)
    scene.add_policy(
        name="Link-CBF Dodge",
        policy=onnx.load(str(root / DODGE_POLICY_ONNX)),
        observations=observations,
        # The three numbers the policy reads as its velocity command, held at zero and
        # given no controls: that is the deployed dodge mode, which ignores the
        # operator's velocity outright. Upstream's sim-play command — a goal tracker
        # wrapped in a ball-avoiding CBF filter — is a training-time construct the
        # hardware never runs. See README.
        commands={COMMAND_NAME: mjswan.ui_command([])},
        policy_joint_names=[f"robot/{name}" for name in contract.POLICY_JOINT_NAMES],
        default_joint_pos=[float(value) for value in contract.DEFAULT_POS],
        default=True,
    )


def _add_walk_scene(project: mjswan.ProjectHandle, root, contract) -> None:
    """The locomotion half of the same stack, on flat ground and on the sliders."""
    env_cfg = load_env_cfg(WALK_TASK_ID, play=True)
    for name in MOTION_EVENTS:
        env_cfg.events.pop(name, None)

    scene = project.add_scene_mjlab(WALK_TASK_ID, env_cfg=env_cfg)
    ranges = env_cfg.commands[COMMAND_NAME].ranges
    scene.add_policy(
        name="AMP Walk",
        policy=onnx.load(str(root / WALK_POLICY_ONNX)),
        commands={
            COMMAND_NAME: mjswan.velocity_command(
                lin_vel_x=ranges.lin_vel_x,
                lin_vel_y=ranges.lin_vel_y,
                ang_vel_z=ranges.ang_vel_z,
                default_lin_vel_x=DEFAULT_FORWARD_SPEED,
            )
        },
        policy_joint_names=[f"robot/{name}" for name in contract.POLICY_JOINT_NAMES],
        default_joint_pos=[float(value) for value in contract.DEFAULT_POS],
        default=True,
    )


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    contract = upstream.deployed_contract(root)

    builder = mjswan.Builder()
    project = builder.add_project(name="PAC-MAN")
    # Dodging first: it is what the paper is about, and the scene the viewer opens on.
    _add_dodge_scene(project, root, contract)
    _add_walk_scene(project, root, contract)
    return builder
