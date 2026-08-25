"""PAC-MAN dodgeball demo, in two scenes. See ``README.md``."""

from __future__ import annotations

from typing import Any, get_args

import mjswan
import onnx
from mjlab.tasks.registry import load_env_cfg
from mjswan.adapters import DEFAULT_OBS_GROUP_KEY, adapt_observations
from mjswan.managers.event_manager import EventMode, EventTermCfg
from mjswan.managers.observation_manager import ObservationTermCfg

from . import terms, upstream

DODGE_TASK_ID = "Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat"
DODGE_POLICY_ONNX = "deploy/ckpts/dodge_link_cbf.onnx"
WALK_TASK_ID = "Unitree-G1-AMP-Flat"
WALK_POLICY_ONNX = "deploy/ckpts/walk_policy.onnx"

CAMERA = "head_camera_single"
COMMAND_NAME = "twist"
ACTOR_GROUP = "actor"
DEPTH_GROUP = "depth"
DEPTH_TERM = "head_depth"
#: The rendering camera sensor, which the browser cannot serve.
CAMERA_SENSOR = "head_depth_single"
BALL_GEOM = "ball_collision"

#: Walking, not standing. Inside the play config's own `lin_vel_x` range.
DEFAULT_FORWARD_SPEED = 0.5

#: Reference-state initialization from the AMP clips — see README.
MOTION_EVENTS = ("init_motion_loader", "reset_from_motion")
#: Dodge events with nothing to trace, or nothing to trace them from — see README.
#: `randomize_ball_size` could come back — mjswan describes `dr.geom_size` now — but this
#: scene's depth image is analytic and bakes the radius at build time, so a size drawn in
#: the browser would show the policy a ball it is not facing. It stays dropped until a
#: traced term can read a model field as a slot.
DROPPED_DODGE_EVENTS = MOTION_EVENTS + ("throw_ball_on_dwell", "randomize_ball_size")
#: A dwell counter mjswan has no state for; `bad_base_height` still catches the fall.
DROPPED_DODGE_TERMINATIONS = ("collapsed_crouch",)

#: The throw's two threat types, each on its own button as upstream's play viewer offers
#: them. Name, button label, `high_fraction`: the interval throw keeps upstream's 50/50
#: mix, a button forces its branch.
MANUAL_THROWS = (
    # Upstream's HIGH branch: launched low, rising to torso/head height.
    ("throw_overhead", "Throw overhead", 1.0),
    # Its LOW branch: launched at ~2 m with no upward speed, descending across the legs.
    ("throw_underbody", "Throw underbody", 0.0),
)
#: The interval throw's arm checkbox — upstream's "Pause ball throws" the other way up.
AUTO_THROW_LABEL = "Auto throw"

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
#: Depth-term params this task reproduces; the rest is domain randomization.
_DEPTH_KEYS = (
    "sensor_name",
    "near",
    "far",
    "flatten",
    "ball_geom_name",
    "update_period",
)


def _require_manual_events() -> None:
    """Refuse an engine whose event modes stop at startup / reset / interval.

    An engine that has never heard of `mode="manual"` buckets an unknown mode with the
    reset terms, so the throw buttons would not fail — they would throw on every reset.
    """
    if "manual" not in get_args(EventMode):
        raise RuntimeError(
            f'The pacman task\'s throw buttons need mjswan with `mode="manual"` event '
            f"terms; the installed {mjswan.__version__} has "
            f"{sorted(get_args(EventMode))}. Upgrade mjswan to the release carrying "
            "ttktjmt/mjswan#104 (see pyproject.toml)."
        )


def _depth_geometry(params: dict[str, Any]) -> dict[str, Any]:
    """``near`` / ``far`` from upstream's depth term; refuse what we cannot reproduce."""
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
    """Camera fovy and ball radius off the specs — a traced term sees only state."""
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
    # Untraceable `torch.randint` countdown; the interval event owns the timing now.
    env_cfg.events["reset_dodge_state"].params["throw_interval_range"] = None
    # Contact sensor only: the fallback needs the previous step's ball velocity.
    env_cfg.terminations["ball_hit"].params["delta_v_threshold"] = 0.0
    # Drop the rendering camera and every group reading it; only the actor's ships.
    env_cfg.scene.sensors = tuple(
        sensor
        for sensor in (env_cfg.scene.sensors or ())
        if sensor.name != CAMERA_SENSOR
    )
    env_cfg.observations = {ACTOR_GROUP: env_cfg.observations[ACTOR_GROUP]}


def _add_dodge_scene(project: mjswan.ProjectHandle, root, contract) -> None:
    """The paper's regime: a ball every 1–4 s, or one on demand, seen only as depth."""
    _require_manual_events()
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
        label=AUTO_THROW_LABEL,
    )
    # One graph each: `high_fraction` decides the branch at trace time, not at runtime.
    for name, label, high_fraction in MANUAL_THROWS:
        env_cfg.events[name] = EventTermCfg(
            func=terms.throw_ball,
            mode="manual",
            params={**throw_params, "high_fraction": high_fraction},
            label=label,
        )

    # One group: the actor reads `("actor", "depth")` concatenated, image last.
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
        # Zero, no controls: deployed dodge ignores the operator. See README.
        commands={COMMAND_NAME: mjswan.ui_command([])},
        policy_joint_names=[f"robot/{name}" for name in contract.POLICY_JOINT_NAMES],
        default_joint_pos=[float(value) for value in contract.DEFAULT_POS],
        default=True,
    )


def _add_walk_scene(project: mjswan.ProjectHandle, root, contract) -> None:
    """The locomotion half of the same stack, on flat ground.

    No `commands=`: the scene's `env_cfg` carries upstream's own `twist`, which mjswan
    adapts — resampled as mjlab resamples it, with mjlab's joystick panel over the top.
    """
    env_cfg = load_env_cfg(WALK_TASK_ID, play=True)
    for name in MOTION_EVENTS:
        env_cfg.events.pop(name, None)

    scene = project.add_scene_mjlab(WALK_TASK_ID, env_cfg=env_cfg)
    scene.add_policy(
        name="AMP Walk",
        policy=onnx.load(str(root / WALK_POLICY_ONNX)),
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
    _add_dodge_scene(project, root, contract)
    _add_walk_scene(project, root, contract)
    return builder
