"""The two terms the dodge task cannot take from upstream as they are. See ``README.md``.

* :func:`ball_depth` — the browser has no render, so the image is ray-sphere
  intersections instead (as upstream's own ``web-demo`` branch does).
* :func:`throw_ball` — upstream's launch geometry, on an interval event: mjswan has no
  ``mode="step"`` for its per-env countdown.
* :func:`add_camera_pose_sensors` — frame sensors giving the term the camera pose.
"""

from __future__ import annotations

import math

import mujoco
import torch
from mjlab.envs.mdp import observations as obs_fns

# `sample_uniform` must be a module global: mjswan's RNG spy patches these to record
# draws, and an unseen draw is baked in as a constant — the same throw, every time.
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    sample_uniform,
    yaw_quat,
)

POS_SENSOR = "head_camera_pos"
QUAT_SENSOR = "head_camera_quat"


def add_camera_pose_sensors(
    entity_cfg,
    camera_name: str,
    pos_sensor: str = POS_SENSOR,
    quat_sensor: str = QUAT_SENSOR,
) -> None:
    """Add ``framepos`` / ``framequat`` sensors on ``camera_name``, in place.

    The depth term's only route to the camera pose: ``cam_xpos`` is not an mjswan slot,
    a sensor window is — and the model keeps the mount offset and +20° tilt.
    """
    inner_spec_fn = entity_cfg.spec_fn

    def spec_fn():
        spec = inner_spec_fn()
        for name, sensor_type in (
            (pos_sensor, mujoco.mjtSensor.mjSENS_FRAMEPOS),
            (quat_sensor, mujoco.mjtSensor.mjSENS_FRAMEQUAT),
        ):
            spec.add_sensor(
                name=name,
                type=sensor_type,
                objtype=mujoco.mjtObj.mjOBJ_CAMERA,
                objname=camera_name,
            )
        return spec

    entity_cfg.spec_fn = spec_fn


def _pixel_rays(
    width: int,
    height: int,
    subsample: int,
    fovy_deg: float,
    device: torch.device,
) -> torch.Tensor:
    """Unit ray directions in the camera frame, one per sub-sample: ``[H*W*s*s, 3]``.

    MuJoCo cameras look down -z, +x right, +y up; ``fovy`` is vertical, so the
    horizontal half-extent scales by the aspect ratio. Row-major from the top.

    The caller min-pools ``subsample`` rays per pixel axis, as the deployed stack does
    (full-res ZED depth, masked, min-pooled to 9×16). One ray per pixel centre misses
    a 0.076 m ball until ~1.65 m — most of the reaction window gone.
    """
    tan_v = math.tan(math.radians(fovy_deg) / 2.0)
    tan_h = tan_v * width / height
    shape = {"device": device, "dtype": torch.float32}
    row = torch.arange(height, **shape).view(height, 1, 1, 1)
    col = torch.arange(width, **shape).view(1, width, 1, 1)
    sub_row = torch.arange(subsample, **shape).view(1, 1, subsample, 1)
    sub_col = torch.arange(subsample, **shape).view(1, 1, 1, subsample)
    # Sub-sample centres inside each pixel's footprint.
    fx = col + (sub_col + 0.5) / subsample
    fy = row + (sub_row + 0.5) / subsample
    x, y = torch.broadcast_tensors(
        (2.0 * fx / width - 1.0) * tan_h,
        (1.0 - 2.0 * fy / height) * tan_v,
    )
    rays = torch.stack([x, y, -torch.ones_like(x)], dim=-1).reshape(-1, 3)
    return rays / rays.norm(dim=-1, keepdim=True)


def ball_depth(
    env,
    fovy: float,
    ball_radius: float,
    pos_sensor: str = f"robot/{POS_SENSOR}",
    quat_sensor: str = f"robot/{QUAT_SENSOR}",
    ball_name: str = "ball",
    width: int = 16,
    height: int = 9,
    subsample: int = 3,
    near: float = 0.1,
    far: float = 5.0,
) -> torch.Tensor:
    """Ball-only masked depth, normalized to ``[0, 1]``: ``[B, height * width]``.

    Ball pixels carry their depth, every other pixel reads ``far``. The value is the
    **perpendicular** depth (hit projected on the optical axis), which is what the
    ``mujoco_warp`` sensor and a stereo depth map report — median disagreement 0.017 m
    against 0.121 m for the ray distance. Normalization is upstream's
    ``depth_metres_to_obs``: below ``near`` reads as ``far``, then clamp and scale.

    Self-occlusion is not modelled, so the view is only ever cleaner than training's —
    inside the distribution, given its per-pixel and whole-ball dropout.

    ``fovy`` and ``ball_radius`` are baked in at build time: a traced term sees state,
    not the model.
    """
    cam_pos = obs_fns.builtin_sensor(env, sensor_name=pos_sensor)
    cam_quat = obs_fns.builtin_sensor(env, sensor_name=quat_sensor)
    ball_pos = env.scene[ball_name].data.root_link_pos_w
    rays = _pixel_rays(width, height, subsample, fovy, cam_pos.device)

    # Ball centre in the camera frame: one rotation, not one per ray.
    centre = quat_apply_inverse(cam_quat, ball_pos - cam_pos)
    along = centre @ rays.transpose(0, 1)  # [B, N] centre projected on each ray
    offset = (centre * centre).sum(-1, keepdim=True) - ball_radius * ball_radius
    disc = along * along - offset
    root = torch.sqrt(disc.clamp(min=0.0))
    # Near intersection, or the far one when the camera sits inside the sphere.
    hit_near = along - root
    distance = torch.where(hit_near > 0.0, hit_near, along + root)
    empty = torch.full_like(distance, far)
    # Onto the optical axis, before the mask: -z is the ray's forward component.
    depth = distance * (-rays[:, 2]).unsqueeze(0)
    depth = torch.where((disc > 0.0) & (distance > 0.0), depth, empty)

    per_pixel = depth.reshape(-1, height, width, subsample, subsample).amin(dim=(3, 4))
    per_pixel = torch.where(
        per_pixel < near, torch.full_like(per_pixel, far), per_pixel
    ).clamp(near, far)
    return ((per_pixel - near) / (far - near)).reshape(per_pixel.shape[0], -1)


def throw_ball(
    env,
    env_ids,
    ball_name: str = "ball",
    robot_name: str = "robot",
    dist_range: tuple[float, float] = (2.0, 3.0),
    height_range: tuple[float, float] = (1.5, 2.3),
    angle_deg: float = 25.0,
    flight_time_range: tuple[float, float] = (0.58, 0.63),
    high_fraction: float = 0.5,
    high_launch_height_range: tuple[float, float] = (0.4, 0.9),
    high_target_z_range: tuple[float, float] = (0.9, 1.1),
    aim_noise: float = 0.1,
    lead_target: bool = True,
    gravity: float = 9.81,
) -> None:
    """Launch the ball at the robot, teleporting it to a fresh launch point.

    Upstream's ``throw_ball_on_dwell`` minus the trigger (an interval event fires it).
    Two threat types, mixed as its play config mixes them: a **descending** ball (high,
    ``vz0 = 0``) falls across the body, a **low-arc** ball (low, upward ``vz0``) arrives
    at torso height and must be ducked. Launch stays in the frontal cone so the head
    camera sees it; only the aim point is led and jittered.
    """
    robot = env.scene[robot_name]
    ball = env.scene[ball_name]
    device = robot.data.root_link_pos_w.device
    draw = {"size": (1, 1), "device": device}

    root_pos = robot.data.root_link_pos_w
    yaw = yaw_quat(robot.data.root_link_quat_w)

    # Launch point: `dist` ahead in the yaw frame, `angle` off the heading.
    dist = sample_uniform(dist_range[0], dist_range[1], **draw)
    angle = sample_uniform(-angle_deg, angle_deg, **draw) * (math.pi / 180.0)
    offset_b = torch.cat(
        [dist, dist * torch.tan(angle), torch.zeros_like(dist)], dim=-1
    )
    start_xy = root_pos[:, :2] + quat_apply(yaw, offset_b)[:, :2]

    high = sample_uniform(0.0, 1.0, **draw) < high_fraction
    start_z = torch.where(
        high,
        sample_uniform(*high_launch_height_range, **draw),
        sample_uniform(*height_range, **draw),
    )

    # Cap a descending throw so it cannot fall below ~0.05 m before arriving; a low-arc
    # throw rises first, so it uses the whole window.
    requested = sample_uniform(*flight_time_range, **draw)
    ceiling = torch.sqrt(2.0 * (start_z - 0.05).clamp(min=1e-3) / gravity)
    flight = torch.where(high, requested, torch.minimum(requested, ceiling))

    # Aim at the robot's xy, led by its velocity so it cannot walk out of the throw, then
    # jittered so no two throws are identical.
    target_xy = root_pos[:, :2]
    if lead_target:
        target_xy = target_xy + robot.data.root_link_lin_vel_w[:, :2] * flight
    if aim_noise > 0.0:
        target_xy = target_xy + sample_uniform(
            -aim_noise, aim_noise, size=(1, 2), device=device
        )

    target_z = sample_uniform(*high_target_z_range, **draw)
    velocity_xy = (target_xy - start_xy) / flight
    # z0 + vz0*t - g*t^2/2 = z_target at arrival, or a pure horizontal toss.
    velocity_z = torch.where(
        high,
        (target_z - start_z) / flight + 0.5 * gravity * flight,
        torch.zeros_like(flight),
    )

    upright = torch.zeros(1, 4, device=device)
    upright[:, 0] = 1.0
    ball.write_root_link_pose_to_sim(
        torch.cat([start_xy, start_z, upright], dim=-1), env_ids=env_ids
    )
    ball.write_root_link_velocity_to_sim(
        torch.cat([velocity_xy, velocity_z, torch.zeros(1, 3, device=device)], dim=-1),
        env_ids=env_ids,
    )
