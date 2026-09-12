"""Playback adjustments to myosuite's mimic terms. None of their math is rewritten.

1. The clip cache is keyed by entity instead of ``id(env)``: the tracer wraps the env in a
   fresh proxy per pass, and each proxy would otherwise build its own clip source with
   its own random phase.
2. The clip frame depends on sim time alone. Training draws a random start frame per
   episode; the browser resets ``mjData.time`` to 0, so the two would disagree.
3. The reset event goes to clip frame 0 instead of a random frame, matching (2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as _mimic
from myosuite.envs.myo.backends.mjlab.clip_trajectory_source import (
    ClipTrajectorySource,
)

ENTITY = "mimic_fullbody_robot"


class _SharedCache(dict):
    """myosuite keys its cache on ``(id(env), entity, variant)``; drop the env."""

    @staticmethod
    def _key(key: tuple) -> tuple:
        return key[1:]

    def __contains__(self, key: object) -> bool:
        return super().__contains__(self._key(key))  # type: ignore[arg-type]

    def __getitem__(self, key: tuple):
        return super().__getitem__(self._key(key))

    def __setitem__(self, key: tuple, value) -> None:
        super().__setitem__(self._key(key), value)


_mimic._mimic_mjlab_cache = _SharedCache()

ClipTrajectorySource._frame_indices = (  # type: ignore[method-assign]
    lambda self, t: (t / self.ctrl_dt).long() % self.n_frames
)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by the (w, x, y, z) quaternion ``q``."""
    w, xyz = q[0], q[1:]
    return v + 2.0 * torch.cross(xyz, torch.cross(xyz, v, dim=0) + w * v, dim=0)


def make_reset(clip_path: Path) -> Callable[[Any, Any], None]:
    """A reset event that puts the body on clip frame 0.

    Same writes as myosuite's ``_mimic_rsi_event`` (root state, then joint state) with
    one correction: MuJoCo keeps a free joint's angular velocity in the body frame, while
    mjlab's ``write_root_state_to_sim`` takes it in the world frame. myosuite passes the
    clip's ``qvel[3:6]`` through unrotated, and this clip starts at ~1.1 rad/s with the
    root turned ~120°, so every reset would begin with the wrong spin. No RNG, so it
    traces to a constant-output graph.
    """
    clip = np.load(clip_path)
    qpos0 = torch.as_tensor(clip["qpos"][0], dtype=torch.float32)
    qvel0 = torch.as_tensor(clip["qvel"][0], dtype=torch.float32)
    root_state = torch.cat([qpos0[:7], qvel0[:3], _quat_rotate(qpos0[3:7], qvel0[3:6])])

    def reset_to_clip_start(env: Any, env_ids: Any) -> None:
        n = env.num_envs if env_ids is None else len(env_ids)
        entity = env.scene[ENTITY]
        entity.write_root_state_to_sim(root_state.expand(n, 13), env_ids=env_ids)
        entity.write_joint_state_to_sim(
            qpos0[7:].expand(n, -1), qvel0[6:].expand(n, -1), env_ids=env_ids
        )

    return reset_to_clip_start
