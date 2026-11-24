"""`CartpoleBalanceVision`: the cart balances the pole from the fixed camera's pixels
alone, its last three frames in grayscale."""

from __future__ import annotations

import math
from typing import Any

import mjswan
import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as lift_mdp

from . import _cartpole

#: The policy `train.py` trains on `CartpoleBalanceVision`, as its repository and
#: revision; None until it is on the Hub.
POLICY: tuple[str, str] | None = None


def _zeros(env: Any) -> torch.Tensor:
    """Playground's `dummy` group: the MLP's own input, empty beside the image."""
    return torch.zeros(env.num_envs, 1, device=env.device)


def _grayscale(env: Any, sensor_name: str) -> torch.Tensor:
    """Playground's `_obs_grayscale`: the mean of the three channels, centred on 0."""
    return lift_mdp.camera_rgb(env, sensor_name).mean(dim=1, keepdim=True) - 0.5


def _fallen(env: Any) -> torch.Tensor:
    """Playground's termination: the pole past horizontal, or the cart off its rail."""
    slider, hinge = env.scene["cartpole"].data.joint_pos.unbind(-1)
    return (hinge.abs() > math.pi / 2) | (slider.abs() > 3.0)


def env_cfg() -> ManagerBasedRlEnvCfg:
    """`CartpoleBalance` at 50 Hz, seen through the fixed camera at 64 x 64 without
    shadows, and reset once the pole falls or 5 s pass. three.js draws the frames, not
    Playground's ray tracer, so their shading differs from what the policy trained on."""
    cfg = _cartpole.env_cfg(swing_up=False)
    cfg.scene.sensors = (
        CameraSensorCfg(
            name="fixed",
            camera_name="cartpole/fixed",
            width=64,
            height=64,
            data_types=("rgb",),
            use_shadows=False,
        ),
    )
    cfg.observations = {
        "dummy": ObservationGroupCfg({"zeros": ObservationTermCfg(func=_zeros)}),
        "pixels": ObservationGroupCfg(
            {
                # Oldest first, as Playground stacks them.
                "frames": ObservationTermCfg(
                    func=_grayscale, params={"sensor_name": "fixed"}, history_length=3
                )
            }
        ),
    }
    cfg.terminations["fallen"] = TerminationTermCfg(func=_fallen)
    cfg.decimation = 2
    cfg.episode_length_s = 5.0
    return cfg


def add_scenes(project: mjswan.ProjectHandle) -> None:
    _cartpole.add_scene(
        project,
        "CartpoleBalanceVision",
        env_cfg=env_cfg(),
        policy=POLICY,
        policy_name="Balance Vision",
        in_keys=["dummy", "pixels"],
    )
