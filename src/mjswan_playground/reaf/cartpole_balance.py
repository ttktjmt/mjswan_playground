"""`CartpoleBalance`: the cart keeps the pole upright, from close to upright."""

from __future__ import annotations

import mjswan

from . import _cartpole

#: The policy `train.py` trains on `CartpoleBalance`, as its repository and revision;
#: None until it is on the Hub.
POLICY: tuple[str, str] | None = None


def add_scenes(project: mjswan.ProjectHandle) -> None:
    _cartpole.add_scene(
        project,
        "CartpoleBalance",
        env_cfg=_cartpole.env_cfg(swing_up=False),
        policy=POLICY,
        policy_name="Balance",
    )
