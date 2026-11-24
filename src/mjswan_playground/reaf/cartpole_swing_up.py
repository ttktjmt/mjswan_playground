"""`CartpoleSwingUp`: the cart swings the pole up from hanging down, then balances it."""

from __future__ import annotations

import mjswan

from . import _cartpole

#: The policy `train.py` trains on `CartpoleSwingUp`, as its repository and revision;
#: None until it is on the Hub.
POLICY: tuple[str, str] | None = None


def add_scenes(project: mjswan.ProjectHandle) -> None:
    _cartpole.add_scene(
        project,
        "CartpoleSwingUp",
        env_cfg=_cartpole.env_cfg(swing_up=True),
        policy=POLICY,
        policy_name="Swing-up",
    )
