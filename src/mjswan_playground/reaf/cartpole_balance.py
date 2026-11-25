"""`CartpoleBalance`: the cart keeps the pole upright, from close to upright."""

from __future__ import annotations

import mjswan

from . import _cartpole

#: Trained by `train.py` on Playground 1.1.0's `CartpoleBalance`, all 200 iterations.
POLICY: tuple[str, str] | None = (
    "ttktjmt/cartpole-balance",
    "9a0d9f03ca364a801d1bb42cd8197b34f3b632ab",
)


def add_scenes(project: mjswan.ProjectHandle, *, colors: bool = True) -> None:
    """In colour, as it is published on mjswan Cloud."""
    _cartpole.add_scene(
        project,
        "CartpoleBalance",
        env_cfg=_cartpole.env_cfg(swing_up=False, colors=colors),
        policy=POLICY,
        policy_name="Balance",
    )
