"""`CartpoleSwingUp`: the cart swings the pole up from hanging down, then balances it."""

from __future__ import annotations

import mjswan

from . import _cartpole

#: Trained by `train.py` on Playground 1.1.0's `CartpoleSwingUp`, all 200 iterations.
POLICY: tuple[str, str] | None = (
    "ttktjmt/cartpole-swing-up",
    "75170d81600268afc1530a2126b6b8e44b3a873b",
)


def add_scenes(project: mjswan.ProjectHandle, *, colors: bool = True) -> None:
    """In colour, as it is published on mjswan Cloud."""
    _cartpole.add_scene(
        project,
        "CartpoleSwingUp",
        env_cfg=_cartpole.env_cfg(swing_up=True, colors=colors),
        policy=POLICY,
        policy_name="Swing-up",
    )
