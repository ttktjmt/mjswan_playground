"""MuJoCo Playground's RL environments, one scene each, built for the browser. See
``README.md``."""

from __future__ import annotations

import mjswan

from . import (
    cartpole_balance,
    cartpole_balance_vision,
    cartpole_swing_up,
    g1_whole_body_tracking,
    so101_pick_cube,
    so101_pick_cube_vision,
    trossen_pick_cube,
    trossen_pick_cube_vision,
)

#: One part per environment, built on its own for mjswan Cloud and embedded in its own
#: section of reaf-sim's docs; the whole lists them in this order.
PARTS = {
    "reaf-so101-pick-cube": so101_pick_cube.add_scenes,
    "reaf-so101-pick-cube-vision": so101_pick_cube_vision.add_scenes,
    "reaf-cartpole-balance": cartpole_balance.add_scenes,
    "reaf-cartpole-swing-up": cartpole_swing_up.add_scenes,
    "reaf-cartpole-balance-vision": cartpole_balance_vision.add_scenes,
    "reaf-trossen-pick-cube": trossen_pick_cube.add_scenes,
    "reaf-trossen-pick-cube-vision": trossen_pick_cube_vision.add_scenes,
    "reaf-g1-whole-body-tracking": g1_whole_body_tracking.add_scenes,
}


def setup_builder(part: str | None = None) -> mjswan.Builder:
    builder = mjswan.Builder()
    project = builder.add_project(
        name="MuJoCo Playground", license="Apache-2.0", copyright="2026 Google LLC"
    )
    for add_scenes in [PARTS[part]] if part else PARTS.values():
        add_scenes(project)
    return builder
