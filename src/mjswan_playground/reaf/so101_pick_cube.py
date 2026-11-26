"""`SO101PickCube`: the SO-101 lifting a cube to a target that moves, from its state."""

from __future__ import annotations

import mjswan

from . import _pick_cube

#: Trained on Playground 1.0.1's `SO101PickCube` by mjswan's
#: `examples/train/so101_pick_cube.py`.
POLICY = ("ttktjmt/so101-pick-cube", "fcdf0776a21b5f7560f129192e8a87907931e92f")


def add_scenes(project: mjswan.ProjectHandle) -> None:
    _pick_cube.add_scene(
        project,
        "SO101PickCube",
        env_cfg=_pick_cube.env_cfg(),
        policy=POLICY,
        policy_name="PickCube",
    )
