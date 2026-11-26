"""`TrossenPickCube`: the Trossen WXAI's right arm lifting a cube to a target that moves,
from its state."""

from __future__ import annotations

import mjswan

from . import _trossen

#: Trained by `train.py` on Playground 1.1.0's `TrossenPickCube`, all 1000 iterations.
POLICY: tuple[str, str] | None = (
    "ttktjmt/trossen-pick-cube",
    "e0b52c5e5a94f4d416e75a4762df7fa889453b53",
)


def add_scenes(project: mjswan.ProjectHandle) -> None:
    _trossen.add_scene(
        project,
        "TrossenPickCube",
        env_cfg=_trossen.env_cfg(),
        policy=POLICY,
        policy_name="PickCube",
    )
