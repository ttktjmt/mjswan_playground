"""`TrossenPickCube`: the Trossen WXAI's right arm lifting a cube to a target that moves,
from its state."""

from __future__ import annotations

import mjswan

from . import _trossen

#: The policy `train.py` trains on `TrossenPickCube`, as its repository and revision;
#: None until it is on the Hub.
POLICY: tuple[str, str] | None = None


def add_scenes(project: mjswan.ProjectHandle) -> None:
    _trossen.add_scene(
        project,
        "TrossenPickCube",
        env_cfg=_trossen.env_cfg(),
        policy=POLICY,
        policy_name="PickCube",
    )
