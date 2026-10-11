"""Microduck Playground: every experiment it publishes a policy for, one or more scenes
each, on the same robot and servos as ``microduck``. See ``README.md``."""

from __future__ import annotations

from functools import partial

import mjswan

from . import (
    _common,
    backflip,
    basketball,
    chimney,
    desk_climb,
    long_jump,
    running,
    stilts,
    swing,
)

#: In the order the scenes are listed.
EXPERIMENTS = (
    running.add_scenes,
    swing.add_scenes,
    basketball.add_scenes,
    stilts.add_scenes,
    desk_climb.add_scenes,
    chimney.add_scenes,
    long_jump.add_scenes,
    backflip.add_scenes,
)

#: Parts built on their own for mjswan Cloud, which takes 100 MB where the whole is
#: 175 MB: every experiment once, Stilts at 1.0 m only, then Stilts at every height.
PARTS = {
    "microduck-moves": (
        running.add_scenes,
        swing.add_scenes,
        basketball.add_scenes,
        partial(stilts.add_scenes, heights_cm=(100,)),
    ),
    "microduck-parkour": (
        desk_climb.add_scenes,
        chimney.add_scenes,
        long_jump.add_scenes,
        backflip.add_scenes,
    ),
    "microduck-stilts": (stilts.add_scenes,),
}


def setup_builder(part: str | None = None) -> mjswan.Builder:
    root = _common.resolve_root()
    builder = mjswan.Builder()
    project = builder.add_project(name="Microduck Playground", license=root / "LICENSE")
    project.set_notice(root / "NOTICE")
    for add_scenes in PARTS[part] if part else EXPERIMENTS:
        add_scenes(project, root)
    return builder
