"""Two shims that let in-hand-rotation-mjlab's *robot definition* import under mjlab
1.5.3. The repo was written against mjlab v1.1 (Feb 2026) and has not been updated.

Only the robot constants are borrowed — the actuator SysID numbers, the collision
bitmasks, the mesh embedding — so that this demo runs the real LEAP hand rather than a
hand-retyped copy of 100 calibration constants. The task's own env config is not
imported; it is rebuilt against current mjlab in `leap_inhand_task.py`.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import mjlab.actuator as _actuator
import mjlab.terrains as _terrains
import mjlab.utils.os as _mjlab_os
from mjlab.actuator import IdealPdActuatorCfg

IN_HAND_SRC = Path(__file__).resolve().parent / "in-hand-rotation-mjlab" / "src"


def _delayed_actuator_cfg(
    *,
    base_cfg: IdealPdActuatorCfg,
    delay_target: str = "position",
    delay_min_lag: int = 0,
    delay_max_lag: int = 0,
    delay_hold_prob: float = 0.0,
    delay_update_period: int = 0,
    delay_per_env_phase: bool = True,
):
    """mjlab v1.2 folded the `DelayedActuatorCfg` wrapper into `ActuatorCfg` itself.

    `delay_target` has no field any more — position is the only thing mjlab delays now,
    which is what every call site here asked for.
    """
    del delay_target
    return dataclasses.replace(
        base_cfg,
        delay_min_lag=delay_min_lag,
        delay_max_lag=delay_max_lag,
        delay_hold_prob=delay_hold_prob,
        delay_update_period=delay_update_period,
        delay_per_env_phase=delay_per_env_phase,
    )


def _update_assets(assets: dict[str, bytes], path: Path, meshdir: str = "") -> None:
    """`mjlab.utils.os.update_assets`, removed in mjlab 1.2. Reads every file under
    *path* into *assets*, keyed by the name MuJoCo will look up (`meshdir` + filename).

    The callers here point one directory above where the meshes actually sit
    (`leap_hand/assets`, while the repo keeps them in `leap_hand/xmls/assets`), so a
    missing directory falls back to the nearest `assets/` below it — which is what the
    old recursive helper effectively did for them.
    """
    path = Path(path)
    if not path.is_dir():
        found = sorted(p for p in path.parent.rglob("assets") if p.is_dir())
        if not found:
            raise FileNotFoundError(f"No asset directory at or below {path}")
        path = found[0]
    prefix = meshdir.rstrip("/") + "/" if meshdir else ""
    for entry in sorted(path.rglob("*")):
        if entry.is_file():
            assets[prefix + entry.relative_to(path).as_posix()] = entry.read_bytes()


def install() -> None:
    """Patch the two names in, then put the repo's `src/` on the path."""
    if not hasattr(_actuator, "DelayedActuatorCfg"):
        _actuator.DelayedActuatorCfg = _delayed_actuator_cfg  # type: ignore[attr-defined]
    if not hasattr(_mjlab_os, "update_assets"):
        _mjlab_os.update_assets = _update_assets  # type: ignore[attr-defined]
    # Only needed to import the repo's *task* package, whose `__init__` walks every
    # config module. The robot definition alone does not touch it. `TerrainImporterCfg`
    # became `TerrainEntityCfg` in mjlab 1.2.
    if not hasattr(_terrains, "TerrainImporterCfg"):
        _terrains.TerrainImporterCfg = _terrains.TerrainEntityCfg  # type: ignore[attr-defined]
    if str(IN_HAND_SRC) not in sys.path:
        sys.path.insert(0, str(IN_HAND_SRC))
