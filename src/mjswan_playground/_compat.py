"""Shims for upstream tasks written against mjlab before 1.6."""

from __future__ import annotations

import dataclasses
import posixpath
from pathlib import Path
from unittest import mock

from mjlab.actuator import ActuatorCfg, CommandField, XmlActuatorCfg
from mjlab.utils.spec_config import CollisionCfg

_COLLISION_DEFAULTS = {"contype": 1, "conaffinity": 1, "condim": 3, "priority": 0}


def collision_defaults():
    """Patch ``CollisionCfg`` to default the fields mjlab 1.6 made required."""
    return mock.patch("mjlab.utils.spec_config.CollisionCfg", _collision_cfg)


def legacy_update_assets():
    """Restore mjlab 1.2's ``update_assets``, which 1.6 removed."""
    return mock.patch("mjlab.utils.os.update_assets", _update_assets, create=True)


def legacy_xml_actuators():
    """Restore mjlab 1.2's per-type XML actuators and ``DelayedActuatorCfg``, which 1.6
    folded into ``XmlActuatorCfg`` and the delay fields every actuator has."""
    return mock.patch.multiple(
        "mjlab.actuator",
        create=True,
        XmlPositionActuatorCfg=_XmlPositionActuatorCfg,
        XmlMotorActuatorCfg=_XmlMotorActuatorCfg,
        DelayedActuatorCfg=_delayed_actuator_cfg,
    )


@dataclasses.dataclass(kw_only=True)
class _XmlPositionActuatorCfg(XmlActuatorCfg):
    command_field: CommandField | None = "position"


@dataclasses.dataclass(kw_only=True)
class _XmlMotorActuatorCfg(XmlActuatorCfg):
    command_field: CommandField | None = "effort"


def _delayed_actuator_cfg(
    *, base_cfg: ActuatorCfg, delay_target: str, delay_min_lag: int, delay_max_lag: int
) -> ActuatorCfg:
    del delay_target  # 1.6 delays whatever the actuator commands.
    return dataclasses.replace(
        base_cfg, delay_min_lag=delay_min_lag, delay_max_lag=delay_max_lag
    )


def _update_assets(assets: dict[str, bytes], path: Path, meshdir: str | None) -> None:
    """Every file in ``path``, keyed under ``meshdir``."""
    for file in Path(path).iterdir():
        if file.is_file():
            key = file.name if meshdir is None else posixpath.join(meshdir, file.name)
            assets[key] = file.read_bytes()


def _collision_cfg(**kwargs) -> CollisionCfg:
    """A pattern dict without a catch-all gets the default as a last ``.*``: patterns
    match first-wins."""
    for name, default in _COLLISION_DEFAULTS.items():
        value = kwargs.setdefault(name, default)
        if isinstance(value, dict):
            kwargs[name] = {**value, ".*": value.get(".*", default)}
    return CollisionCfg(**kwargs)


def dropping_env_ids(method):
    """1.6 passes the reset env ids; before it, a term acted on every env, so drop them.
    Not ``functools.wraps``: mjlab checks the signature, which would follow it."""
    kept = method.__code__.co_argcount

    def wrapper(*args, env_ids=None, **kwargs):
        return method(*args[:kept], **kwargs)

    return wrapper
