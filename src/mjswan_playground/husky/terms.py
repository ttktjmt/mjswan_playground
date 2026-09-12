"""The two HUSKY policy terms mjlab's own MDP functions do not cover: ``heading``, an
attribute of the task's env subclass that the tracing proxy does not forward, and
``phase``, a gait clock the env keeps in a step counter with no browser counterpart."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjswan import CommandBinding, register_command

_ROBOT = SceneEntityCfg(name="robot")

# A resampling time no episode reaches, so `CommandTerm.compute` never fires the timer
# and the clock is resampled on episode reset alone.
_NEVER = 1.0e9


def heading(env: Any, *, asset_cfg: SceneEntityCfg = _ROBOT, **_) -> torch.Tensor:
    """Robot yaw in the world frame, as one column."""
    return env.scene[asset_cfg.name].data.heading_w.unsqueeze(-1)


@dataclass(kw_only=True)
class PhaseCommandCfg(CommandTermCfg):
    """Gait-phase clock: where the robot is in HUSKY's push -> steer cycle, in [0, 1)."""

    cycle_time: float = 6.0
    """Seconds per cycle (``G1SkaterManagerBasedRlEnvCfg.cycle_time``)."""

    control_dt: float = 0.02
    """Seconds per control step, baked into the graph. Explicit because the tracing env's
    ``step_dt`` is a bare single-entity env's, not the policy's."""

    resampling_time_range: tuple[float, float] = (_NEVER, _NEVER)

    def build(self, env: Any) -> "PhaseCommand":
        return PhaseCommand(self, env)


class PhaseCommand(CommandTerm):
    """``_get_phase()`` as a command term: a step counter, divided by the cycle length.

    Counting steps rather than accumulating the phase costs a second state field, but
    ``control_dt / cycle_time`` has no exact float: a running sum drifts, and lands a
    full cycle out at the frame where it wraps. Whole steps are exact.
    """

    cfg: PhaseCommandCfg

    def __init__(self, cfg: PhaseCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, 1, device=self.device)
        self.step_count = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.phase

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Only ever fires on reset. `_update_command` runs after, so an episode's first
        # frame sees one step, as upstream's does.
        del env_ids
        self.phase = torch.zeros_like(self.phase)
        self.step_count = torch.zeros_like(self.step_count)

    def _update_command(self) -> None:
        steps_per_cycle = self.cfg.cycle_time / self.cfg.control_dt
        self.step_count = torch.remainder(self.step_count + 1.0, steps_per_cycle)
        self.phase = self.step_count / steps_per_cycle

    def _update_metrics(self) -> None:
        pass


register_command(
    "PhaseCommandCfg",
    CommandBinding(state_fields=["phase", "step_count"], command_field="phase"),
)
