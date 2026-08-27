"""The two terms mjlab's own MDP functions do not cover: ``zeros``, the command slots an
env pads rather than drives, and ``GroundPickPhaseCommand``, the cyclic clock the pick and
crouch policies read where the others read a twist."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjswan import CommandBinding, register_command

#: A resampling time no episode reaches, so ``CommandTerm.compute``'s timer never fires:
#: the clock is continuous and owns its own wrap.
_NEVER = 1.0e9


def zeros(env: Any, *, dim: int, **_) -> torch.Tensor:
    """``dim`` columns of zeros — upstream's ``zero_command_padding``.

    Every policy reads the same 13-wide command block, so an env that does not drive a
    slot pads it rather than dropping it; that is what keeps one runtime able to swap any
    policy in.
    """
    return torch.zeros(env.num_envs, dim, device=env.device)


@dataclass(kw_only=True)
class GroundPickPhaseCommandCfg(CommandTermCfg):
    """The twist slot as a cyclic clock: ``[cos(2*pi*phase), sin(2*pi*phase), 0]``.

    ``GroundPickPhaseCommand`` in upstream's ``tasks/mdp.py``. Phase in [0, 0.5) takes the
    mouth down, [0.5, 1) brings it back up.
    """

    period: float = 4.0
    """Seconds per cycle (``GroundPickPhaseCommand.PERIOD``; the crouch policy's env
    keeps the same default)."""

    control_dt: float = 0.02
    """Seconds per control step, baked into the graph. Explicit because the tracing env's
    ``step_dt`` is a bare single-entity env's, not the policy's."""

    resampling_time_range: tuple[float, float] = (_NEVER, _NEVER)

    def build(self, env: Any) -> "GroundPickPhaseCommand":
        return GroundPickPhaseCommand(self, env)


class GroundPickPhaseCommand(CommandTerm):
    """Upstream advances ``phase`` by ``dt / period`` and wraps at 1; this counts whole
    steps instead and divides.

    ``control_dt / period`` has no exact float — a running sum drifts and lands a full
    cycle out at the frame where it wraps, which for this command means the mouth diving
    on the way up. Whole steps are exact.
    """

    cfg: GroundPickPhaseCommandCfg

    def __init__(self, cfg: GroundPickPhaseCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.step_count = torch.zeros(self.num_envs, 1, device=self.device)
        self.phase_cmd = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.phase_cmd

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Only ever fires on reset, and the runtime's button starts the cycle from
        # standing, which is phase 0 — upstream's `randomize_phase=False` branch.
        del env_ids
        self.step_count = torch.zeros_like(self.step_count)
        self.phase_cmd = torch.zeros_like(self.phase_cmd)

    def _update_command(self) -> None:
        steps_per_cycle = self.cfg.period / self.cfg.control_dt
        self.step_count = torch.remainder(self.step_count + 1.0, steps_per_cycle)
        phase = self.step_count / steps_per_cycle
        self.phase_cmd = torch.cat(
            [
                torch.cos(2.0 * math.pi * phase),
                torch.sin(2.0 * math.pi * phase),
                torch.zeros_like(phase),
            ],
            dim=-1,
        )

    def _update_metrics(self) -> None:
        pass


register_command(
    "GroundPickPhaseCommandCfg",
    CommandBinding(state_fields=["step_count", "phase_cmd"], command_field="phase_cmd"),
)


class CommandValues:
    """Width-only stand-in for a command term: ``generated_commands`` traces against an
    env with no command manager, and the runtime serves the live command browser-side."""

    def __init__(self, width: int) -> None:
        self.command = torch.zeros(1, width)
