"""Build-time shims for mjswan's ONNX tracer, shared by the tasks that need them."""

from __future__ import annotations

import torch


class CommandValues:
    """Width-only stand-in for a command term: ``generated_commands`` traces against an
    env with no command manager, and the runtime serves the live command browser-side.

    Pass one per command a traced term reads to
    ``build_single_entity_trace_env(commands=...)``.
    """

    def __init__(self, width: int) -> None:
        self.command = torch.zeros(1, width)
