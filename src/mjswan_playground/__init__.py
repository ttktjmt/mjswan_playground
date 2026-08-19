"""A collection of tasks built on mjswan."""

import os

# mjlab's task packages pick a mujoco GL backend at import time, so set it before any
# task module loads. mjswan setdefaults the same, but only once imported.
os.environ.setdefault("MUJOCO_GL", "disable")

from mjswan_playground.registry import ALL_TASKS, load  # noqa: E402

__all__ = ["ALL_TASKS", "load"]
