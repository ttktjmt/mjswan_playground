"""Check a task's traced graphs against mjlab, on the env its ``main.py`` builds.

    MUJOCO_GL=disable uv run --with onnxruntime python scripts/parity.py <task-id>

Runs ``setup_builder()`` with ``add_scene_mjlab`` stubbed to collect the ``env_cfg`` of
every scene, imports the task's ``terms.py`` if it has one, then for each scene runs
``run_parity`` on the actor's observation group and ``run_command_parity`` on every
traced command. A tracking task
whose env config leaves the clip unset takes one with ``--motion-file``, since the env
loads it on construction. Exits 1 when anything fails.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from unittest import mock

import mjswan.project
from mjswan.compile import run_command_parity, run_parity
from mjswan.mjlab import resolve_runner_defaults
from mjswan.mjlab.command import _adapt_command_cfg
from mjswan.mjlab.env import build_mjlab_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task_id")
    parser.add_argument("--motion-file", type=Path, help="clip for a tracking task")
    args = parser.parse_args()

    scenes = []

    def collect(self, task_id, *args, env_cfg, **kwargs):
        scenes.append((task_id, env_cfg))
        return mock.MagicMock()

    mjswan.project.ProjectHandle.add_scene_mjlab = collect
    importlib.import_module(f"mjswan_playground.{args.task_id}.main").setup_builder()
    try:
        importlib.import_module(f"mjswan_playground.{args.task_id}.terms")
    except ModuleNotFoundError:
        pass
    passed = bool(scenes)
    for task_id, cfg in scenes:
        print(f"== {task_id}")
        passed = _check(task_id, cfg, args.motion_file) and passed
    return 0 if passed else 1


def _check(task_id: str, cfg, motion_file: Path | None) -> bool:
    if motion_file:
        for term in cfg.commands.values():
            if hasattr(term, "motion_file"):
                term.motion_file = str(motion_file)
    cfg.scene.num_envs = 1

    groups = (resolve_runner_defaults(task_id).obs_groups or {}).get("actor")
    env = build_mjlab_env(cfg)
    report = run_parity(env, obs_group=groups[0] if groups else "actor", n_steps=16)
    print(report.summary())
    passed = report.passed
    for name, term_cfg in cfg.commands.items():
        pending = _adapt_command_cfg(term_cfg).pending_trace
        if pending is None:
            continue
        term = pending.mjlab_cfg.build(env)
        if pending.trace_override is not None:
            pending.trace_override(term)
        result = run_command_parity(
            term, pending.state_fields, name=name, command_field=pending.command_field
        )
        print(f"command {name}: {'OK' if result.passed else 'FAIL'} {result}")
        passed = passed and result.passed
    return passed


if __name__ == "__main__":
    raise SystemExit(main())
