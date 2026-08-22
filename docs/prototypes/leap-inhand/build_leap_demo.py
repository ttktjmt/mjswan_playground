"""Build the LEAP in-hand rotation demo with the locally-patched mjswan."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")

import onnx  # noqa: E402

import leap_inhand_task  # noqa: E402
import mjswan  # noqa: E402
from export_leap_policy import OUT as POLICY_ONNX  # noqa: E402
from export_leap_policy import export  # noqa: E402

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist-leap"


def main() -> None:
    if not POLICY_ONNX.exists():
        export()

    task_id = leap_inhand_task.register()
    print(f"registered {task_id}")

    builder = mjswan.Builder()
    project = builder.add_project(name="LEAP In-Hand Rotation")
    scene = project.add_scene_mjlab(task_id, play=True)
    scene.add_policy(
        name="In-Hand Rotation",
        policy=onnx.load(str(POLICY_ONNX)),
        # The task's own group, restated with per-term look-back: the env config keeps
        # mjlab's group-level `history_length` so the tracer can build a live env.
        observations=leap_inhand_task.mjswan_actor_group(),
        policy_joint_names=leap_inhand_task.policy_joint_names(),
        default_joint_pos=leap_inhand_task.default_joint_pos(),
        default=True,
    )

    app = builder.build(output_dir=DIST)
    print(f"built -> {DIST}")
    return app


if __name__ == "__main__":
    main()
