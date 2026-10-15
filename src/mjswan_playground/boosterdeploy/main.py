"""Booster K1 dancing and fighting with booster_deploy's BeyondMimic policies. See
``README.md``."""

from __future__ import annotations

import mjswan

from . import env, upstream

#: (scene, booster_deploy task, clip): the K1's BeyondMimic tasks, as booster_deploy's
#: `scripts/deploy.py --task` names them.
SIMULATIONS = (
    ("K1 MJ Dance", "k1_mj2", "MJ dance, segment 1"),
    ("K1 Fight", "k1_fight", "Fight"),
)


def setup_builder() -> mjswan.Builder:
    deploy, assets = upstream.resolve_roots()
    builder = mjswan.Builder()
    project = builder.add_project(name="booster_deploy", license=deploy / "LICENSE")
    for scene_name, task_name, clip_name in SIMULATIONS:
        task = upstream.task_cfg(deploy, task_name)
        env_cfg = env.tracking_env_cfg(task, assets)
        clip = upstream.clip_npz(deploy, task, env_cfg)
        motion = env_cfg.commands["motion"]
        motion.motion_file = str(clip)
        scene = project.add_scene_mjlab(scene_name, env_cfg=env_cfg)
        scene.add_attribution("booster_assets", license=assets / "LICENSE")
        handle = scene.add_policy(
            name=task_name,
            policy=upstream.policy_onnx(deploy, task),
            policy_joint_names=[f"robot/{name}" for name in task.robot.joint_names],
            default_joint_pos=list(task.robot.default_joint_pos),
        )
        handle.add_motion(
            name=clip_name,
            source=str(clip),
            fps=1.0 / task.policy_dt,
            anchor_body_name=motion.anchor_body_name,
            body_names=motion.body_names,
            default=True,
        )
    return builder
