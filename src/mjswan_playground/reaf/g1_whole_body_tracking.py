"""`G1WholeBodyTracking`: the Unitree G1 tracking a LAFAN1 dance with its whole body,
BeyondMimic-style."""

from __future__ import annotations

from pathlib import Path

import mjswan
import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg

from mjswan_playground._deps import CACHE_DIR

from ._common import MENAGERIE, fetch

#: The policy `train.py` trains on `G1WholeBodyTracking` with `--motion dance1_subject2`,
#: as its repository and revision; None until it is on the Hub.
POLICY: tuple[str, str] | None = None
#: LAFAN1 retargeted to the G1, at the revision `train.py` takes its clips from.
LAFAN1 = (
    "lvhaidong/LAFAN1_Retargeting_Dataset",
    "ce1572906efe6157840e8474d5a0d7aa87481e74",
)
CLIP = "dance1_subject2"
#: The clip is CC BY-NC-ND 4.0, not the Apache-2.0 of the rest.
LAFAN1_LICENSE = Path(__file__).parent / "LICENSE.lafan1"
FPS = 50.0


def motion_file(clip: str = CLIP) -> Path:
    """The clip as Playground's `csv_to_npz` converts it, once: interpolated from 30 to
    50 fps, with every body's pose and velocity in the world, on mjlab's G1, which is
    Playground's (Menagerie's `g1_mjlab.xml`)."""
    target = CACHE_DIR / "reaf" / "lafan1" / LAFAN1[1] / f"{clip}.npz"
    if target.exists():
        return target
    from huggingface_hub import hf_hub_download
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation, Slerp

    repo, revision = LAFAN1
    csv = hf_hub_download(
        repo, f"g1/{clip}.csv", repo_type="dataset", revision=revision
    )
    motion = np.loadtxt(csv, delimiter=",")
    t_in = np.linspace(0, (len(motion) - 1) / 30.0, len(motion))
    t_out = np.arange(0, t_in[-1], 1.0 / FPS)
    pos = interp1d(t_in, motion[:, :3], axis=0)(t_out)
    joints = interp1d(t_in, motion[:, 7:], axis=0)(t_out)
    xyzw = Slerp(t_in, Rotation.from_quat(motion[:, 3:7]))(t_out).as_quat()
    linvel = np.gradient(pos, 1.0 / FPS, axis=0)
    jvel = np.gradient(joints, 1.0 / FPS, axis=0)
    turns = Rotation.from_quat(xyzw[1:]) * Rotation.from_quat(xyzw[:-1]).inv()
    angvel = turns.as_rotvec() * FPS
    angvel = np.concatenate([angvel[:1], angvel])

    model = get_spec().compile()
    data = mujoco.MjData(model)
    log: dict[str, list] = {
        key: []
        for key in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
    }
    for i in range(len(t_out)):
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = (
            pos[i],
            xyzw[i, [3, 0, 1, 2]],
            joints[i],
        )
        data.qvel[:3], data.qvel[3:6], data.qvel[6:] = linvel[i], angvel[i], jvel[i]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        log["joint_pos"].append(data.qpos[7:].copy())
        log["joint_vel"].append(data.qvel[6:].copy())
        # Every body but the world; cvel moved from the subtree's centre of mass to
        # the body's origin.
        log["body_pos_w"].append(data.xpos[1:].copy())
        log["body_quat_w"].append(data.xquat[1:].copy())
        ang = data.cvel[1:, :3]
        offset = data.subtree_com[model.body_rootid[1:]] - data.xpos[1:]
        log["body_lin_vel_w"].append(data.cvel[1:, 3:] - np.cross(ang, offset))
        log["body_ang_vel_w"].append(ang.copy())
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part.npz")
    np.savez(partial, fps=np.array([FPS]), **{k: np.stack(v) for k, v in log.items()})
    partial.replace(target)
    return target


def env_cfg() -> ManagerBasedRlEnvCfg:
    """mjlab's BeyondMimic G1 tracking, which Playground restates: the same G1, servos,
    rest pose, observations, action scale and terminations, at 50 Hz. Played as
    Playground plays it: from the clip's first frame, without randomization."""
    cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=True)
    cfg.events = {}
    motion = cfg.commands["motion"]
    motion.motion_file = str(motion_file())
    motion.joint_position_range = (0.0, 0.0)
    return cfg


def add_scenes(project: mjswan.ProjectHandle) -> None:
    cfg = env_cfg()
    scene = project.add_scene_mjlab(
        "G1WholeBodyTracking", name="G1WholeBodyTracking", env_cfg=cfg
    )
    scene.add_attribution("unitree_g1", license=fetch(MENAGERIE, "unitree_g1/LICENSE"))
    scene.add_attribution("lafan1", license=LAFAN1_LICENSE)
    if POLICY is None:
        return
    repo, revision = POLICY
    policy = scene.add_policy_hf(repo, revision=revision, name="Dance")
    motion = cfg.commands["motion"]
    policy.add_motion(
        name=CLIP,
        source=motion.motion_file,
        fps=FPS,
        anchor_body_name=motion.anchor_body_name,
        body_names=motion.body_names,
        default=True,
    )
