"""The muscle-supermimic checkout, its MS-Human-700 submodule, and what each policy needs.

Upstream's policy is a motion encoder over the reference alone and a decoder over the
body's state. The encoder's input depends on nothing the body does, so its latent is
evaluated here for every frame of the clips, as upstream's own browser demo does
(``export_web.py``), and the browser runs the decoder alone. Both come from the Hub
checkpoint. The clips are rebuilt from the demo's files in upstream's git, since the
dataset they were retargeted from is gated, each into an mjlab motion ``.npz`` that also
carries its frames' latents.
"""

from __future__ import annotations

import dataclasses
import gzip
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from mjswan_playground._deps import CACHE_DIR, ensure_repo

REPO_URL = "https://github.com/kaibohex/muscle-supermimic"
REPO_COMMIT = "d73bbcb866da83bef186a730547ab09ecb96bd3d"
#: Upstream's submodule, whose ssh URL a plain fetch cannot reach.
MS_HUMAN_700_URL = "https://github.com/LNSGroup/MS-Human-700"
MS_HUMAN_700_COMMIT = "2d686957aefd5739cf4d2859a6acfd94c8f84400"
TASK_PACKAGE = "muscle_supermimic.tasks"

POLICY_REPO_ID = "Beanpow/muscle-supermimic"
POLICY_REVISION = "d3fdd54eedea09411f5c9750ff59a61996febc9f"

#: The control rate, which the demo's clips are sampled at.
FPS = 100
#: The demo's clip menu groups (``main.mjs``'s ``KINDS``): the first match names a clip's.
KINDS = (
    (r"WalkingStraightForward|/walking_?\d|walking_forward", "Walk forward"),
    (r"Backwards", "Walk backward"),
    (r"walking_slow", "Walk slowly"),
    (r"walking_medium", "Walk at a medium pace"),
    (r"walking_fast", "Walk fast"),
    (r"walking_run", "Run"),
    (r"LeftTurn|turn_left", "Turn left"),
    (r"RightTurn|turn_right", "Turn right"),
    (r"CounterClockwise", "Walk a counter-clockwise circle"),
    (r"Clockwise", "Walk a clockwise circle"),
    (r"support|handrail", "Walk with a support"),
)

_CACHE = CACHE_DIR / "msm"


@dataclasses.dataclass(frozen=True)
class Body:
    key: str
    """The body's directory under upstream's ``docs/website/demo/data``."""
    task_id: str
    checkpoint: str
    """The checkpoint's path in the Hub repository."""


MS_HUMAN_700 = Body(
    key="mshuman700",
    task_id="Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands",
    checkpoint="mshuman700/kit_mm_euler_sgc100/model_016000.pt",
)
MYOFULLBODY = Body(
    key="myofullbody",
    task_id="Mjlab-SuperMimic-Flat-MyoFullBody",
    checkpoint="myofullbody/kit_n4096_euler_sgc100/model_016000.pt",
)


@dataclasses.dataclass(frozen=True)
class Clip:
    name: str
    """Upstream's, without its ``_poses`` suffix."""
    group: str
    path: Path
    """An mjlab motion ``.npz`` with the per-frame arrays :data:`FRAME_FIELDS` too."""


#: ``latent``: the encoder's latent of each frame's goal. ``frames_left``: the frames
#: after it in the clip, which upstream's ``motion_end`` counts down.
FRAME_FIELDS = ("latent", "frames_left")


def resolve_root() -> Path:
    return ensure_repo(
        name="msm/muscle-supermimic",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker="src/muscle_supermimic/__init__.py",
        root_env_var="MJSWAN_MSM_ROOT",
    )


def register_tasks(root: Path) -> None:
    """Import upstream's task package, which registers its three tasks with mjlab."""
    if TASK_PACKAGE in sys.modules:
        return
    # Upstream reads the model's location from this variable on import.
    os.environ["MS_HUMAN_700_ROOT"] = str(
        ensure_repo(
            name="msm/MS-Human-700",
            url=MS_HUMAN_700_URL,
            commit=MS_HUMAN_700_COMMIT,
            marker="MS-Human-700.xml",
            root_env_var="MS_HUMAN_700_ROOT",
        )
    )
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    importlib.import_module(TASK_PACKAGE)


def ensure_policy(body: Body, root: Path, env_cfg: object) -> tuple[Path, list[Clip]]:
    """The decoder as ONNX, and the demo's clips as motions, converted once into the
    cache."""
    demo = root / "docs" / "website" / "demo" / "data" / body.key
    meta = json.loads((demo / "meta.json").read_text())
    onnx_path = _CACHE / body.key / f"{Path(body.checkpoint).stem}.onnx"
    clips = []
    for clip in meta["clips"]:
        name = clip["name"].removesuffix("_poses")
        group = next(label for pattern, label in KINDS if re.search(pattern, name))
        path = _CACHE / body.key / "motions" / f"{name.replace('/', '_')}.npz"
        clips.append(Clip(name, group, path))
    if not (onnx_path.exists() and all(clip.path.exists() for clip in clips)):
        _convert(body, demo, meta, env_cfg, onnx_path, clips)
    return onnx_path, clips


def _frames(demo: Path, meta: dict, clip: dict) -> np.ndarray:
    """One clip's frames: per frame the latent, then the reference qpos."""
    raw = gzip.decompress((demo / clip["file"]).read_bytes())
    return np.frombuffer(raw, "<f4").reshape(clip["length"], meta["frame_width"])


def _convert(
    body: Body,
    demo: Path,
    meta: dict,
    env_cfg: object,
    onnx_path: Path,
    clips: list[Clip],
) -> None:
    import onnxruntime as ort
    import torch
    from huggingface_hub import hf_hub_download
    from muscle_supermimic.rl.supermimic.networks import Policy
    from muscle_supermimic.tasks.tracking.mdp.heads import StepCtx, TrackingHeads
    from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics
    from muscle_supermimic.tasks.tracking.mdp.motion import MotionLib
    from muscle_supermimic.tasks.tracking.scripts.export_web import (
        Decoder,
        round_weights_fp16,
        store_fp16,
    )

    command = env_cfg.commands["motion"]
    model = env_cfg.scene.entities["robot"].spec_fn().compile()
    kin = SkeletonKinematics(command.body_names, "cpu", model)

    # Each clip as upstream's MuscleMimic trajectory format at the control rate, which
    # MotionLib reads back frame for frame.
    joint_names = np.array([kin.info.root_joint_name, *kin.info.joint_names])
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for k, clip in enumerate(meta["clips"]):
            qpos = _frames(demo, meta, clip)[:, meta["nz"] :].astype(np.float64)
            files.append(str(Path(tmp) / f"{k}.npz"))
            np.savez(files[-1], qpos=qpos, joint_names=joint_names, frequency=FPS)
        motion = MotionLib(files, kin, 1.0 / FPS, "cpu")
    heads = TrackingHeads(kin, motion, None, command.lookahead, command.stride)

    checkpoint = torch.load(
        hf_hub_download(POLICY_REPO_ID, body.checkpoint, revision=POLICY_REVISION),
        map_location="cpu",
        weights_only=False,
    )
    state, cfg = checkpoint["pi"], checkpoint["cfg"]
    policy = Policy(
        state["obs_norm.mean"].numel(),
        state["goal_norm.mean"].numel(),
        model.nu,
        cfg["enc_hidden"],
        cfg["dec_hidden"],
        cfg["latent_dim"],
    )
    policy.load_state_dict(state)
    policy.eval()
    with torch.no_grad():
        goal = heads.goal(StepCtx((), torch.arange(motion.num_frames)))
        latent = torch.tanh(policy.enc(policy.goal_norm(goal))).numpy()
    shipped = np.concatenate(
        [_frames(demo, meta, c)[:, : meta["nz"]] for c in meta["clips"]]
    )
    error = float(np.abs(latent - shipped).max())
    if error > 1e-3:
        raise RuntimeError(f"The latents disagree with upstream's demo by {error:.2e}")

    # Upstream's own export, weight matrices stored as float16 as its demo ships them.
    round_weights_fp16(policy)
    nz, nobs = cfg["latent_dim"], state["obs_norm.mean"].numel()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Decoder(policy),
        (torch.zeros(1, nz), torch.zeros(1, nobs)),
        str(onnx_path),
        input_names=["z", "obs"],
        output_names=["excitation"],
        dynamo=False,
    )
    store_fp16(str(onnx_path))
    # On observations drawn from the checkpoint's own running statistics; both graphs
    # take one row at a time.
    mean, var = state["obs_norm.mean"].numpy(), state["obs_norm.var"].numpy()
    rng = np.random.default_rng(0)
    ours = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    theirs = ort.InferenceSession(
        gzip.decompress((demo / meta["files"]["policy"]).read_bytes()),
        providers=["CPUExecutionProvider"],
    )
    error = 0.0
    for row in np.linspace(0, len(latent) - 1, 16).astype(int):
        sample = {
            "z": latent[row : row + 1],
            "obs": rng.normal(mean, np.sqrt(var) + 1e-3, (1, nobs)).astype(np.float32),
        }
        error = max(
            error,
            float(
                np.abs(ours.run(None, sample)[0] - theirs.run(None, sample)[0]).max()
            ),
        )
    if error > 1e-5:
        onnx_path.unlink()
        raise RuntimeError(f"The decoder disagrees with upstream's demo by {error:.2e}")

    init = np.fromfile(demo / meta["files"]["init"], "<f8")
    init = init.reshape(len(clips), meta["nq"] + meta["nv"])
    # Their rows in the motion, which skips the world body.
    tracked = np.array([model.body(name).id - 1 for name in command.body_names])
    for k, clip in enumerate(clips):
        start, n = int(motion.clip_start[k]), int(motion.clip_len[k])
        frames = slice(start, start + n)
        motion_npz = _replay(model, kin, motion, frames, init[k])
        error = np.abs(
            motion_npz["body_pos_w"][:, tracked] - motion.body_p[frames].numpy()
        ).max()
        if error > 1e-4:
            raise RuntimeError(
                f"{clip.name}: MuJoCo puts the tracked bodies {error:.2e} m from "
                "upstream's kinematics"
            )
        clip.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            clip.path,
            **motion_npz,
            latent=latent[frames].astype(np.float32),
            frames_left=np.arange(n - 1, -1, -1, dtype=np.float32)[:, None],
        )


def _replay(
    model: object,
    kin: object,
    motion: object,
    frames: slice,
    init: np.ndarray,
) -> dict[str, np.ndarray]:
    """One clip as ``mjlab.scripts.csv_to_npz`` writes it: each frame's state set and
    every body of the entity recorded. The state is upstream's reference, velocities
    included, and at the first frame the state upstream's reset writes (``init.bin``)."""
    import mujoco
    import torch
    from muscle_supermimic.utils.rotations import mat_to_quat

    root_R = motion.root_R[frames] @ kin.R_off.T
    q, qd = kin.full_joint_state(motion.q[frames], motion.qd[frames])
    # The free joint's qvel holds the angular velocity in the root's own frame.
    w = (root_R.transpose(-1, -2) @ motion.root_w[frames].unsqueeze(-1)).squeeze(-1)
    qpos = torch.cat([motion.root_p[frames], mat_to_quat(root_R), q], -1).numpy()
    qvel = torch.cat([motion.root_v[frames], w, qd], -1).numpy()
    if np.abs(np.concatenate([qpos[0], qvel[0]]) - init).max() > 1e-4:
        raise RuntimeError("The first frame is not the state upstream's reset writes")
    qpos[0], qvel[0] = init[: model.nq], init[model.nq :]

    data = mujoco.MjData(model)
    bodies = slice(1, model.nbody)
    out = {name: [] for name in ("pos", "quat", "lin", "ang")}
    for qpos_t, qvel_t in zip(qpos, qvel):
        data.qpos[:], data.qvel[:] = qpos_t, qvel_t
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        # cvel is about the subtree com of each body's root; mjlab's body_link_vel_w.
        pos, cvel = data.xpos[bodies], data.cvel[bodies]
        com = data.subtree_com[model.body_rootid[bodies]]
        out["pos"].append(pos.copy())
        out["quat"].append(data.xquat[bodies].copy())
        out["lin"].append(cvel[:, 3:] + np.cross(cvel[:, :3], pos - com))
        out["ang"].append(cvel[:, :3].copy())
    ang = np.stack(out["ang"])
    # ponytail: mjswan's TrackingCommand writes the root's body_ang_vel_w into qvel
    # unrotated, where mjlab rotates it into the root's frame, so the first frame holds it
    # in that frame and `terms.MotionCommand` writes it as mjswan does. Drop both once
    # mjswan rotates it.
    ang[0, kin.info.root_id - 1] = qvel[0, 3:6]
    return {
        "fps": np.array([FPS]),
        "joint_pos": qpos[:, 7:].astype(np.float32),
        "joint_vel": qvel[:, 6:].astype(np.float32),
        "body_pos_w": np.stack(out["pos"]).astype(np.float32),
        "body_quat_w": np.stack(out["quat"]).astype(np.float32),
        "body_lin_vel_w": np.stack(out["lin"]).astype(np.float32),
        "body_ang_vel_w": ang.astype(np.float32),
        "body_names": np.array([model.body(b).name for b in range(1, model.nbody)]),
    }
