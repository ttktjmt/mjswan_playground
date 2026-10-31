"""The muscle-supermimic checkout, its MS-Human-700 submodule, and what each policy needs.

Upstream's policy is a motion encoder over the reference alone and a decoder over the
body's state. The encoder's input depends on nothing the body does, so its latent is
evaluated here for every frame of the clips, as upstream's own browser demo does
(``export_web.py``), and the browser runs the decoder alone. Both come from the Hub
checkpoint; the clips are rebuilt from the demo's files in upstream's git, since the
dataset they were retargeted from is gated.
"""

from __future__ import annotations

import dataclasses
import gzip
import importlib
import json
import os
import sys
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


@dataclasses.dataclass
class Tables:
    """Per frame of every clip, back to back, and per clip, as the port reads them."""

    clip_start: np.ndarray
    clip_len: np.ndarray
    latent: np.ndarray
    """``(frames, 32)``: the encoder's latent of each frame's goal."""
    init: np.ndarray
    """``(clips, nq + nv)``: the state upstream's reset writes at each clip's first frame."""
    root_pos: np.ndarray
    root_rot: np.ndarray
    """``(frames, 3, 3)``: the canonical root frame (x forward, z up) of the reference."""
    body_rpos: np.ndarray
    """``(frames, tracked - 1, 3)``: the tracked bodies relative to the first one."""


def ensure_policy(body: Body, root: Path, env_cfg: object) -> tuple[Path, Tables]:
    """The decoder as ONNX, and the clip tables, converted once into the cache."""
    stem = Path(body.checkpoint).stem
    onnx_path = _CACHE / body.key / f"{stem}.onnx"
    tables_path = _CACHE / body.key / f"{stem}_tables.npz"
    if not (onnx_path.exists() and tables_path.exists()):
        _convert(body, root, env_cfg, onnx_path, tables_path)
    with np.load(tables_path) as data:
        fields = {f.name: data[f.name] for f in dataclasses.fields(Tables)}
    return onnx_path, Tables(**fields)


def _frames(demo: Path, meta: dict, clip: dict) -> np.ndarray:
    """One clip's frames: per frame the latent, then the reference qpos."""
    raw = gzip.decompress((demo / clip["file"]).read_bytes())
    return np.frombuffer(raw, "<f4").reshape(clip["length"], meta["frame_width"])


def _convert(
    body: Body, root: Path, env_cfg: object, onnx_path: Path, tables_path: Path
) -> None:
    import onnxruntime as ort
    import torch
    from huggingface_hub import hf_hub_download
    from muscle_supermimic.assets.robots.model_info import ModelInfo
    from muscle_supermimic.rl.supermimic.networks import Policy
    from muscle_supermimic.tasks.tracking.mdp.heads import StepCtx, TrackingHeads
    from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics
    from muscle_supermimic.tasks.tracking.mdp.motion import MotionLib
    from muscle_supermimic.tasks.tracking.scripts.export_web import (
        Decoder,
        round_weights_fp16,
        store_fp16,
    )

    demo = root / "docs" / "website" / "demo" / "data" / body.key
    meta = json.loads((demo / "meta.json").read_text())
    command = env_cfg.commands["motion"]
    model = env_cfg.scene.entities["robot"].spec_fn().compile()
    info = ModelInfo(model)

    # Each clip as upstream's MuscleMimic trajectory format at the control rate, which
    # MotionLib reads back frame for frame.
    clip_dir = onnx_path.parent / "clips"
    joint_names = np.array([info.root_joint_name, *info.joint_names])
    files = []
    for clip in meta["clips"]:
        path = clip_dir / f"{clip['name']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        qpos = _frames(demo, meta, clip)[:, meta["nz"] :].astype(np.float64)
        np.savez(path, qpos=qpos, joint_names=joint_names, frequency=100.0)
        files.append(str(path))
    kin = SkeletonKinematics(command.body_names, "cpu", model)
    motion = MotionLib(files, kin, 0.01, "cpu")
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
    np.savez(
        tables_path,
        clip_start=motion.clip_start.numpy(),
        clip_len=motion.clip_len.numpy(),
        latent=latent.astype(np.float32),
        init=init.reshape(len(meta["clips"]), meta["nq"] + meta["nv"]),
        root_pos=motion.root_p.numpy(),
        root_rot=motion.root_R.numpy(),
        body_rpos=heads.ref_rpos.numpy(),
    )
