"""The public MuscleMimic checkpoint, and the contract it was trained against.

``amathislab/mm-10m-2`` (2.05e9 steps) read upstream ``MjxMyoFullBody``'s 2418-wide
observation, not the 1152 values myosuite's mjlab task feeds. myosuite ships that layout as
:class:`FullbodyObsAdapter`, in numpy over ``mujoco.MjData``, and mjswan can only trace
torch. :func:`build_terms` reproduces it in torch from raw sim fields the browser serves,
taking every index and the whole clip-derived half from the numpy adapter so the two
cannot drift.

Layout, in order (``FullbodyObsAdapter.build``):

==================================================  =====
root qpos (z + quat) / non-root qpos                  87
root qvel / non-root qvel                             88
per actuator: length, velocity, force, ctrl, act    1770
touch sensor sums (r_foot, r_toes, l_foot, l_toes)     4
current relative site pos / angles / vel             192
clip lookahead (5 steps, stride 20) + phase          277
==================================================  =====
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from mjlab.utils.lab_api.math import quat_from_matrix

from mjswan_playground._deps import CACHE_DIR

POLICY_REPO_ID = "amathislab/mm-10m-2"
#: Gated (auto-approved) dataset: ``hf auth login`` once.
CLIP_REPO_ID = "amathislab/musclemimic-retargeted"
CLIP_FILENAME = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"

ENTITY = "mimic_fullbody_robot"
_CACHE = CACHE_DIR / "musclemimic"


def fetch_clip() -> Path:
    return Path(hf_hub_download(CLIP_REPO_ID, CLIP_FILENAME, repo_type="dataset"))


def ensure_policy() -> tuple[Path, dict[str, Any]]:
    """The actor as float16 ONNX, plus the observation params it was trained with.

    Converted once into the cache: the Orbax train state is read, the actor rebuilt with
    myosuite's own ``MimicActorModule``, exported, and checked against myosuite's numpy
    forward pass. float16 weights (float32 in/out) halve the 38 MB file to fit hosts that
    cap a file at 25 MiB; they move the actions by at most 0.018 and no rollout.
    """
    onnx_path = _CACHE / "policy_mm10m2.onnx"
    params_path = _CACHE / "policy_mm10m2_env.json"
    if not (onnx_path.exists() and params_path.exists()):
        _convert(onnx_path, params_path)
    return onnx_path, json.loads(params_path.read_text())


def _convert(onnx_path: Path, params_path: Path) -> None:
    import onnx
    import onnxruntime as ort
    from myosuite.integrations.musclemimic.actor_onnx import export_to_onnx
    from myosuite.integrations.musclemimic.actor_torch import MimicActorModule
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        _actor_forward,
        fullbody_obs_adapter_params_from_metadata,
        load_local_policy_artifacts,
        read_checkpoint_config_metadata,
    )
    from onnxconverter_common import float16

    root = Path(snapshot_download(POLICY_REPO_ID))
    artifacts = load_local_policy_artifacts(root)
    module = MimicActorModule.from_artifacts(artifacts)

    fp32 = export_to_onnx(module, _CACHE / "policy_mm10m2.fp32.onnx")
    # keep_io_types: the browser feeds and reads float32; only the weights shrink.
    onnx.save(
        float16.convert_float_to_float16(onnx.load(fp32), keep_io_types=True), onnx_path
    )

    # The export is only trusted if it matches myosuite's forward pass on observations
    # drawn from the checkpoint's own running statistics.
    sample = (
        np.random.default_rng(0)
        .normal(
            artifacts.obs_mean,
            np.sqrt(artifacts.obs_var) + 1e-3,
            (16, artifacts.obs_dim),
        )
        .astype(np.float32)
    )
    reference = _actor_forward(
        artifacts.params,
        module.normalize(torch.from_numpy(sample)).detach().numpy(),
    )
    exported = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).run(
        ["actions"], {"obs": sample}
    )[0]
    fp32.unlink()
    error = float(np.abs(exported - reference).max())
    if error > 1e-4:
        onnx_path.unlink()
        raise RuntimeError(f"Exported actor disagrees with myosuite's by {error:.2e}")

    params_path.write_text(
        json.dumps(
            fullbody_obs_adapter_params_from_metadata(
                read_checkpoint_config_metadata(root)
            ),
            indent=1,
        )
    )


# --- The observation and termination, in torch ------------------------------------


@contextlib.contextmanager
def _resolve_entity_names(entity: str = ENTITY):
    """Let name lookups fall back to mjlab's ``entity/name`` form: the adapter asks for
    bare names (``root``, ``r_foot``), and mjlab prefixes every attached element."""
    original = mujoco.mj_name2id

    def prefixed(model: Any, obj_type: Any, name: str) -> int:
        found = original(model, obj_type, name)
        return found if found >= 0 else original(model, obj_type, f"{entity}/{name}")

    mujoco.mj_name2id = prefixed
    try:
        yield
    finally:
        mujoco.mj_name2id = original


def _mat_to_rotvec(mat: torch.Tensor) -> torch.Tensor:
    """Rotation matrices to rotation vectors, as ``scipy``'s ``as_rotvec`` does it.

    mjlab's ``quat_from_matrix`` is branchless, so the graph has no data-dependent
    control flow. Its ``axis_angle_from_quat`` is not usable here: it flips the sign by
    multiplying with a bool, which traces to an ONNX ``Mul`` on a bool that onnxruntime
    rejects.
    """
    quat = quat_from_matrix(mat)
    # w >= 0 puts the angle in [0, pi], as `as_rotvec` requires.
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
    w, vec = quat[..., 0], quat[..., 1:]
    angle = 2.0 * torch.atan2(torch.linalg.vector_norm(vec, dim=-1), w)
    squared = angle * angle
    small = 2.0 + squared / 12.0 + 7.0 * squared * squared / 2880.0
    large = angle / torch.sin(angle / 2.0).clamp_min(1e-12)
    return vec * torch.where(angle <= 1e-3, small, large).unsqueeze(-1)


def _relative_site_quantities(
    pos: torch.Tensor,
    mat: torch.Tensor,
    body_cvel: torch.Tensor,
    root_com: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mimic-site positions, angles and velocities relative to the first (pelvis) site.

    Every argument is already narrowed to the mimic sites, ``(B, n_sites, ...)``.
    ``body_cvel`` is the parent body's spatial velocity (angular ++ linear) and
    ``root_com`` its root subtree centre of mass.
    """
    angular = body_cvel[..., :3]
    linear = body_cvel[..., 3:] - torch.cross(pos - root_com, angular, dim=-1)

    main_mat = mat[:, 0]
    site_rpos = pos[:, 1:] - pos[:, :1]
    rel_rot = torch.matmul(main_mat.transpose(-1, -2).unsqueeze(1), mat[:, 1:])
    rel_lin = torch.matmul(linear[:, :1] - linear[:, 1:], main_mat.transpose(-1, -2))
    rel_ang = (
        torch.matmul(rel_rot.transpose(-1, -2), angular[:, 1:].unsqueeze(-1)).squeeze(
            -1
        )
        - angular[:, :1]
    )
    return site_rpos, _mat_to_rotvec(rel_rot), torch.cat([rel_ang, rel_lin], dim=-1)


@dataclasses.dataclass
class Terms:
    observation: Callable[[Any], torch.Tensor]
    """``func(env) -> (B, 2418)``."""
    termination: Callable[[Any], torch.Tensor]
    """``func(env) -> (B,)`` bool: upstream's mean-site-deviation-with-root handler."""
    adapter: Any
    """myosuite's numpy builder, kept so a check can compare against it."""


def build_terms(
    model: mujoco.MjModel,
    clip: Any,
    params: dict[str, Any],
    ctrl_dt: float,
    site_threshold: float = 1.0,
    root_threshold: float = 1.0,
) -> Terms:
    """The observation and termination the checkpoint needs, as traceable torch terms.

    ``params`` are the checkpoint's own ``enable_*`` flags and goal params, from
    :func:`ensure_policy`. The thresholds are upstream's; myosuite's own
    ``mimic_deviation`` uses 0.3 m against the *mean of the target sites*, which is
    already 0.15 m off for a perfect tracker.
    """
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        FullbodyObsAdapter,
    )

    with _resolve_entity_names():
        adapter = FullbodyObsAdapter(model, clip, params)

    n_frames = int(adapter._traj_len)
    # The clip-derived half, lookahead (276) ++ phase (1), reads no live state, so the
    # numpy adapter computes it per frame and a lookup reproduces it exactly.
    goal_table = torch.as_tensor(
        np.asarray(
            [
                np.concatenate([adapter._traj_goal_obs(f), [f / max(n_frames, 1)]])
                for f in range(n_frames)
            ],
            dtype=np.float32,
        )
    )
    ref_root = torch.as_tensor(np.asarray(clip.qpos[:n_frames, :3], dtype=np.float32))

    site = np.asarray(adapter._site_ids)
    parent_body = np.asarray(adapter._sim_site_bodyid)[site]
    ix = {
        k: torch.as_tensor(np.asarray(v, dtype=np.int64))
        for k, v in {
            "qpos_root": adapter._root_qpos_idx_full[2:],
            "qpos_rest": adapter._qpos_non_root_ind,
            "qvel_root": adapter._root_qvel_idx_full,
            "qvel_rest": adapter._qvel_non_root_ind,
            "site": site,
            "parent_body": parent_body,
            "root_body": np.asarray(adapter._sim_body_rootid)[parent_body],
        }.items()
    }
    touch = [
        (int(model.sensor_adr[s]), int(model.sensor_dim[s]))
        for s in adapter._touch_sensor_ids
    ]

    def sites(data: Any):
        # Index the raw field: that is what tells the tracer to ship 17 rows, not 2038.
        return _relative_site_quantities(
            data.site_xpos[:, ix["site"]],
            data.site_xmat[:, ix["site"]],
            data.cvel[:, ix["parent_body"]],
            data.subtree_com[:, ix["root_body"]],
        )

    def frame_of(data: Any) -> torch.Tensor:
        return (data.time / ctrl_dt).long() % n_frames

    def observation(env: Any) -> torch.Tensor:
        data = env.sim.data
        qpos, qvel = data.qpos, data.qvel
        parts = [
            qpos[:, ix["qpos_root"]],
            qpos[:, ix["qpos_rest"]],
            qvel[:, ix["qvel_root"]],
            qvel[:, ix["qvel_rest"]],
            # Per actuator, in actuator order: length, velocity, force, ctrl, act.
            # `[:]` unwraps the `TorchArray` a live env serves; `stack` won't in a list.
            torch.stack(
                [
                    data.actuator_length[:],
                    data.actuator_velocity[:],
                    data.actuator_force[:],
                    data.ctrl[:],
                    data.act[:],
                ],
                dim=-1,
            ).flatten(1),
        ]
        if touch:
            sensordata = data.sensordata
            parts.append(
                torch.stack(
                    [sensordata[:, a : a + d].sum(-1) for a, d in touch], dim=-1
                )
            )
        parts += [q.flatten(1) for q in sites(data)]
        parts.append(goal_table[frame_of(data)])
        return torch.cat(parts, dim=-1)

    def termination(env: Any) -> torch.Tensor:
        data = env.sim.data
        frame = frame_of(data)
        site_rpos, _, _ = sites(data)
        # The goal row opens with the reference site_rpos at this frame.
        reference = goal_table[frame][:, : site_rpos.shape[1] * 3]
        deviation = torch.linalg.vector_norm(
            site_rpos - reference.unflatten(-1, (-1, 3)), dim=-1
        ).mean(-1)
        root = torch.linalg.vector_norm(data.qpos[:, :3] - ref_root[frame], dim=-1)
        return (deviation > site_threshold) | (root > root_threshold)

    return Terms(observation=observation, termination=termination, adapter=adapter)
