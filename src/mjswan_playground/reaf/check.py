"""Check a scene against Playground's own numbers: what `train.py export` wrote.

    uv run --extra reaf python -m mjswan_playground.reaf.check CartpoleBalance EXPORT_DIR

EXPORT_DIR holds `contract.json`, `reference.npz` and `policy.onnx`. Each step of the
reference rollout, Playground's state goes into the scene's mjlab env; the scene's
observations must then be Playground's, the policy's actions on them Playground's, and
the controls those actions give Playground's next ones. A camera's own frames are only
reported, since each renderer draws its own: Playground's frame goes into the camera
before the observations are taken, so they check what the scene makes of the pixels.

Playground reads bodies and sites from the kinematics `mj_step` computes before its last
physics substep moves the joints, so after the first step of an episode the check takes
them one substep back too; the scene itself reads them where the joints are.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np

#: Largest differences that pass: float32 against float32, through different code.
OBS_TOLERANCE = 1e-4
ACTION_TOLERANCE = 1e-3


def _env_cfg(env_id: str) -> tuple[object, dict]:
    """The scene's env config, and what its policy is added with: in_keys, and for a
    policy the scene builds itself, the model and out_keys."""
    import mjswan.project

    from . import main

    scenes = {}

    def collect(self, task_id, *args, env_cfg, **kwargs):
        scenes[task_id] = (env_cfg, handle := mock.MagicMock())
        return handle

    with mock.patch.object(mjswan.project.ProjectHandle, "add_scene_mjlab", collect):
        main.setup_builder()
    if env_id not in scenes:
        raise SystemExit(f"No scene for {env_id}: {sorted(scenes)}")
    env_cfg, handle = scenes[env_id]
    calls = handle.add_policy_hf.call_args_list + handle.add_policy.call_args_list
    return env_cfg, calls[0].kwargs if calls else {}


def _widths(joint_type: int) -> tuple[int, int]:
    """qpos and qvel widths of a joint type: free, ball, slide, hinge."""
    return {0: (7, 6), 1: (4, 3)}.get(int(joint_type), (1, 1))


def _match(name: str, names: list[str]) -> int:
    """The index of the one name in `names` that is `name` or ends with it, after a `/`
    or a `_`: a port may flatten Playground's `right/joint_0` to `right_joint_0`."""
    flat = name.replace("/", "_")
    found = [
        i
        for i, other in enumerate(names)
        if (o := other.replace("/", "_")) == flat or o.endswith(f"_{flat}")
    ]
    if len(found) != 1:
        raise SystemExit(f"{name!r} matches {[names[i] for i in found]} in the scene.")
    return found[0]


def _state_map(
    contract: dict, model
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Playground's qpos and qvel indices, and the scene's they go to, joint by joint."""
    import mujoco

    pg = contract["model"]
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        for j in range(model.njnt)
    ]
    pg_q, pg_v, q, v = [], [], [], []
    for joint, qadr, vadr in zip(
        pg["qpos_joints"], pg["jnt_qposadr"], pg["jnt_dofadr"]
    ):
        j = _match(joint, names)
        nq, nv = _widths(model.jnt_type[j])
        pg_q += range(qadr, qadr + nq)
        pg_v += range(vadr, vadr + nv)
        q += range(model.jnt_qposadr[j], model.jnt_qposadr[j] + nq)
        v += range(model.jnt_dofadr[j], model.jnt_dofadr[j] + nv)
    return np.array(pg_q), np.array(pg_v), np.array(q), np.array(v)


def _ctrl_map(contract: dict, model) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        for a in range(model.nu)
    ]
    scene = [_match(name, names) for name in contract["model"]["actuators"]]
    return np.arange(len(scene)), np.array(scene)


def _stale_kinematics(env) -> None:
    """Bodies and sites one physics substep back, the joints where they are."""
    import mujoco
    import torch

    data, model = env.sim.data, env.sim.mj_model
    qpos = data.qpos[0].clone()
    stale = qpos.cpu().numpy().astype(np.float64)
    qvel = data.qvel[0].cpu().numpy().astype(np.float64)
    mujoco.mj_integratePos(model, stale, qvel, -model.opt.timestep)
    data.qpos[0] = torch.as_tensor(stale, dtype=qpos.dtype, device=qpos.device)
    env.sim.forward()
    data.qpos[0] = qpos


def _camera(env, ref) -> tuple[object, str | None]:
    """The scene's one camera and the reference's frames from Playground's, if any."""
    from mjlab.sensor import CameraSensor

    cameras = [s for s in env.scene.sensors.values() if isinstance(s, CameraSensor)]
    frames = [key for key in ref.files if key.endswith("_frame")]
    if not cameras and not frames:
        return None, None
    if len(cameras) != 1 or len(frames) != 1:
        raise SystemExit(f"Cannot pair cameras {cameras} with frames {frames}.")
    return cameras[0], frames[0]


def _sync_target(env, ref, t: int) -> None:
    """A pick-cube target: Playground's mocap cube, the scene's lifting command."""
    import torch

    target = env.command_manager.get_term("target").target_pos
    value = ref["mocap_pos"][t][0]
    target[0] = torch.as_tensor(value, dtype=target.dtype, device=target.device)


def _sync_motion(env, ref, t: int) -> None:
    """A tracking task's clip frame: Playground's, in the scene's motion command."""
    env.command_manager.get_term("motion").time_steps[0] = int(ref["motion_step"][t])


#: Terms Playground computes from velocities a substep old, which the check cannot take
#: back as it does positions: reported, then fed to the policy as Playground's.
STALE = {"G1WholeBodyTracking": ("base_lin_vel", "base_ang_vel")}

#: What a scene keeps beyond the joints, written from the reference each step.
SYNC: dict[str, Callable] = dict.fromkeys(
    (
        "SO101PickCube",
        "SO101PickCubeVision",
        "TrossenPickCube",
        "TrossenPickCubeVision",
    ),
    _sync_target,
) | {"G1WholeBodyTracking": _sync_motion}


def check(env_id: str, export_dir: Path, in_keys: list[str] | None) -> bool:
    import onnxruntime
    import torch
    from mjswan.mjlab.env import build_mjlab_env

    contract = json.loads((export_dir / "contract.json").read_text())
    ref = np.load(export_dir / "reference.npz")
    env_cfg, added = _env_cfg(env_id)
    # A policy the scene builds itself (CartpoleBalanceVision's frame stack) is built
    # from the Hub's; it is checked as the scene runs it.
    model = added.get("policy")
    source = (
        model.SerializeToString() if model is not None else export_dir / "policy.onnx"
    )
    session = onnxruntime.InferenceSession(source if model is not None else str(source))
    inputs = session.get_inputs()
    out_keys = [
        k if isinstance(k, str) else ",".join(k)
        for k in added.get("out_keys") or ["action"]
    ]
    in_keys = in_keys or added.get("in_keys") or ["actor"]
    if len(in_keys) != len(inputs):
        raise SystemExit(
            f"policy.onnx takes {len(inputs)} inputs; in_keys is {in_keys}."
        )
    env_cfg.scene.num_envs = 1
    env = build_mjlab_env(env_cfg)
    model = env.sim.mj_model
    pg_q, pg_v, q, v = _state_map(contract, model)
    pg_u, u = _ctrl_map(contract, model)
    low, high = model.actuator_ctrlrange[u].T
    limited = model.actuator_ctrllimited[u].astype(bool)
    sync = SYNC.get(env_id)
    stale = STALE.get(env_id, ())
    data = env.sim.data
    camera, frame_key = _camera(env, ref)

    def clip(ctrl: np.ndarray) -> np.ndarray:
        return np.where(limited, np.clip(ctrl, low, high), ctrl)

    images = [k for k in ref.files if ref[k].ndim == 4 and not k.endswith("_frame")]
    errors: dict[str, float] = {}

    def note(key: str, value: float) -> None:
        errors[key] = max(errors.get(key, 0.0), value)

    steps = len(ref["action"])
    env.reset()
    for t in range(steps):
        fresh = t == 0 or bool(ref["done"][t - 1])
        if fresh and t:
            env.reset()
        data.qpos[0, torch.as_tensor(q)] = torch.as_tensor(
            ref["qpos"][t][pg_q], dtype=data.qpos.dtype
        )
        data.qvel[0, torch.as_tensor(v)] = torch.as_tensor(
            ref["qvel"][t][pg_v], dtype=data.qvel.dtype
        )
        if fresh:
            env.sim.forward()
        else:
            _stale_kinematics(env)
        if sync is not None:
            sync(env, ref, t)
        env.sim.sense()
        if camera is not None:
            rgb = camera.data.rgb[0]
            frame = torch.as_tensor(ref[frame_key][t], device=rgb.device)
            gap = (rgb.float() - frame.float()).abs().mean() / 255
            note("camera frame, mean (reported only)", float(gap))
            rgb.copy_(frame)
        env_ids = torch.zeros(1, dtype=torch.long, device=env.device) if fresh else None
        obs = env.observation_manager.compute(update_history=True, env_ids=env_ids)

        feed = {}
        if fresh:
            carry = None
        for key, i in zip(in_keys, inputs):
            if key == "is_init":
                feed[i.name] = np.array([fresh])
                continue
            if key == "adapt_hx":
                feed[i.name] = (
                    carry if carry is not None else np.zeros(i.shape, np.float32)
                )
                continue
            value = obs[key][0].cpu().numpy()
            if len(i.shape) > 2:
                # A stacked image comes flattened, as the browser feeds it.
                value = value.reshape(i.shape[1:])
            if len(i.shape) > 2:
                feed[i.name] = value[None].astype(np.float32)
                # Playground's image input; a stacked one ends with the newest frames.
                image = ref[i.name] if i.name in ref.files else ref[images[0]]
                gap = np.abs(value - image[t][-len(value) :]).max()
                note(f"obs {key} (image)", float(gap))
                continue
            # Playground's flat input is its 1-D groups side by side: here, one group.
            value = value.copy()
            terms = env.observation_manager.active_terms[key]
            dims = env.observation_manager.group_obs_term_dim[key]
            offset = 0
            for name, width in zip(terms, (int(np.prod(d)) for d in dims)):
                end = offset + width
                gap = np.abs(value[offset:end] - ref["obs"][t][offset:end]).max()
                if name in stale:
                    note(f"obs {key}/{name}, one substep apart (reported only)", gap)
                    value[offset:end] = ref["obs"][t][offset:end]
                else:
                    note(f"obs {key}/{name}", float(gap))
                offset = end
            feed[i.name] = value[None].astype(np.float32)
        outputs = dict(zip(out_keys, session.run(None, feed)))
        action = outputs["action"][0]
        carry = outputs.get("next,adapt_hx")
        note("action", float(np.abs(action - ref["action"][t]).max()))

        env.action_manager.process_action(
            torch.as_tensor(
                ref["action"][t][None], dtype=torch.float32, device=env.device
            )
        )
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        if t + 1 < steps and not ref["done"][t]:
            ctrl = clip(data.ctrl[0].cpu().numpy()[u])
            note("ctrl", float(np.abs(ctrl - clip(ref["ctrl"][t + 1][pg_u])).max()))

    passed = True
    for key, value in errors.items():
        tolerance = ACTION_TOLERANCE if key in ("action", "ctrl") else OBS_TOLERANCE
        ok = "reported only" in key or value <= tolerance
        passed = passed and ok
        print(f"{'ok  ' if ok else 'FAIL'} {key}: {value:.2e}")
    print(f"{env_id}: {'PASS' if passed else 'FAIL'} over {steps} steps")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("env_id", help="Playground's ID, e.g. CartpoleBalance")
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--in-keys", help="the policy's input groups, comma-separated")
    args = parser.parse_args()
    in_keys = args.in_keys.split(",") if args.in_keys else None
    return 0 if check(args.env_id, args.export_dir, in_keys) else 1


if __name__ == "__main__":
    raise SystemExit(main())
