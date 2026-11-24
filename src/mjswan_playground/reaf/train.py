"""Train one of MuJoCo Playground's RL environments for its mjswan scene.

Playground 1.1.0's environment, trained with Playground's own environment and PPO settings
on one free GPU (a Colab T4), then exported for mjswan. `--env` is the environment's ID in
Playground's `env_registry.py`. `train.ipynb` runs this on Colab. It runs in Playground's
own Python environment, not this package's: Playground needs MuJoCo 4, mjswan 3.11.

Four steps, each a subcommand:

  setup   install Playground 1.1.0 for this machine's CUDA driver
  train   PPO, resuming from the newest checkpoint in --out or on the Hub
  export  the policy as ONNX, its contract and a reference rollout (below)
  upload  --out's exports and the newest checkpoint to a Hugging Face model repository

`all` runs train, export and upload in order. Every step can be run again and picks up
from the newest checkpoint. A Colab session takes its disk with it when it ends, so pass
--hf-repo there (a write token in HF_TOKEN): every checkpoint is also kept on the Hub, one
at least every --save-minutes, and a session that starts with none on disk takes the Hub's
newest. A repository holds one environment's run, named in its `run.json`.

`G1WholeBodyTracking` learns one clip of LAFAN1 retargeted to the G1, --motion, which
`train` fetches from the Hub and converts with Playground's `csv_to_npz`.

`export` writes, under --out/export:

  policy.onnx     the actor in one file, its observation normalizer included
  contract.json   what the policy reads and drives, in order, for the mjswan port
  reference.npz   a deterministic rollout (no observation noise, no domain randomization,
                  no latency): each step's qpos, qvel, mocap_pos, ctrl, observations and
                  action, and each camera frame, so the port's observations can be checked
                  against Playground's numbers

    python train.py setup
    python train.py all --env CartpoleBalance --out cartpole --hf-repo user/cartpole-balance
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PLAYGROUND = "1.1.0"
ENVS = (
    "CartpoleBalance",
    "CartpoleSwingUp",
    "CartpoleBalanceVision",
    "SO101PickCube",
    "SO101PickCubeVision",
    "TrossenPickCube",
    "TrossenPickCubeVision",
    "G1WholeBodyTracking",
)
TRACKING = "G1WholeBodyTracking"
#: LAFAN1 retargeted to the G1, at the revision the clips are taken from.
LAFAN1 = (
    "lvhaidong/LAFAN1_Retargeting_Dataset",
    "ce1572906efe6157840e8474d5a0d7aa87481e74",
)
#: Kept in a deterministic rollout: every training episode drew its geoms flat.
UNRANDOMIZED_TERMS = ("override_flat_materials",)
#: The reference rollout's length.
REFERENCE_SECONDS = 12.0
#: Warp 1.18.0's CUDA 12 build, for drivers that predate CUDA 13 (Colab's T4).
WARP_CU12_WHEELS = "https://github.com/NVIDIA/warp/releases/expanded_assets/v1.18.0"
TORCH_CU12_INDEX = "https://download.pytorch.org/whl/cu129"
TORCH = "torch>=2.7.0,<2.16"
EXTRAS = ["onnx", "onnxruntime", "huggingface_hub"]
CHECKPOINT = re.compile(r"model_(\d+)\.pt$")
GPU_CHECK = (
    "import torch, warp; "
    "print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available()); "
    "print('warp', warp.__version__, 'CUDA devices:', warp.get_cuda_device_count()); "
    "exit(0 if torch.cuda.is_available() and warp.get_cuda_device_count() else 1)"
)


# setup


def _driver_cuda() -> tuple[int, int] | None:
    """The CUDA version the driver supports, as `nvidia-smi` reports it."""
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    match = re.search(r"CUDA Version: (\d+)\.(\d+)", out)
    return (int(match[1]), int(match[2])) if match else None


def _installer() -> list[str]:
    """uv inside a virtual environment, pip for a system Python (Colab's), which uv may
    refuse to write to."""
    if sys.prefix != sys.base_prefix and shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "install"]


def _gpu_usable() -> bool:
    return subprocess.run([sys.executable, "-c", GPU_CHECK]).returncode == 0


def setup(_: argparse.Namespace) -> None:
    if sys.version_info < (3, 12):
        sys.exit(
            f"Playground {PLAYGROUND} needs Python 3.12 to 3.14, not {sys.version}."
        )
    cuda = _driver_cuda()
    packages = [f"playground[rl]=={PLAYGROUND}", *EXTRAS]
    if cuda is None:
        print(
            "No NVIDIA driver: installing for CPU, enough to check the steps, not to train."
        )
        subprocess.run([*_installer(), *packages], check=True)
        return
    if cuda >= (13, 0):
        print(
            f"Driver supports CUDA {cuda[0]}.{cuda[1]}: installing the CUDA 13 build."
        )
        subprocess.run([*_installer(), *packages], check=True)
    else:
        print(
            f"Driver supports CUDA {cuda[0]}.{cuda[1]}: installing the CUDA 12 builds."
        )
        subprocess.run(
            [
                *_installer(),
                "--extra-index-url",
                TORCH_CU12_INDEX,
                "--find-links",
                WARP_CU12_WHEELS,
                "warp-lang==1.18.0+cu12",
                *packages,
            ],
            check=True,
        )
    if _gpu_usable():
        return
    # PyPI's newest torch can be a CUDA 13 build, which a CUDA 12 driver cannot run.
    print("torch cannot use the GPU: reinstalling it from PyTorch's CUDA 12.9 index.")
    subprocess.run(
        [*_installer(), "--force-reinstall", "--index-url", TORCH_CU12_INDEX, TORCH],
        check=True,
    )
    if not _gpu_usable():
        sys.exit("torch or Warp still cannot use the GPU: see the versions above.")


# Shared


def _device() -> str:
    from mujoco_playground._src import xpu

    return xpu.resolve_device(None)


def _entry(env_id: str):
    from mujoco_playground._src.rl import env_registry

    return env_registry.get(env_id)


def _identity(args: argparse.Namespace) -> dict:
    """What a repository's checkpoints were trained on, so a run resumes only its own."""
    identity = {"env": args.env, "playground": PLAYGROUND}
    if args.env == TRACKING:
        identity["motion"] = args.motion
    return identity


def _model_card(args: argparse.Namespace) -> str:
    motion = (
        f"\nThe motion is LAFAN1's `{args.motion}`, retargeted to the G1 in "
        f"[{LAFAN1[0]}](https://huggingface.co/datasets/{LAFAN1[0]}); "
        "LAFAN1 is CC BY-NC-ND 4.0.\n"
        if args.env == TRACKING
        else ""
    )
    return f"""\
---
tags: [mujoco-playground, mjswan, robotics, reinforcement-learning]
---
# {args.env} for mjswan

PPO on MuJoCo Playground {PLAYGROUND}'s `{args.env}`, trained by mjswan_playground's
`reaf/train.py`.
{motion}
- `checkpoints/`: rsl-rl checkpoints, the newest resumes training
- `policy.onnx`, `contract.json`, `reference.npz`: the export an mjswan scene runs
"""


def _motion(args: argparse.Namespace) -> Path:
    """`--motion` as the `.npz` Playground's tracking task reads, converted once."""
    from huggingface_hub import hf_hub_download

    npz = args.out / "motion" / f"{args.motion}.npz"
    if not npz.exists():
        npz.parent.mkdir(parents=True, exist_ok=True)
        repo, revision = LAFAN1
        csv = hf_hub_download(
            repo, f"g1/{args.motion}.csv", repo_type="dataset", revision=revision
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mujoco_playground.tasks.whole_body.tracking.csv_to_npz",
                f"--input_csv={csv}",
                f"--output_npz={npz}",
            ],
            check=True,
        )
    return npz


def _env_kwargs(args: argparse.Namespace, *, deterministic: bool) -> dict:
    if args.env != TRACKING:
        return {}
    # `play` starts each episode at the clip's first frame, without pushes.
    return {"trajectory_path": str(_motion(args)), "play": deterministic}


def _make_deterministic(env_cfg) -> None:
    """Drop the observation and pixel noise, the domain randomization and the simulated
    servo latency, so a rollout's numbers depend on the state alone."""
    for group in env_cfg.observation_adapter.values():
        group.enable_noise = False
        for term in group.terms.values():
            if "pixel_noise_std" in term.params:
                term.params["pixel_noise_std"] = 0.0
    dr = env_cfg.resetters.get("domain_randomization")
    if dr is not None:
        terms = dr.runtime_params["terms"]
        kept = {name: terms[name] for name in UNRANDOMIZED_TERMS if name in terms}
        if isinstance(terms, dict) and kept:
            dr.runtime_params["terms"] = kept
        else:
            env_cfg.resetters.pop("domain_randomization")
    layer = env_cfg.device_layer
    for processors in (layer.command_processors, layer.substep_measurement_processors):
        for name, component in list(processors.items()):
            if component.component_type.__name__.startswith("Stochastic"):
                processors.pop(name)


def _build(args: argparse.Namespace, nworld: int, device: str, *, deterministic: bool):
    """Playground's own env for `--env`, built as its registry builds it."""
    from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
    from reaf.simulation.torch import flat_scene

    entry = _entry(args.env)
    env_cfg = entry.env_cfg_fn(**_env_kwargs(args, deterministic=deterministic))
    if deterministic:
        _make_deterministic(env_cfg)
    env_cfg.sim.nworld = nworld
    env_cfg.sim.xpu_device = device
    env_cfg.sim.impl = "warp"
    env = flat_scene.build_env(env_cfg)
    runner_cfg = entry.runner_cfg
    runner_cfg.nworld = nworld
    if not device.startswith("cuda"):
        runner_cfg.torch_compile_mode = None
    wrapped = vecenv_wrapper.RslRlVecEnvWrapper(
        env, clip_actions=runner_cfg.clip_actions, cfg=runner_cfg.vecenv_wrapper_cfg
    )
    return env_cfg, runner_cfg, wrapped


def _iteration(path: Path) -> int:
    match = CHECKPOINT.search(path.name)
    return int(match[1]) if match else -1


def _newest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=_iteration)
    return checkpoints[-1] if checkpoints else None


def _hub():
    from huggingface_hub import HfApi

    return HfApi()


def _checkpoint(args: argparse.Namespace) -> Path | None:
    """The newest checkpoint on disk, else the Hub's, for a session that starts empty."""
    run_dir = args.out / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _newest_checkpoint(run_dir)
    if checkpoint is None and args.hf_repo:
        checkpoint = _pull_newest_checkpoint(args.hf_repo, run_dir)
    return checkpoint


def _pull_newest_checkpoint(repo: str, run_dir: Path) -> Path | None:
    from huggingface_hub import hf_hub_download

    names = [
        name
        for name in _hub().list_repo_files(repo)
        if name.startswith("checkpoints/") and CHECKPOINT.search(name)
    ]
    if not names:
        return None
    newest = max(names, key=lambda name: _iteration(Path(name)))
    print(f"Fetched {newest} from the Hub.")
    cached = hf_hub_download(repo, newest)
    target = run_dir / Path(newest).name
    shutil.copyfile(cached, target)
    return target


def _claim_repo(args: argparse.Namespace) -> None:
    """Create the repository and write its card, so a token that cannot write fails now
    rather than in the background on the first checkpoint, and refuse a repository that
    holds another environment's run."""
    from huggingface_hub import hf_hub_download

    api = _hub()
    repo = args.hf_repo
    api.create_repo(repo, exist_ok=True)
    identity = _identity(args)
    if api.file_exists(repo, "run.json"):
        held = json.loads(Path(hf_hub_download(repo, "run.json")).read_text())
        if held != identity:
            sys.exit(f"{repo} holds a run of {held}, not of {identity}: use another.")
    else:
        api.upload_file(
            path_or_fileobj=(json.dumps(identity, indent=2) + "\n").encode(),
            path_in_repo="run.json",
            repo_id=repo,
        )
    api.upload_file(
        path_or_fileobj=_model_card(args).encode(),
        path_in_repo="README.md",
        repo_id=repo,
    )


def _push(repo: str, path: Path, path_in_repo: str) -> None:
    _hub().upload_file(
        path_or_fileobj=str(path), path_in_repo=path_in_repo, repo_id=repo
    )


def _nworld_for(device: str, default: int) -> int:
    """Playground's own count where the GPU holds it, at most 4096 on a 16 GB card."""
    import torch

    if not device.startswith("cuda"):
        return 16
    memory = torch.cuda.get_device_properties(torch.device(device)).total_memory
    return default if memory >= 40 * 2**30 else min(default, 4096)


# train


def train(args: argparse.Namespace) -> None:
    import dataclasses
    import os

    import warp as wp
    from mujoco_playground._src.rl import rng
    from rsl_rl.runners import OnPolicyRunner

    defaults = _entry(args.env).runner_cfg
    iterations = args.iterations or defaults.max_iterations
    run_dir = args.out / "checkpoints"
    if args.hf_repo:
        _claim_repo(args)
    resume = _checkpoint(args)
    done = _iteration(resume) if resume else -1
    if done >= iterations - 1:
        print(f"Already trained: {resume.name}.")
        return

    device = _device()
    if device == "cpu" and _driver_cuda() is not None:
        sys.exit("There is a GPU, but torch cannot use it: run `setup` again.")
    rng.seed(args.seed)
    if args.warp_cache:
        wp.config.kernel_cache_dir = str(args.warp_cache)
    nworld = args.nworld or _nworld_for(device, defaults.nworld)
    print(f"Training {args.env} on {device} with {nworld} worlds, to {iterations}.")
    _, runner_cfg, env = _build(args, nworld, device, deterministic=False)
    for key, value in json.loads(args.overrides or "{}").items():
        setattr(runner_cfg, key, value)
    runner_cfg.seed = args.seed

    uploads: list[threading.Thread] = []
    failed: set[Path] = set()

    def push(saved: Path) -> None:
        try:
            _push(args.hf_repo, saved, f"checkpoints/{saved.name}")
        except Exception as error:  # The next checkpoint may still get there.
            failed.add(saved)
            print(f"Could not upload {saved.name} to the Hub: {error}", flush=True)

    class Runner(OnPolicyRunner):
        saved_at = time.monotonic()

        def save(self, path: str, infos: dict | None = None) -> None:
            super().save(path, infos)
            self.saved_at = time.monotonic()
            if args.hf_repo:
                # Off the training loop: the next iterations need not wait for the Hub.
                upload = threading.Thread(target=push, args=(Path(path),))
                upload.start()
                uploads.append(upload)

    runner = Runner(
        env, dataclasses.asdict(runner_cfg), log_dir=str(run_dir), device=device
    )
    if args.save_minutes:
        # A session can end between rsl-rl's `save_interval` saves; rsl-rl logs every
        # iteration, so the log also saves once --save-minutes have passed.
        log = runner.logger.log

        def log_and_save(*log_args, **log_kwargs) -> None:
            log(*log_args, **log_kwargs)
            it = runner.current_learning_iteration
            due = time.monotonic() - runner.saved_at > args.save_minutes * 60
            if due and it % runner_cfg.save_interval:
                runner.save(os.path.join(runner.logger.log_dir, f"model_{it}.pt"))

        runner.logger.log = log_and_save
    if resume:
        # Actor, critic, optimizer and iteration: a fresh critic would undo the value fit.
        print(f"Resuming from {resume.name}.")
        runner.load(str(resume), map_location=device)
    # rsl-rl reruns the checkpoint's own iteration, so this ends on `iterations - 1`.
    runner.learn(
        num_learning_iterations=iterations - max(done, 0),
        init_at_random_ep_len=True,
    )
    env.close()
    for upload in uploads:
        upload.join()
    newest = _newest_checkpoint(run_dir)
    if newest in failed:
        # The last one is what the next session resumes from: this time a failure stops.
        _push(args.hf_repo, newest, f"checkpoints/{newest.name}")


# export


def _numpy(value):
    """A copy: on CPU the simulator's buffers would be aliased, every step the last."""
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.array(value.numpy(), copy=True)


def _jsonable(value):
    """A config value as JSON: sequences and tensors as lists, a term as its function and
    parameters, any other function or class by its name."""
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "func") and hasattr(value, "params"):
        return {"func": _jsonable(value.func), "params": _jsonable(value.params)}
    return getattr(value, "__name__", repr(value))


def _component(component) -> dict:
    return {
        "type": component.component_type.__name__,
        "params": _jsonable(component.runtime_params),
    }


def _term(name: str, cfg) -> dict:
    entry = {
        "name": name,
        "shape": list(cfg.output_shape or []),
        "source": _jsonable(cfg.func),
        "params": _jsonable(cfg.params),
    }
    for key in ("scale", "offset", "bias", "clip"):
        if getattr(cfg, key) is not None:
            entry[key] = _jsonable(getattr(cfg, key))
    if cfg.history_length:
        entry["history_length"] = cfg.history_length
    return entry


def _contract(args, env_cfg, actor_groups, mj_model, control_dt: float) -> dict:
    """What the policy reads and drives, by name, in the order it reads them, and every
    device and resetter it was trained with."""
    import mujoco

    def names(kind, count: int) -> list[str]:
        return [mujoco.mj_id2name(mj_model, kind, i) or "" for i in range(count)]

    devices = env_cfg.device_layer.devices
    contract = {
        "env": args.env,
        "playground": PLAYGROUND,
        "control_dt": control_dt,
        "physics_dt": float(mj_model.opt.timestep),
        "observations": {
            group: [
                _term(name, cfg)
                for name, cfg in env_cfg.observation_adapter[group].terms.items()
            ]
            for group in actor_groups
        },
        "action": {
            name: _component(component)
            for name, component in env_cfg.task_layer.command_processors.items()
        },
        "devices": {name: _component(component) for name, component in devices.items()},
        "resetters": {
            name: _component(component) for name, component in env_cfg.resetters.items()
        },
        "model": {
            "qpos_joints": names(mujoco.mjtObj.mjOBJ_JOINT, mj_model.njnt),
            "jnt_qposadr": [int(a) for a in mj_model.jnt_qposadr],
            "jnt_dofadr": [int(a) for a in mj_model.jnt_dofadr],
            "actuators": names(mujoco.mjtObj.mjOBJ_ACTUATOR, mj_model.nu),
            "mocap_bodies": [
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b)
                for b in range(mj_model.nbody)
                if mj_model.body_mocapid[b] >= 0
            ],
        },
    }
    if "camera" in devices:
        camera = _jsonable(devices["camera"].runtime_params)
        name = camera["cam_name"]
        name = name[0] if isinstance(name, list) else name
        camera["fovy"] = float(mj_model.cam_fovy[mj_model.camera(name).id])
        contract["camera"] = camera
    if args.env == TRACKING:
        contract["motion"] = {
            "dataset": LAFAN1[0],
            "revision": LAFAN1[1],
            "file": f"g1/{args.motion}.csv",
        }
    return contract


def _motion_steps(env):
    """The tracking task's clip frame per world, or None for a task without a clip."""
    tasks = env.unwrapped.task_layer
    if not tasks.has_component("motion"):
        return None
    return tasks.get_component_by_name("motion").time_steps


def export(args: argparse.Namespace) -> None:
    import dataclasses

    import numpy as np
    import onnx
    import onnxruntime
    import torch
    import warp as wp
    from mujoco_playground._src.rl import rng
    from rsl_rl.runners import OnPolicyRunner

    checkpoint = _checkpoint(args)
    if checkpoint is None:
        sys.exit("No checkpoint on disk or on the Hub: run `train` first.")
    export_dir = args.out / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    device = _device()
    rng.seed(args.seed)
    if args.warp_cache:
        wp.config.kernel_cache_dir = str(args.warp_cache)
    env_cfg, runner_cfg, env = _build(args, 1, device, deterministic=True)
    runner = OnPolicyRunner(
        env, dataclasses.asdict(runner_cfg), log_dir=None, device=device
    )
    runner.load(
        str(checkpoint),
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
        },
        map_location=device,
    )
    runner.export_policy_to_onnx(str(export_dir), "policy.onnx")
    # torch writes the weights to a file of their own; a host loads one file.
    onnx_path = export_dir / "policy.onnx"
    onnx.save(onnx.load(str(onnx_path)), str(onnx_path))
    (export_dir / "policy.onnx.data").unlink(missing_ok=True)
    policy = runner.get_inference_policy(device)
    print(f"Exported {checkpoint.name} to {onnx_path}.")

    layer = env.unwrapped.device_layer
    sim = layer.sim
    control_dt = float(layer.control_timestep)
    steps = args.reference_steps or round(REFERENCE_SECONDS / control_dt)
    actor_groups = list(runner_cfg.obs_groups["actor"])
    obs = env.get_observations()
    # The network's flat input is its 1-D groups side by side; each image is an input.
    flat = [group for group in actor_groups if obs[group].ndim == 2]
    images = [group for group in actor_groups if obs[group].ndim > 2]
    camera = (
        layer.get_device("camera") if "camera" in env_cfg.device_layer.devices else None
    )
    motion_steps = _motion_steps(env)
    record: dict[str, list] = {
        key: []
        for key in ("qpos", "qvel", "mocap_pos", "ctrl", "obs", *images, "action")
    }
    if motion_steps is not None:
        record["motion_step"] = []
    for _ in range(steps):
        with torch.inference_mode():
            action = policy(obs)
        record["qpos"].append(_numpy(sim.data.qpos)[0])
        record["qvel"].append(_numpy(sim.data.qvel)[0])
        record["mocap_pos"].append(_numpy(sim.data.mocap_pos)[0])
        record["ctrl"].append(_numpy(sim.data.ctrl)[0])
        record["obs"].append(np.concatenate([_numpy(obs[g])[0] for g in flat]))
        for group in images:
            record[group].append(_numpy(obs[group])[0])
        if camera is not None:
            for key, frame in camera.get_measurements(sim).items():
                record.setdefault(f"{key.replace('/', '_')}_frame", []).append(
                    _numpy(frame)[0]
                )
        if motion_steps is not None:
            record["motion_step"].append(int(motion_steps[0]))
        record["action"].append(_numpy(action)[0])
        obs, _, _, _ = env.step(action)
    arrays = {
        key: np.asarray(
            values,
            dtype=np.uint8
            if key.endswith("_frame")
            else np.int64
            if key == "motion_step"
            else np.float32
            if key in images
            else np.float64,
        )
        for key, values in record.items()
    }
    np.savez_compressed(export_dir / "reference.npz", **arrays)

    # The ONNX graph must give the torch policy's actions on the same observations.
    session = onnxruntime.InferenceSession(str(onnx_path))
    inputs = session.get_inputs()
    onnx_actions = np.concatenate(
        [
            session.run(
                None,
                {i.name: arrays[i.name][t][None].astype(np.float32) for i in inputs},
            )[0]
            for t in range(steps)
        ]
    )
    gap = float(np.abs(onnx_actions - arrays["action"]).max())
    if gap > 1e-4:
        sys.exit(f"policy.onnx disagrees with the checkpoint by {gap:.2e}.")

    contract = _contract(args, env_cfg, actor_groups, sim.mj_model, control_dt)
    contract["inputs"] = [
        {
            "name": i.name,
            "groups": flat if i.name == "obs" else [i.name],
            "shape": list(i.shape),
        }
        for i in inputs
    ]
    contract["checkpoint"] = checkpoint.name
    contract["reference_steps"] = steps
    contract["onnx_vs_torch_max_abs"] = gap
    (export_dir / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"Wrote contract.json and reference.npz ({steps} steps).")
    env.close()


# upload


def upload(args: argparse.Namespace) -> None:
    if not args.hf_repo:
        print("No --hf-repo: nothing to upload. Everything is in", args.out)
        return
    export_dir = args.out / "export"
    if not (export_dir / "policy.onnx").exists():
        sys.exit("No export yet: run `export` first.")
    api = _hub()
    api.create_repo(args.hf_repo, exist_ok=True)
    api.upload_folder(folder_path=str(export_dir), repo_id=args.hf_repo)
    checkpoint = _newest_checkpoint(args.out / "checkpoints")
    if checkpoint:
        _push(args.hf_repo, checkpoint, f"checkpoints/{checkpoint.name}")
    print(f"Uploaded to https://huggingface.co/{args.hf_repo}")


def run_all(args: argparse.Namespace) -> None:
    train(args)
    export(args)
    upload(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("setup", help="install Playground for this machine").set_defaults(
        run=setup
    )
    for name, run in (
        ("train", train),
        ("export", export),
        ("upload", upload),
        ("all", run_all),
    ):
        step = steps.add_parser(name)
        step.set_defaults(run=run)
        step.add_argument("--env", choices=ENVS, required=True, help="Playground's ID")
        step.add_argument(
            "--out", type=Path, required=True, help="where everything is kept"
        )
        step.add_argument(
            "--hf-repo", help="a Hugging Face model repository, user/name"
        )
        step.add_argument(
            "--motion",
            default="dance1_subject2",
            help=f"{TRACKING}'s clip, a file name in {LAFAN1[0]}'s g1/ without .csv",
        )
        step.add_argument(
            "--iterations", type=int, help="default: Playground's own for --env"
        )
        step.add_argument(
            "--nworld", type=int, help="parallel worlds (default: by GPU memory)"
        )
        step.add_argument("--seed", type=int, default=0)
        step.add_argument(
            "--overrides", help='runner settings as JSON, e.g. {"save_interval": 50}'
        )
        step.add_argument(
            "--save-minutes",
            type=float,
            default=10.0,
            help="save a checkpoint at least this often (0: only Playground's interval)",
        )
        step.add_argument(
            "--reference-steps",
            type=int,
            help=f"default: {REFERENCE_SECONDS:g} s of control steps",
        )
        step.add_argument(
            "--warp-cache",
            type=Path,
            help="compiled kernels (default: Warp's per-user cache)",
        )
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
