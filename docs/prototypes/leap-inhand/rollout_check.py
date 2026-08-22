"""Roll the ONNX policy out in the real mjlab env and measure what the cube does.

The browser is the wrong place to debug a policy: this runs the same observations and
the same graph against mjlab itself, so a broken pipeline shows up as numbers instead of
a video. Compares the two candidate action terms:

  integrator  upstream's JointPositionDeltaAction — q_cmd += delta, clamped
  relative    mjlab's RelativeJointPositionAction — q_cmd = q + delta
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

import leap_inhand_task as task  # noqa: E402
from export_leap_policy import OUT as POLICY_ONNX  # noqa: E402


def cube_yaw(env) -> np.ndarray:
    from mjlab.utils.lab_api.math import euler_xyz_from_quat

    quat = env.scene["cube"].data.root_link_quat_w
    _, _, yaw = euler_xyz_from_quat(quat)
    return yaw.detach().cpu().numpy()


def unwrap(series: list[float]) -> np.ndarray:
    return np.unwrap(np.asarray(series))


def run(kind: str, steps: int) -> dict:
    from mjlab.envs import ManagerBasedRlEnv

    cfg = task.make_env_cfg(action=kind)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    session = ort.InferenceSession(str(POLICY_ONNX))

    obs, _ = env.reset()
    yaws, heights = [], []
    for _ in range(steps):
        vector = obs["actor"].detach().cpu().numpy().astype(np.float32)
        action = session.run(None, {"obs": vector})[0]
        action = np.clip(action, -1.0, 1.0)
        obs, _, terminated, truncated, _ = env.step(torch.from_numpy(action))
        yaws.append(float(cube_yaw(env)[0]))
        heights.append(float(env.scene["cube"].data.root_link_pos_w[0, 2]))
        if bool(terminated[0]) or bool(truncated[0]):
            break

    total = unwrap(yaws)
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    turned = float(total[-1] - total[0]) if len(total) > 1 else 0.0
    return {
        "kind": kind,
        "steps": len(yaws),
        "seconds": len(yaws) * dt,
        "yaw_rad": turned,
        "yaw_rate": turned / max(len(yaws) * dt, 1e-9),
        "min_height": float(np.min(heights)),
        "dropped": len(yaws) < steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--kinds", nargs="+", default=["integrator", "relative"])
    args = ap.parse_args()

    print(f"{'action':<12}{'steps':>7}{'sim s':>8}{'yaw rad':>10}"
          f"{'rad/s':>9}{'min z':>8}  note")
    for kind in args.kinds:
        r = run(kind, args.steps)
        note = "DROPPED/terminated early" if r["dropped"] else ""
        print(f"{r['kind']:<12}{r['steps']:>7}{r['seconds']:>8.1f}{r['yaw_rad']:>10.2f}"
              f"{r['yaw_rate']:>9.3f}{r['min_height']:>8.3f}  {note}")
    print("\nupstream's training target for rotation_progress was 0.20 rad/s.")


if __name__ == "__main__":
    main()
