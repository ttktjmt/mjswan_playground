"""Compare the browser's first observation vector against mjlab's, element by element."""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "disable")
import numpy as np
from capture_leap import serve, DIST, PORT


def browser_obs() -> dict:
    from playwright.sync_api import sync_playwright
    server = serve(DIST, PORT)
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(
                executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
                args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
            )
            page = b.new_page(viewport={"width": 640, "height": 480})
            page.goto(f"http://127.0.0.1:{PORT}/main/", wait_until="load", timeout=120_000)
            page.wait_for_function("() => window.__mjswanRuntime?.debugProbe?.()?.obs",
                                   timeout=300_000)
            page.wait_for_timeout(4000)
            p = page.evaluate("() => window.__mjswanRuntime.debugProbe()")
            b.close()
            return p
    finally:
        server.shutdown()


def mjlab_obs(steps: int):
    import onnxruntime as ort, torch
    import leap_inhand_task as task
    from mjlab.envs import ManagerBasedRlEnv
    from export_leap_policy import OUT as POLICY_ONNX
    env = ManagerBasedRlEnv(cfg=task.make_env_cfg("integrator"), device="cpu")
    sess = ort.InferenceSession(str(POLICY_ONNX))
    obs, _ = env.reset()
    for _ in range(steps):
        v = obs["actor"].numpy().astype(np.float32)
        a = np.clip(sess.run(None, {"obs": v})[0], -1, 1)
        obs, *_ = env.step(torch.from_numpy(a))
    return obs["actor"].numpy()[0], a[0]


def main() -> None:
    p = browser_obs()
    b = np.asarray(p["obs"]["policy"], dtype=np.float32)
    print(f"browser obs: {b.shape}, sim time {p['time']:.2f}s")
    steps = int(round(p["time"] / 0.05))
    m, ma = mjlab_obs(steps)
    print(f"mjlab  obs: {m.shape}, after {steps} steps\n")

    for name, sl in (("joint_pos  (t-9..t)", slice(0, 160)),
                     ("cmd_pos    (t-9..t)", slice(160, 320))):
        print(f"{name}: browser |v| mean {np.abs(b[sl]).mean():.4f}  "
              f"min {b[sl].min():+.3f} max {b[sl].max():+.3f}")
        print(f"{' '*len(name)}  mjlab   |v| mean {np.abs(m[sl]).mean():.4f}  "
              f"min {m[sl].min():+.3f} max {m[sl].max():+.3f}")
    print("\nlast frame of each term (16 values):")
    print("  browser joint_pos[t]:", np.round(b[144:160], 3))
    print("  mjlab   joint_pos[t]:", np.round(m[144:160], 3))
    print("  browser cmd_pos[t]  :", np.round(b[304:320], 3))
    print("  mjlab   cmd_pos[t]  :", np.round(m[304:320], 3))
    print("\nbrowser action:", np.round(np.asarray(p["action"]), 3))
    print("mjlab   action:", np.round(ma, 3))


if __name__ == "__main__":
    main()
