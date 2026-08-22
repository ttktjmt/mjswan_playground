"""Read the running sim out of the browser: cube yaw, action, integrator target."""
from __future__ import annotations
import json, math
from pathlib import Path
from capture_leap import serve, DIST, PORT

def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def main() -> None:
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
            page.wait_for_function("() => window.__mjswanReady === true", timeout=300_000)
            page.wait_for_function(
                "() => window.__mjswanRuntime?.debugProbe?.()?.qpos", timeout=120_000
            )
            page.wait_for_timeout(3000)

            samples = []
            for _ in range(14):
                p = page.evaluate("() => window.__mjswanRuntime.debugProbe()")
                samples.append(p)
                page.wait_for_timeout(2000)
            b.close()
    finally:
        server.shutdown()

    first = samples[0]
    print("controlType :", first["controlType"], "/ relative_to:", first["relativeTo"])
    print("n terms     :", first["nTerms"], " decimation:", first["decimation"])
    print("nq          :", len(first["qpos"]))
    print("limitLo[:4] :", None if not first["limitLo"] else [round(v,3) for v in first["limitLo"][:4]])
    print()
    print(" t(sim s)  cubeYaw(rad)  |action|max  cmdTarget[:3]                 jointPosTarget[:3]")
    for s in samples:
        q = s["qpos"]
        # the cube's free joint is the last 7 of qpos: 16 hand joints then pos+quat
        cube_quat = q[-4:]
        yaw = quat_yaw(cube_quat)
        a = s["action"] or []
        amax = max((abs(v) for v in a), default=float("nan"))
        ct = s["commandTarget"] or []
        jt = s["jointPosTarget"] or []
        print(f"{s['time']:9.2f} {yaw:13.3f} {amax:12.3f}  "
              f"{[round(v,3) for v in ct[:3]]!s:28} {[round(v,3) for v in jt[:3]]}")

if __name__ == "__main__":
    main()
