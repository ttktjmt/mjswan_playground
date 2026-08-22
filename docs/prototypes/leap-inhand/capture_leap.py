"""Serve the built demo, drive it in headless Chromium, and record what it does.

mjswan needs COOP/COEP for `SharedArrayBuffer` (MuJoCo WASM threading), so the stdlib
server it ships with is the one to use. Frames come out as PNG screenshots and are
assembled into a GIF with Pillow — no ffmpeg in this container.
"""

from __future__ import annotations

import argparse
import http.server
import threading
from functools import partial
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist-leap"
PORT = 8123


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Cross-origin isolation, or `SharedArrayBuffer` is undefined and MuJoCo will not
    start. Same headers `MjswanApp.launch` sets."""

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):  # keep the console for the browser's own output
        pass


def serve(directory: Path, port: int) -> http.server.ThreadingHTTPServer:
    handler = partial(_Handler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def capture(seconds: float, fps: int, out_gif: Path, settle: float) -> dict:
    from playwright.sync_api import sync_playwright

    frames_dir = HERE / "frames"
    frames_dir.mkdir(exist_ok=True)
    for stale in frames_dir.glob("*.png"):
        stale.unlink()

    console: list[str] = []
    stats: dict = {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
            args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        context = browser.new_context(
            viewport={"width": 960, "height": 720},
            record_video_dir=str(HERE / "video"),
            record_video_size={"width": 960, "height": 720},
        )
        page = context.new_page()
        page.on("console", lambda m: console.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))

        page.goto(f"http://127.0.0.1:{PORT}/main/", wait_until="load", timeout=120_000)

        # The engine sets this once the scene and policy are loaded and stepping.
        page.wait_for_function("() => window.__mjswanReady === true", timeout=300_000)
        page.wait_for_timeout(int(settle * 1000))

        # Screenshots contend with the render loop, so take only the stills the contact
        # sheet needs; the video track records the compositor without blocking it.
        stills = 12
        every = max(1.0, seconds / stills)
        elapsed = 0.0
        i = 0
        while elapsed < seconds:
            page.screenshot(path=str(frames_dir / f"frame_{i:04d}.png"))
            page.wait_for_timeout(int(every * 1000))
            elapsed += every
            i += 1

        stats["console"] = console
        video = page.video
        context.close()
        if video:
            stats["video"] = video.path()
        browser.close()

    paths = sorted(frames_dir.glob("*.png"))
    images = [Image.open(p).convert("RGB") for p in paths]
    if images and str(out_gif).endswith(".gif"):
        small = [im.resize((im.width // 2, im.height // 2), Image.LANCZOS) for im in images]
        quantized = [im.quantize(colors=128, method=Image.MEDIANCUT) for im in small]
        quantized[0].save(
            out_gif,
            save_all=True,
            append_images=quantized[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
        stats["frames"] = len(images)
        stats["gif_bytes"] = out_gif.stat().st_size
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--settle", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=HERE / "leap_inhand.gif")
    args = parser.parse_args()

    server = serve(DIST, PORT)
    try:
        stats = capture(args.seconds, args.fps, args.out, args.settle)
    finally:
        server.shutdown()

    for line in stats.get("console", []):
        print(line)
    print(f"\nframes: {stats.get('frames')}  gif: {stats.get('gif_bytes')} bytes -> {args.out}")


if __name__ == "__main__":
    main()
