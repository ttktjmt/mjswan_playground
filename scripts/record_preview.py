"""Record a task's README preview GIF straight out of its built browser app.

    uv run --group previews playwright install chromium   # once
    uv run --group previews python scripts/record_preview.py husky
    uv run --group previews python scripts/record_preview.py --all

Builds the task if `dist/<task-id>` is missing, serves it with the COOP/COEP headers
MuJoCo WASM needs, drives the app's own control panel to set the shot up, then films the
canvas into `assets/<task-id>.gif` at 480x351 / 15 fps — the size, aspect ratio and frame
rate mjlab_playground's previews use.

Two things here are not obvious:

- **The browser runs headed.** Headless Chromium steps the simulation at ~0.07x real time
  (the step loop is main-thread work), so a headless recording comes out in slow motion.
  Every run prints its measured control-step rate and warns if the clip came out slow.
- **Frames come from CDP's screencast, not Playwright's video.** Each frame carries its
  presentation timestamp, so the 15 fps pick is evenly spaced instead of resampled off a
  fixed-25 fps encode.

Adding a task: register it in `mjswan_playground.registry`, then add a `Preview` below.
The defaults film whatever the app opens with; `steps` drives the control panel first (the
same widgets a visitor would touch), `orbit` swings the camera, `crop` frames it. Dial a
new one in with `--shot`, which stops after the setup and writes the framing as a PNG:

    uv run --group previews python scripts/record_preview.py husky --shot /tmp/f.png \\
        --orbit 150 --crop 960:702:0:0    # the whole frame, to find the crop from
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import functools
import http.server
import json
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from mjswan_playground.registry import ALL_TASKS, load

ROOT = Path(__file__).resolve().parent.parent

#: The reference previews' geometry (mjlab_playground): 480x351 at 15 fps.
WIDTH, HEIGHT, FPS = 480, 351, 15
#: Filmed at 2x and downscaled into the GIF, so a crop still has real pixels behind it.
CAPTURE_SCALE = 2
#: Fraction of the policy's own control rate a recording has to hold to count as real time.
MIN_STEP_RATE_RATIO = 0.9


@dataclass(frozen=True)
class Preview:
    """How one task's preview is filmed.

    Attributes:
        steps: Control-panel actions, applied before the panel is hidden. Each is one of
            ``("select", <input id>, <option>)`` for the scene/policy/motion pickers,
            ``("number", <slider label>, <value>)``, ``("checkbox", <label>, <bool>)``,
            ``("click", <button label>)`` or ``("wait", <seconds>)``.
        motions: Motions to film one after another, concatenated into one GIF with
            `seconds` split evenly between them — for a tracking policy, where the point
            is the range of motions rather than any single one. Empty films one segment
            of whatever `steps` left running.
        query: Extra URL query, e.g. ``"ref=0"`` to drop the motion-tracking ghost.
        orbit: Degrees to swing the camera, as a drag on empty background would. Positive
            adds to the scene's authored `azimuth` (the camera travels anticlockwise seen
            from above). Body tracking keeps whatever angle the drag leaves.
        crop: ffmpeg crop of the 960x702 capture, ``w:h:x:y``, at the 480:351 aspect —
            centres the robot and drops the empty sky above it.
        seconds: Clip length. 4 s is 60 frames at 15 fps.
        settle: Seconds between finishing the setup and rolling, so camera damping and any
            reset from a policy switch are done before the first frame.
    """

    steps: tuple[tuple, ...] = ()
    motions: tuple[str, ...] = ()
    query: str = ""
    orbit: float = 0.0
    crop: str = "719:526:120:120"
    seconds: float = 4.0
    settle: float = 2.0


#: One entry per task in `ALL_TASKS`; a task with no entry is filmed with the defaults.
#: Every `orbit` here swings the authored camera round to the robot's front. Which swing
#: that is, is worth measuring rather than deriving: the scene's authored azimuth is in its
#: `manifest.json`, but whether a robot spawns facing +x or -x is the task's business (of
#: these four only husky faces +x). `--shot` settles it in one run.
PREVIEWS: dict[str, Preview] = {
    # Push Speed defaults to 1.0, so the skater is already riding.
    "husky": Preview(orbit=150, crop="719:526:121:127"),
    # `idle_02` opens the motion list and stands still, so the preview picks its own two:
    # one policy over a sprint and a flip is the point of a tracking task. `ref=0` drops
    # the tracking ghost, which otherwise stands beside the robot and halves its size.
    "wbc": Preview(
        motions=("sprint_01", "flip_02"),
        query="ref=0",
        orbit=150,
        crop="719:526:115:122",
        seconds=6.0,
    ),
    # Auto throw is armed by default: a ball every 1-4 s, so a 5 s clip catches two.
    # A wider crop than the rest, to keep the incoming ball in frame.
    "pacman": Preview(orbit=90, crop="840:614:60:76", seconds=5.0),
    "microduck": Preview(
        steps=(("number", "Forward (m/s)", 0.35), ("wait", 1)),
        orbit=140,
        crop="719:526:121:68",
    ),
}


# ── serving the built app ─────────────────────────────────────────────────────


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Static files plus the headers SharedArrayBuffer (MuJoCo WASM threading) needs."""

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, *args) -> None:
        pass  # a request log per asset drowns out the recording's own output


def serve(directory: Path) -> socketserver.TCPServer:
    """Serve `directory` on a free port from a daemon thread."""
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(
        ("127.0.0.1", 0), functools.partial(_Handler, directory=str(directory))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def ensure_built(task_id: str, dist: Path) -> Path:
    """The task's built app, building it into `dist/<task-id>` if it isn't there yet."""
    app_dir = dist / task_id
    if not (app_dir / "manifest.json").exists():
        print(f"[{task_id}] building into {app_dir}")
        load(task_id).build(output_dir=app_dir.resolve())
    return app_dir


def control_rate(app_dir: Path) -> float:
    """Control steps per second the app's first scene runs at, from its manifest."""
    manifest = json.loads((app_dir / "manifest.json").read_text())
    control_dt = manifest["projects"][0]["scenes"][0].get("control_dt")
    return 1 / control_dt if control_dt else 50.0


# ── the browser side ──────────────────────────────────────────────────────────


def _labelled_input(page, label: str, selector: str) -> str:
    """CSS id of the `selector` input on the panel row labelled `label`.

    Mantine generates those ids, so they are read back off the DOM (and assigned when the
    widget has none) rather than guessed.
    """
    element_id = page.evaluate(
        """([label, selector]) => {
            const l = [...document.querySelectorAll('label')]
                .find((e) => e.textContent.trim() === label);
            for (let n = l; n; n = n.parentElement) {
                const hit = n.querySelector(selector);
                if (hit) return hit.id || (hit.id = 'rec-' + Math.random().toString(36).slice(2));
            }
            return null;
        }""",
        [label, selector],
    )
    if not element_id:
        raise SystemExit(f"no {selector} on the panel row labelled {label!r}")
    return f"#{element_id}"


def _apply(page, step: tuple) -> None:
    """Run one `Preview.steps` entry against the control panel."""
    kind, *rest = step
    if kind == "number":
        label, value = rest
        target = _labelled_input(page, label, "input[type=text]")
        page.fill(target, str(value))
        page.press(target, "Enter")
    elif kind == "select":
        input_id, option = rest
        page.click(f"#{input_id}")
        page.get_by_role("option", name=option, exact=True).click()
        page.wait_for_function("() => window.__mjswanReady", timeout=120_000)
    elif kind == "checkbox":
        label, value = rest
        page.set_checked(_labelled_input(page, label, "input[type=checkbox]"), value)
    elif kind == "click":
        page.get_by_role("button", name=rest[0], exact=True).click()
    elif kind == "wait":
        page.wait_for_timeout(rest[0] * 1000)
    else:
        raise SystemExit(f"unknown preview step {step!r}")


def _orbit(page, degrees: float) -> None:
    """Swing the camera by dragging empty background, as a visitor would.

    OrbitControls turns a full circle per viewport height of horizontal drag, and body
    tracking only carries the orbit target around, so the angle survives the whole clip.
    The drag starts in a top corner on purpose: a pointerdown that lands on a body pulls
    the robot instead (`dragStateManager` takes the drag and OrbitControls sits out).
    """
    if not degrees:
        return
    dx = degrees / 360 * HEIGHT
    x, y = (24, 24) if dx >= 0 else (WIDTH - 24, 24)
    page.mouse.move(x, y)
    page.mouse.down()
    for i in range(1, 21):  # in steps, so the controls integrate it as a real drag
        page.mouse.move(x + dx * i / 20, y)
        page.wait_for_timeout(16)
    page.mouse.up()


#: Counts the step loop's own pacing: one `setTimeout` per on-schedule control step, two
#: `MessageChannel` posts per step that ran late. Installed before the app's own scripts.
#: An IIFE, not a function: `add_init_script` takes source to run, and a bare function
#: expression would be evaluated and dropped, leaving the counters undefined.
_STEP_COUNTER = """(() => {
    window.__paced = 0; window.__yield = 0;
    const timeout = window.setTimeout;
    window.setTimeout = function (fn, ms, ...rest) {
        if (ms > 0 && ms <= 25) window.__paced++;
        return timeout.call(this, fn, ms, ...rest);
    };
    const post = MessagePort.prototype.postMessage;
    MessagePort.prototype.postMessage = function (...args) {
        window.__yield++;
        return post.apply(this, args);
    };
})();"""


def film(
    url: str,
    preview: Preview,
    frames_dir: Path | None = None,
    shot: Path | None = None,
    expected_rate: float = 50.0,
    motion: str | None = None,
    seconds: float | None = None,
    start_index: int = 0,
) -> int:
    """Set the shot up, then write either a framing PNG (`shot`) or a PNG sequence.

    One call is one segment: `motion` picks it in the panel, `seconds` overrides the
    recipe's length and `start_index` continues an earlier segment's numbering, so a
    multi-motion preview concatenates into a single sequence. Returns the next index.
    """
    from playwright.sync_api import sync_playwright

    seconds = preview.seconds if seconds is None else seconds

    with sync_playwright() as pw:
        # Headed: see the module docstring. Headless films slow motion.
        browser = pw.chromium.launch(headless=False, args=["--hide-scrollbars"])
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=CAPTURE_SCALE,
        )
        context.add_init_script(_STEP_COUNTER)
        page = context.new_page()
        page.on("pageerror", lambda e: print(f"  [pageerror] {e}", file=sys.stderr))
        page.goto(url, wait_until="load")
        page.wait_for_function(
            "() => window.__mjswanReady || window.__mjswanError", timeout=240_000
        )
        if page.evaluate("() => window.__mjswanError"):
            raise SystemExit(f"{url} failed to load its scene")

        for step in preview.steps:
            _apply(page, step)
        if motion is not None:
            _apply(page, ("select", "motion-select", motion))

        hide = page.get_by_role("button", name="Hide controls")
        if hide.count():
            hide.click()
        # The panel leaves a reopen affordance over the canvas; nothing else is a button.
        page.add_style_tag(content="button{display:none!important}")
        _orbit(page, preview.orbit)
        page.wait_for_timeout(preview.settle * 1000)

        if shot is not None:
            page.screenshot(path=str(shot))
            browser.close()
            return start_index

        client = context.new_cdp_session(page)
        frames: list[tuple[float, str]] = []

        def on_frame(payload: dict) -> None:
            frames.append((payload["metadata"]["timestamp"], payload["data"]))
            client.send("Page.screencastFrameAck", {"sessionId": payload["sessionId"]})

        client.on("Page.screencastFrame", on_frame)
        before = page.evaluate("() => window.__paced + window.__yield / 2")
        client.send(
            "Page.startScreencast",
            {
                "format": "png",
                "maxWidth": WIDTH * CAPTURE_SCALE,
                "maxHeight": HEIGHT * CAPTURE_SCALE,
                "everyNthFrame": 1,
            },
        )
        page.wait_for_timeout(seconds * 1000)
        client.send("Page.stopScreencast")
        after = page.evaluate("() => window.__paced + window.__yield / 2")
        browser.close()

    if len(frames) < 2:
        raise SystemExit("the screencast delivered no frames")
    rate = (after - before) / seconds
    slow = (
        ""
        if rate >= expected_rate * MIN_STEP_RATE_RATIO
        else (f"  ** slow motion: {expected_rate:.0f}/s expected **")
    )
    label = f" [{motion}]" if motion else ""
    print(
        f"  {len(frames)} frames captured{label}, "
        f"{rate:.1f} of {expected_rate:.0f} steps/s{slow}"
    )

    # The frame nearest each 1/FPS tick: even spacing off real presentation times.
    assert frames_dir is not None
    start, span = frames[0][0], frames[-1][0] - frames[0][0]
    frames_dir.mkdir(parents=True, exist_ok=True)
    index = start_index
    while (index - start_index) / FPS <= span:
        want = start + (index - start_index) / FPS
        _, data = min(frames, key=lambda frame: abs(frame[0] - want))
        (frames_dir / f"seq_{index:04d}.png").write_bytes(base64.b64decode(data))
        index += 1
    return index


def to_gif(frames_dir: Path, out: Path, crop: str) -> None:
    """Crop, downscale and quantize the sequence into a looping 15 fps GIF.

    32 colours without dithering: the scene is mostly one blue and one white, and
    dithering noise costs a megabyte without showing at the 200 px the README renders.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        f"crop={crop},scale={WIDTH}:{HEIGHT}:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=32:stats_mode=full[p];"
        "[b][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "seq_%04d.png"),
            "-vf",
            filters,
            "-loop",
            "0",
            "-y",
            str(out),
        ],
        check=True,
    )
    print(f"  {out.relative_to(ROOT)}  {out.stat().st_size / 1e6:.1f} MB")


def crop_png(path: Path, crop: str) -> None:
    """Apply a recipe's crop to a framing shot, so `--shot` shows the real framing."""
    cropped = path.with_suffix(".crop.png")  # ffmpeg cannot read and write one file
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"crop={crop},scale={WIDTH}:{HEIGHT}",
            "-frames:v",
            "1",
            "-y",
            str(cropped),
        ],
        check=True,
    )
    cropped.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="*", help=f"task ids: {', '.join(ALL_TASKS)}")
    parser.add_argument(
        "--all", action="store_true", help="record every registered task"
    )
    parser.add_argument(
        "--dist", type=Path, default=ROOT / "dist", help="where built apps live"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "assets", help="where GIFs go"
    )
    parser.add_argument(
        "--seconds", type=float, help="override the recipe's clip length"
    )
    parser.add_argument(
        "--orbit", type=float, help="override the recipe's camera swing"
    )
    parser.add_argument("--crop", help="override the recipe's crop, w:h:x:y of 960x702")
    parser.add_argument(
        "--motion", help="film this motion alone, instead of the recipe's `motions`"
    )
    parser.add_argument(
        "--keep-frames", type=Path, help="also keep the PNG sequence here"
    )
    parser.add_argument(
        "--shot",
        type=Path,
        help="write the framing to this PNG instead of filming a GIF",
    )
    args = parser.parse_args()

    tasks = ALL_TASKS if args.all else tuple(args.tasks)
    if not tasks:
        parser.error("name at least one task, or pass --all")
    unknown = [task for task in tasks if task not in ALL_TASKS]
    if unknown:
        parser.error(
            f"unknown task(s) {', '.join(unknown)}; have {', '.join(ALL_TASKS)}"
        )
    if args.shot and len(tasks) > 1:
        parser.error("--shot takes one task at a time")
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is not on PATH")

    for task_id in tasks:
        overrides = {
            key: value
            for key, value in (
                ("seconds", args.seconds),
                ("orbit", args.orbit),
                ("crop", args.crop),
                ("motions", (args.motion,) if args.motion else None),
            )
            if value is not None
        }
        preview = dataclasses.replace(PREVIEWS.get(task_id, Preview()), **overrides)

        app_dir = ensure_built(task_id, args.dist)
        server = serve(app_dir)
        query = f"?{preview.query}" if preview.query else ""
        url = f"http://127.0.0.1:{server.server_address[1]}/{query}"
        try:
            if args.shot:
                print(
                    f"[{task_id}] framing {url} (orbit {preview.orbit:g}, crop {preview.crop})"
                )
                film(
                    url,
                    preview,
                    shot=args.shot,
                    motion=preview.motions[0] if preview.motions else None,
                )
                crop_png(args.shot, preview.crop)
                print(f"  {args.shot}")
                continue
            print(f"[{task_id}] filming {preview.seconds:g}s of {url}")
            with tempfile.TemporaryDirectory() as tmp:
                frames_dir = (
                    args.keep_frames / task_id if args.keep_frames else Path(tmp)
                )
                # One segment per motion, appended into one sequence; a task without
                # `motions` is a single segment of whatever the setup left running.
                segments = preview.motions or (None,)
                index = 0
                for motion in segments:
                    index = film(
                        url,
                        preview,
                        frames_dir=frames_dir,
                        expected_rate=control_rate(app_dir),
                        motion=motion,
                        seconds=preview.seconds / len(segments),
                        start_index=index,
                    )
                to_gif(frames_dir, args.out_dir / f"{task_id}.gif", preview.crop)
        finally:
            server.shutdown()


if __name__ == "__main__":
    main()
