"""Record a task's README preview GIF straight out of its built browser app.

    uv run --group previews playwright install chromium   # once
    uv run --group previews python scripts/record_preview.py husky
    uv run --group previews python scripts/record_preview.py --all

Builds the task if `dist/<task-id>` is missing, serves it with the COOP/COEP headers
MuJoCo WASM needs, drives the app's own control panel to set the shot up, then films the
canvas into `assets/<task-id>.gif` at 480x351 / 15 fps: the size, aspect ratio and frame
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
same widgets a visitor would touch), `orbit` swings the camera, `crop` frames it. A tuple
of them, each opening its own `scene`, cuts those scenes into one GIF. Dial a new one in
with `--shot`, which stops after the setup and writes the framing as a PNG (`--scene`
picks the cut):

    uv run --group previews python scripts/record_preview.py husky --shot /tmp/f.png \\
        --orbit 150 --crop 960:702:0:0    # the whole frame, to find the crop from

Without a GPU, `--software` renders through Mesa under Xvfb and slows the page's clock
so the renderer keeps up; `--chromium` launches an installed browser when Playwright's
own build is missing:

    xvfb-run -a uv run --group previews python scripts/record_preview.py husky \\
        --software --chromium /opt/pw-browsers/chromium

Every recording also checks the task works: the app runs on for `--check-seconds` after
the clip, and the recording fails on a termination other than `time_out`, a root tipped
past `MAX_TILT_DEG`, NaN state, observations or actions, an observation input that stays
all zeros, a traced graph that stops running, actions that never change, a stalled step
loop, a page error, or a slow-motion or choppy clip. It writes `dist/preview/<task-id>.json`
and a contact sheet, `dist/preview/<task-id>.png`: every frame of the GIF, then one every
`SHEET_EVERY` seconds of the run after it, failures outlined in red. A failure exits 1
once every task is filmed.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import functools
import http.server
import io
import json
import re
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from mjswan_playground.registry import ALL_TASKS

ROOT = Path(__file__).resolve().parent.parent

#: The reference previews' geometry (mjlab_playground): 480x351 at 15 fps.
WIDTH, HEIGHT, FPS = 480, 351, 15
#: Filmed at 2x and downscaled into the GIF, so a crop still has real pixels behind it.
CAPTURE_SCALE = 2
#: Fraction of the policy's own control rate a recording has to hold to count as real time.
MIN_STEP_RATE_RATIO = 0.9
#: `--software` runs the page's clock at this fraction of real time, so a renderer that
#: draws a few frames a second still lands one every few control steps of page time.
SOFTWARE_CLOCK = 0.15
#: Seconds the app runs on after the clip, checked like the clip: a 4 s GIF misses a fall
#: that comes later.
CHECK_SECONDS = 20.0
#: A root tipped this far from upright has fallen, whatever the task's terminations say.
MAX_TILT_DEG = 75.0
#: Share of control steps a traced observation, command or termination graph has to run
#: on. Each runs every step; an input the browser cannot fill stops it, silently.
MIN_GRAPH_RUN_RATIO = 0.9
#: The run after the clip goes on the contact sheet one frame per this many seconds.
SHEET_EVERY = 2.0


@dataclass(frozen=True)
class Preview:
    """How one task's preview is filmed.

    Attributes:
        scene: Scene id to open, the app's ``?scene=``; empty opens the first.
        steps: Control-panel actions, applied before the panel is hidden. Each is one of
            ``("select", <input id>, <option>)`` for the scene/policy/motion pickers,
            ``("number", <slider label>, <value>)``, ``("checkbox", <label>, <bool>)``,
            ``("click", <button label>)`` or ``("wait", <seconds>)``.
        motions: Motions to film one after another, concatenated into one GIF with
            `seconds` split evenly between them: for a tracking policy, where the point
            is the range of motions rather than any single one. Empty films one segment
            of whatever `steps` left running.
        query: Extra URL query, e.g. ``"ref=0"`` to drop the motion-tracking ghost.
        from_reset: Press `r` just before rolling, so the clip runs from one reset. With
            `seconds` at the episode length the GIF then loops on the reset rather than
            cutting across it mid-clip.
        tilt: Degrees to drop the camera towards the horizon, same drag. Positive lowers
            it; the authored cameras look down from well above head height.
        orbit: Degrees to swing the camera, as a drag on empty background would. Positive
            adds to the scene's authored `azimuth` (the camera travels anticlockwise seen
            from above). Body tracking keeps whatever angle the drag leaves.
        crop: ffmpeg crop of the 960x702 capture, ``w:h:x:y``, at the 480:351 aspect;
            centres the robot and drops the empty sky above it.
        seconds: Clip length. 4 s is 60 frames at 15 fps.
        settle: Seconds between finishing the setup and rolling, so camera damping and any
            reset from a policy switch are done before the first frame. Without
            `from_reset`, also about how far into the scene's run the clip starts.
        upright: Fail when the root tips past `MAX_TILT_DEG`. Off only for a clip that
            leaves upright on purpose, a flip.
        ok_terminations: Terminations besides `time_out` the task itself plays out, as a
            dodgeball hit ends an episode.
    """

    scene: str = ""
    steps: tuple[tuple, ...] = ()
    motions: tuple[str, ...] = ()
    query: str = ""
    orbit: float = 0.0
    tilt: float = 0.0
    from_reset: bool = False
    crop: str = "719:526:120:120"
    seconds: float = 4.0
    settle: float = 2.0
    upright: bool = True
    ok_terminations: tuple[str, ...] = ()


#: One entry per task in `ALL_TASKS`; a task with no entry is filmed with the defaults.
#: Every `orbit` here swings the authored camera round to the robot's front. Which swing
#: that is, is worth measuring rather than deriving: the scene's authored azimuth is in its
#: `manifest.json`, but whether a robot spawns facing +x or -x is the task's business.
#: `--shot` settles it in one run.
PREVIEWS: dict[str, Preview | tuple[Preview, ...]] = {
    # Push Speed defaults to 1.0, so the skater is already riding.
    "husky": Preview(orbit=150, crop="719:526:121:127"),
    # `idle_02` opens the motion list and stands still, so the preview picks its own two:
    # one policy over a sprint and a flip is the point of a tracking task. `ref=0` drops
    # the tracking ghost, which otherwise stands beside the robot and halves its size.
    # The flip turns the root over, so the tracking terminations judge a fall instead.
    "wbc": Preview(
        motions=("sprint_01", "flip_02"),
        query="ref=0",
        orbit=150,
        crop="719:526:115:122",
        seconds=6.0,
        upright=False,
    ),
    # Auto throw is armed by default: a ball every 1-4 s, so a 5 s clip catches two.
    # A wider crop than the rest, to keep the incoming ball in frame. A hit ends the
    # episode, as in mjlab: the policy dodges most throws, not every one.
    "pacman": Preview(
        orbit=90, crop="840:614:60:76", seconds=5.0, ok_terminations=("ball_hit",)
    ),
    "microduck": Preview(
        steps=(("number", "Forward (m/s)", 0.35), ("wait", 1)),
        orbit=140,
        crop="719:526:121:68",
    ),
    # One cut per experiment, framed roughly as upstream's own videos are: eye level and
    # close on the walkers, side-on to the ladder and the jumps, into the slot.
    "microduckpg": (
        Preview(
            scene="running", orbit=140, tilt=10, crop="600:439:180:96", seconds=2.5
        ),
        # At full span, which it reaches at about 22 s.
        Preview(
            scene="swing", settle=24, crop="672:491:144:70", upright=False, seconds=3.0
        ),
        Preview(
            scene="basketball", orbit=100, tilt=8, crop="540:395:210:200", seconds=2.0
        ),
        Preview(
            scene="stilts_50_cm", orbit=110, tilt=6, crop="860:629:50:20", seconds=2.5
        ),
        # The seeded first attempt over the top of the ladder, which it tops at about 8 s.
        Preview(
            scene="desk_climb",
            orbit=45,
            crop="720:526:120:40",
            settle=5.6,
            upright=False,
            seconds=3.0,
        ),
        # The platform coming into view at about 17 s.
        Preview(
            scene="chimney_climb",
            tilt=15,
            crop="576:421:192:105",
            settle=16.1,
            upright=False,
            seconds=3.0,
        ),
        Preview(
            scene="long_jump", crop="624:456:170:120", from_reset=True, seconds=2.0
        ),
        # A still camera, wide enough for the platform's top and the mat.
        Preview(
            scene="backflip",
            steps=(("checkbox", "Track camera", False), ("number", "FOV (°)", 70)),
            orbit=-55,
            tilt=-15,
            crop="575:420:192:262",
            from_reset=True,
            upright=False,
            seconds=2.0,
        ),
    ),
    # Lowered camera: from the authored view a standing body is foreshortened into
    # mostly floor. 4.84 s is the 484-frame clip, so the GIF is exactly one episode.
    "musclemimic": Preview(
        orbit=270,
        tilt=30,
        crop="480:351:231:189",
        from_reset=True,
        seconds=4.84,
    ),
    # 4.64 s is the 232-frame clip, so the GIF is the whole kick from one reset.
    "spinkick": Preview(
        query="ref=0",
        orbit=150,
        crop="719:526:115:122",
        from_reset=True,
        seconds=4.64,
    ),
    # Wider crop for the taller H1-2. No steps: the twist command resamples every 3-8 s.
    "bipedhrl": Preview(orbit=-30, crop="878:642:41:35"),
    # No steps: the twist and the height command resample every 3-8 s, so the clip
    # catches a squat or a walk without driving the panel.
    "duet": Preview(orbit=-30, crop="766:560:97:90", seconds=6.0),
    # Tight crop: the robot is 0.4 m tall under a camera authored at 3 m.
    "upkie": Preview(orbit=45, crop="540:395:210:170", seconds=5.0),
    # No steps: the twist and the posture command resample every 3-8 s.
    "jumper": Preview(orbit=45, tilt=-20, crop="878:642:41:35", seconds=5.0),
    # duet's G1 framing. No steps: the twist command resamples every 3-8 s.
    "unitreerl": Preview(orbit=-30, crop="766:560:97:90", seconds=6.0),
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
        # A process per task, as `msp site` does: pacman, bipedhrl and duet each import
        # a top-level `src` of their own.
        command = [sys.executable, "-m", "mjswan_playground", "build", task_id]
        command += ["--output-dir", str(app_dir.resolve())]
        if subprocess.run(command).returncode:
            raise RuntimeError(f"msp build {task_id} failed")
    return app_dir


def per_step_graphs(app_dir: Path) -> set[str]:
    """The traced graphs the app runs every control step: observation and termination
    graphs, and a traced command's own. Event and reset graphs run on a reset only."""
    manifest = json.loads((app_dir / "manifest.json").read_text())
    paths: set[str] = set()
    for project in manifest["projects"]:
        for scene in project["scenes"]:
            for mdp in scene.get("mdps", []):
                entries = list((mdp.get("terminations") or {}).values())
                entries += list((mdp.get("commands") or {}).values())
                for group in (mdp.get("observations") or {}).values():
                    entries += group if isinstance(group, list) else [group]
                paths |= {e.get(k) for e in entries for k in ("fused", "onnx")}
    return paths - {None}


def control_rate(app_dir: Path) -> float:
    """Control steps per second the app's first scene runs at, from its manifest."""
    manifest = json.loads((app_dir / "manifest.json").read_text())
    control_dt = manifest["projects"][0]["scenes"][0].get("control_dt")
    return 1 / control_dt if control_dt else 50.0


# ── the browser side ──────────────────────────────────────────────────────────


def _labelled_input(page, label: str, selector: str) -> str:
    """CSS selector, by id, of the `selector` input on the panel row labelled `label`.

    Mantine generates those ids, so they are read back off the DOM (and assigned when the
    widget has none) rather than guessed. An attribute selector, as the viewer controls'
    ids (`viewer:Scene/Camera/Track camera`) are no valid `#id`.
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
    return f'[id="{element_id}"]'


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


def _orbit(page, degrees: float, tilt: float = 0.0) -> None:
    """Swing and drop the camera by dragging empty background, as a visitor would.

    OrbitControls turns a full circle per viewport height of drag, horizontally for the
    azimuth and vertically for the elevation, and body tracking only carries the orbit
    target around, so both survive the whole clip. The drag starts in a corner on purpose:
    a pointerdown that lands on a body pulls the robot instead (`dragStateManager` takes
    the drag and OrbitControls sits out).
    """
    if not degrees and not tilt:
        return
    dx = degrees / 360 * HEIGHT
    dy = -tilt / 360 * HEIGHT  # dragging up swings the camera down
    x = 24 if dx >= 0 else WIDTH - 24
    y = 24 if dy >= 0 else HEIGHT - 24
    page.mouse.move(x, y)
    page.mouse.down()
    for i in range(1, 21):  # in steps, so the controls integrate it as a real drag
        page.mouse.move(x + dx * i / 20, y + dy * i / 20)
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

#: Runs the page's clock at a fraction of real time for `--software`: timers,
#: `performance.now`, `Date.now` and animation-frame timestamps alike. Installed before
#: `_STEP_COUNTER`, which then still sees the step loop's own 20 ms delays.
_SCALED_CLOCK = """((k) => {
    const now = performance.now.bind(performance), t0 = now();
    performance.now = () => t0 + (now() - t0) * k;
    const dateNow = Date.now, d0 = dateNow();
    Date.now = () => d0 + (dateNow() - d0) * k;
    const timeout = window.setTimeout;
    window.setTimeout = function (fn, ms, ...rest) {
        return timeout.call(this, fn, (ms || 0) / k, ...rest);
    };
    const raf = window.requestAnimationFrame;
    window.requestAnimationFrame = function (cb) {
        return raf.call(this, (ts) => cb(t0 + (ts - t0) * k));
    };
})(%r);"""

#: The app exposes no handle on its runtime, so the bundle is patched in flight to hand
#: `_TELEMETRY` the runtime from the top of its step loop.
_HOOK_AT = "async runLoop(){"
_HOOK = _HOOK_AT + "window.__recHook&&window.__recHook(this);"

#: Records, never changes, what the app does: per control step whether state,
#: observations and actions are finite, which observation inputs are all zeros, whether
#: the actions moved, and the root's height and tilt; every termination; every run of
#: each traced graph. The root is the free joint carrying the most bodies, the robot's
#: rather than a ball's.
_TELEMETRY = """(() => {
    const rec = (window.__rec = { hooked: false, steps: [], terms: [], runs: {} });
    const finite = (a) => a.every(Number.isFinite);
    const rootOf = (m) => {
        const size = new Int32Array(m.nbody).fill(1);
        for (let b = m.nbody - 1; b > 0; b--) size[m.body_parentid[b]] += size[b];
        let adr = null, most = 0;
        for (let j = 0; j < m.njnt; j++) {
            const n = size[m.jnt_bodyid[j]];
            if (m.jnt_type[j] === 0 && n > most) [adr, most] = [m.jnt_qposadr[j], n];
        }
        return adr;
    };
    window.__recHook = (rt) => {
        if (rt.__rec) return;
        rt.__rec = rec.hooked = true;
        rec.paths = () => [...(rt.policyGraphs?.sessions?.keys() ?? [])];
        let obs = null, model = null, root = null, last = null;
        // Loading a policy builds new sessions and a new termination manager.
        const watch = () => {
            for (const [path, session] of rt.policyGraphs?.sessions ?? []) {
                if (session.__rec) continue;
                session.__rec = true;
                rec.runs[path] ??= 0;
                const run = session.run.bind(session);
                session.run = (...a) => (rec.runs[path]++, run(...a));
            }
            const tm = rt.terminationManager;
            if (tm && !tm.__rec) {
                tm.__rec = true;
                const evaluate = tm.evaluate.bind(tm);
                tm.evaluate = (...a) => {
                    const k = rec.steps.length - 1;
                    const note = (r) => {
                        if (r.done) rec.terms.push({ k, reasons: r.reasons });
                        return r;
                    };
                    // Async in mjswan 0.11.5: `done` read off the promise is undefined.
                    const r = evaluate(...a);
                    return r instanceof Promise ? r.then(note) : note(r);
                };
            }
        };
        const infer = rt.runOnnxInference.bind(rt);
        rt.runOnnxInference = (o) => (watch(), (obs = o), infer(o));
        const step = rt.executeSimulationSteps.bind(rt);
        rt.executeSimulationSteps = () => {
            step();
            const m = rt.mjModel, d = rt.mjData;
            if (!m || !d) return;
            if (m !== model) [model, root] = [m, rootOf(m)];
            const s = { nan: null, zero: [], same: null };
            if (!finite(d.qpos) || !finite(d.qvel)) s.nan = "state";
            for (const [key, v] of Object.entries(obs ?? {})) {
                if (v.every((x) => x === 0)) s.zero.push(key);
                if (!finite(v)) s.nan ??= "observation " + key;
            }
            const act = rt.policyRunner?.getLastActions();
            if (act) {
                if (!finite(act)) s.nan ??= "actions";
                s.same = !!last && last.length === act.length && act.every((x, i) => x === last[i]);
                last = Float32Array.from(act);
            }
            if (root !== null) {
                const q = d.qpos, x = q[root + 4], y = q[root + 5];
                s.z = q[root + 2];
                s.tilt = (Math.acos(Math.max(-1, Math.min(1, 1 - 2 * (x * x + y * y)))) * 180) / Math.PI;
            }
            rec.steps.push(s);
        };
    };
})();"""


@dataclass
class Segment:
    """What one `film` call saw: where its frames end, and what the app did from the
    first filmed frame on, the clip's `filmed` steps first and then the run after it."""

    next_index: int
    #: The recipe it was filmed with.
    preview: Preview = Preview()
    motion: str | None = None
    first_index: int = 0
    seconds: float = 0.0
    filmed: int = 0
    steps: list[dict] = dataclasses.field(default_factory=list)
    terms: list[dict] = dataclasses.field(default_factory=list)
    runs: dict[str, int] = dataclasses.field(default_factory=dict)
    errors: list[str] = dataclasses.field(default_factory=list)
    rate: float = 0.0
    fps: float = 0.0
    #: Screenshots of the run after the clip, one per `SHEET_EVERY` seconds.
    after: list[bytes] = dataclasses.field(default_factory=list)

    @property
    def label(self) -> str:
        """`` [<motion or scene>]``, whichever picked it, for the printed output."""
        name = self.motion or self.preview.scene
        return f" [{name}]" if name else ""


def _hook_runtime(route, hooked: list[bool]) -> None:
    """Serve a bundle file with `_HOOK` inserted when it holds the step loop."""
    response = route.fetch()
    body = response.text()
    if _HOOK_AT not in body:
        route.fulfill(response=response)
        return
    hooked.append(True)
    route.fulfill(response=response, body=body.replace(_HOOK_AT, _HOOK, 1))


def film(
    url: str,
    preview: Preview,
    frames_dir: Path | None = None,
    shot: Path | None = None,
    expected_rate: float = 50.0,
    motion: str | None = None,
    seconds: float | None = None,
    start_index: int = 0,
    clock: float = 1.0,
    chromium: str | None = None,
    check_seconds: float | None = None,
) -> Segment:
    """Set the shot up, then write either a framing PNG (`shot`) or a PNG sequence.

    One call is one segment: `motion` picks it in the panel, `seconds` overrides the
    recipe's length and `start_index` continues an earlier segment's numbering, so a
    multi-motion preview concatenates into a single sequence. `clock` below 1 is
    `--software`: every wait and the frame pick run on page time. `check_seconds` records
    the app from the first frame on and keeps it running that long after the clip;
    `None` records nothing.
    """
    from playwright.sync_api import sync_playwright

    seconds = preview.seconds if seconds is None else seconds
    segment = Segment(
        next_index=start_index, preview=preview, motion=motion, first_index=start_index
    )
    checked = check_seconds is not None

    with sync_playwright() as pw:
        # Headed (see the module docstring), and at 2x even on a 1x display.
        args = ["--hide-scrollbars", f"--force-device-scale-factor={CAPTURE_SCALE}"]
        if clock < 1:
            # Mesa's llvmpipe leaves the step loop on time, where SwiftShader stalls it.
            args += ["--use-angle=gl", "--ignore-gpu-blocklist"]
        browser = pw.chromium.launch(
            headless=False, args=args, executable_path=chromium
        )
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=CAPTURE_SCALE,
        )
        if clock < 1:
            context.add_init_script(_SCALED_CLOCK % clock)
        context.add_init_script(_STEP_COUNTER)
        hooked: list[bool] = []
        if checked:
            context.add_init_script(_TELEMETRY)
            context.route(
                re.compile(r"/assets/[^/]+\.js$"),
                functools.partial(_hook_runtime, hooked=hooked),
            )
        page = context.new_page()

        def on_error(error) -> None:
            print(f"  [pageerror] {error}", file=sys.stderr)
            segment.errors.append(str(error))

        page.on("pageerror", on_error)
        page.goto(url, wait_until="load")
        page.wait_for_function(
            "() => window.__mjswanReady || window.__mjswanError", timeout=240_000
        )
        if page.evaluate("() => window.__mjswanError"):
            raise SystemExit(f"{url} failed to load its scene")
        if checked:
            if not hooked:
                raise SystemExit(
                    f"no {_HOOK_AT!r} in this build's bundle, so the run cannot be "
                    "checked: update `_HOOK_AT` for this mjswan, or pass --no-check"
                )
            page.wait_for_function("() => window.__rec.hooked", timeout=60_000)

        for step in preview.steps:
            _apply(page, step)
        if motion is not None:
            _apply(page, ("select", "motion-select", motion))

        hide = page.get_by_role("button", name="Hide controls")
        if hide.count():
            hide.click()
        # The panel leaves a reopen affordance over the canvas; nothing else is a button.
        page.add_style_tag(content="button{display:none!important}")
        _orbit(page, preview.orbit, preview.tilt)
        page.wait_for_timeout(preview.settle / clock * 1000)

        if shot is not None:
            page.screenshot(path=str(shot))
            browser.close()
            return segment

        if preview.from_reset:
            # The panel's own shortcut, on a window listener, so it still works hidden.
            # `engine.reset()` leaves the camera alone, so the framing survives.
            page.keyboard.press("r")
            # Two control steps to land and two drawn frames to show, or the first
            # frame is the pose before it.
            page.wait_for_timeout(2 / expected_rate / clock * 1000)
            page.evaluate(
                "() => new Promise((drawn) =>"
                " requestAnimationFrame(() => requestAnimationFrame(drawn)))"
            )

        client = context.new_cdp_session(page)
        frames: list[tuple[float, str]] = []

        def on_frame(payload: dict) -> None:
            frames.append((payload["metadata"]["timestamp"], payload["data"]))
            client.send("Page.screencastFrameAck", {"sessionId": payload["sessionId"]})

        client.on("Page.screencastFrame", on_frame)
        if checked:
            first = page.evaluate("() => window.__rec.steps.length")
            runs = page.evaluate("() => ({...window.__rec.runs})")
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
        page.wait_for_timeout(seconds / clock * 1000)
        client.send("Page.stopScreencast")
        after = page.evaluate("() => window.__paced + window.__yield / 2")
        if checked:
            segment.filmed = page.evaluate("() => window.__rec.steps.length") - first
            for _ in range(round(check_seconds / SHEET_EVERY)):
                page.wait_for_timeout(SHEET_EVERY / clock * 1000)
                segment.after.append(page.screenshot())
            segment.seconds = seconds + check_seconds
            seen = page.evaluate(
                """(first) => ({
                    steps: window.__rec.steps.slice(first),
                    terms: window.__rec.terms
                        .filter((t) => t.k >= first)
                        .map((t) => ({ ...t, k: t.k - first })),
                    runs: window.__rec.runs,
                    paths: window.__rec.paths(),
                })""",
                first,
            )
            segment.steps, segment.terms = seen["steps"], seen["terms"]
            # Only the graphs of the policy still loaded: a setup step may switch it.
            segment.runs = {
                path: seen["runs"].get(path, 0) - runs.get(path, 0)
                for path in seen["paths"]
            }
        browser.close()

    if len(frames) < 2:
        raise SystemExit("the screencast delivered no frames")
    segment.rate = (after - before) / seconds
    segment.fps = len(frames) / seconds
    slow = (
        ""
        if segment.rate >= expected_rate * MIN_STEP_RATE_RATIO
        else (f"  ** slow motion: {expected_rate:.0f}/s expected **")
    )
    if segment.fps < FPS:
        slow += f"  ** fewer than {FPS} frames per second: frames repeat **"
    print(
        f"  {len(frames)} frames captured{segment.label}, "
        f"{segment.rate:.1f} of {expected_rate:.0f} steps/s{slow}"
    )

    # The frame nearest each 1/FPS tick of page time (`clock` times real time).
    assert frames_dir is not None
    start, span = frames[0][0], (frames[-1][0] - frames[0][0]) * clock
    frames_dir.mkdir(parents=True, exist_ok=True)
    index = start_index
    while (index - start_index) / FPS <= span:
        want = start + (index - start_index) / FPS / clock
        _, data = min(frames, key=lambda frame: abs(frame[0] - want))
        (frames_dir / f"seq_{index:04d}.png").write_bytes(base64.b64decode(data))
        index += 1
    segment.next_index = index
    return segment


def check_run(
    segment: Segment,
    preview: Preview,
    control_dt: float,
    expected_rate: float,
    per_step: set[str],
) -> list[tuple[list[int], str]]:
    """What went wrong in `segment`: each failure with the steps it happens at, counted
    from the first filmed frame, or none when it spans the run. Empty when it passed.

    `per_step` names the graphs that have to run every step (`per_step_graphs`).
    """
    failures: list[tuple[list[int], str]] = []
    steps = segment.steps

    def at(k: int) -> str:
        after = k >= segment.filmed
        return f"{k * control_dt:.1f} s" + (" (after the clip)" if after else "")

    failures += [([], f"page error: {error}") for error in segment.errors]
    if segment.rate < expected_rate * MIN_STEP_RATE_RATIO:
        failures.append(
            ([], f"slow motion: {segment.rate:.1f} of {expected_rate:.0f} steps/s")
        )
    if segment.fps < FPS:
        failures.append(
            ([], f"{segment.fps:.1f} frames/s captured, under {FPS}: frames repeat")
        )
    if not steps:
        return failures + [([], "no control step ran while filming")]

    want = round(segment.seconds / control_dt)
    if len(steps) < want * MIN_STEP_RATE_RATIO:
        failures.append(
            (
                [],
                f"the step loop ran {len(steps)} of {want} steps: it stopped or stalled",
            )
        )
    nan = next((k for k, s in enumerate(steps) if s["nan"]), None)
    if nan is not None:
        failures.append(([nan], f"{steps[nan]['nan']} went NaN at {at(nan)}"))

    ok = {"time_out", *preview.ok_terminations}
    bad = [t["k"] for t in segment.terms if not set(t["reasons"]) <= ok]
    if bad:
        reasons = {r for t in segment.terms for r in t["reasons"]} - ok
        failures.append(
            (
                bad,
                f"terminated {len(bad)}x ({', '.join(sorted(reasons))}), first at {at(bad[0])}",
            )
        )
    tilts = [s.get("tilt", 0.0) for s in steps]
    tipped = [
        k
        for k, tilt in enumerate(tilts)
        if tilt > MAX_TILT_DEG and (k == 0 or tilts[k - 1] <= MAX_TILT_DEG)
    ]
    if preview.upright and tipped:
        failures.append(
            (
                tipped,
                f"root tipped past {MAX_TILT_DEG:g}° {len(tipped)}x, first at "
                f"{at(tipped[0])}, to {max(tilts):.0f}°",
            )
        )

    zeros: dict[str, int] = {}
    for s in steps:
        for key in s["zero"]:
            zeros[key] = zeros.get(key, 0) + 1
    failures += [
        ([], f"observation {key!r} all zeros on every step")
        for key, n in zeros.items()
        if n == len(steps)
    ]
    if len(steps) > 1 and all(s["same"] for s in steps[1:]):
        failures.append(([], f"actions never changed over {len(steps)} steps"))
    for path, n in sorted(segment.runs.items()):
        if path in per_step and n < len(steps) * MIN_GRAPH_RUN_RATIO:
            failures.append(([], f"{path} ran on {n} of {len(steps)} steps"))
    return failures


def _ffmpeg() -> str:
    """`ffmpeg` on PATH, else `imageio-ffmpeg`'s, which mjlab depends on."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def to_gif(frames_dir: Path, out: Path, segments: list[Segment]) -> None:
    """Crop each segment by its own recipe, downscale, join and quantize the sequence
    into a looping 15 fps GIF.

    32 colours without dithering: the scene is mostly one blue and one white, and
    dithering noise costs a megabyte without showing at the 200 px the README renders.
    Cuts of several scenes get 64, or the palette shifts their colours. One palette, not
    one per scene: ffmpeg encodes a frame off the first palette whole, not as a diff.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    colors = 32 if len({s.preview.scene for s in segments}) == 1 else 64
    inputs, filters = [], ""
    for i, s in enumerate(segments):
        inputs += ["-framerate", str(FPS), "-start_number", str(s.first_index)]
        inputs += ["-i", str(frames_dir / "seq_%04d.png")]
        filters += (
            f"[{i}:v]trim=end_frame={s.next_index - s.first_index},crop={s.preview.crop},"
            f"scale={WIDTH}:{HEIGHT}:flags=lanczos,setsar=1[v{i}];"
        )
    filters += (
        "".join(f"[v{i}]" for i in range(len(segments)))
        + f"concat=n={len(segments)},split[a][b];"
        f"[a]palettegen=max_colors={colors}:stats_mode=full[p];"
        "[b][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    subprocess.run(
        [_ffmpeg(), "-v", "error", *inputs, "-filter_complex", filters]
        + ["-loop", "0", "-y", str(out)],
        check=True,
    )
    print(f"  {_shown(out)}  {out.stat().st_size / 1e6:.1f} MB")


def crop_png(path: Path, crop: str) -> None:
    """Apply a recipe's crop to a framing shot, so `--shot` shows the real framing."""
    cropped = path.with_suffix(".crop.png")  # ffmpeg cannot read and write one file
    subprocess.run(
        [
            _ffmpeg(),
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


def contact_sheet(
    gif: Path, segments: list[Segment], marks: list[set[int]], out: Path
) -> None:
    """Tile every frame of `gif`, then each segment's run after the clip, into `out`.

    `marks` holds, per segment, the indices into that segment's frames, clip then after,
    where a failure starts; those are outlined in red.
    """
    from PIL import Image, ImageDraw, ImageSequence

    w, h, columns, header = WIDTH // 3, HEIGHT // 3, 10, 18
    rows: list = []

    def add(title: str, tiles: list) -> None:
        rows.append(title)
        rows.extend(tiles[i : i + columns] for i in range(0, len(tiles), columns))

    with Image.open(gif) as image:
        clip = [f.convert("RGB").resize((w, h)) for f in ImageSequence.Iterator(image)]
    for segment, marked in zip(segments, marks):
        label = segment.label
        frames = range(segment.first_index, min(segment.next_index, len(clip)))
        add(
            f"clip{label}",
            [(clip[i], f"{i / FPS:.2f}s", i in marked) for i in frames],
        )
        if segment.after:
            cw, ch, cx, cy = map(int, segment.preview.crop.split(":"))
            shots = [
                Image.open(io.BytesIO(png))
                .convert("RGB")
                .crop((cx, cy, cx + cw, cy + ch))
                .resize((w, h))
                for png in segment.after
            ]
            offset = segment.next_index
            add(
                f"after the clip{label}, every {SHEET_EVERY:g} s",
                [
                    (shot, f"+{(j + 1) * SHEET_EVERY:g}s", offset + j in marked)
                    for j, shot in enumerate(shots)
                ],
            )

    sheet = Image.new(
        "RGB",
        (columns * w, sum(header if isinstance(r, str) else h for r in rows)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    y = 0
    for row in rows:
        if isinstance(row, str):
            draw.text((4, y + 3), row, fill="black")
            y += header
            continue
        for x, (tile, label, marked) in enumerate(row):
            sheet.paste(tile, (x * w, y))
            draw.text(
                (x * w + 3, y + 2),
                label,
                fill="white",
                stroke_width=1,
                stroke_fill="black",
            )
            if marked:
                draw.rectangle(
                    (x * w, y, (x + 1) * w - 1, y + h - 1), outline="red", width=4
                )
        y += h
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def report(
    task_id: str,
    segments: list[Segment],
    control_dt: float,
    expected_rate: float,
    per_step: set[str],
    gif: Path,
    out_dir: Path,
) -> list[str]:
    """Check each segment, write `<task-id>.json` and the contact sheet into `out_dir`,
    and return the failures."""
    failures: list[str] = []
    marks: list[set[int]] = []
    for segment in segments:
        found = check_run(segment, segment.preview, control_dt, expected_rate, per_step)
        failures += [f"{segment.label.strip()} {m}".lstrip() for _, m in found]
        marked = set()
        for k in (k for ks, _ in found for k in ks):
            if k < segment.filmed:
                frame = segment.first_index + round(k * control_dt * FPS)
                marked.add(min(frame, segment.next_index - 1))
            elif segment.after:
                j = int((k - segment.filmed) * control_dt / SHEET_EVERY)
                marked.add(segment.next_index + min(j, len(segment.after) - 1))
        marks.append(marked)

    sheet = out_dir / f"{task_id}.png"
    contact_sheet(gif, segments, marks, sheet)
    summary = {
        "task": task_id,
        "passed": not failures,
        "failures": failures,
        "gif": str(gif),
        "sheet": str(sheet),
        "segments": [
            {
                "scene": s.preview.scene or None,
                "motion": s.motion,
                "seconds": s.seconds,
                "filmed_steps": s.filmed,
                "steps": len(s.steps),
                "steps_per_s": round(s.rate, 1),
                "frames_per_s": round(s.fps, 1),
                "terminations": s.terms,
                "graph_runs": s.runs,
                "root_z": [round(step["z"], 3) for step in s.steps if "z" in step],
                "root_tilt": [
                    round(step["tilt"], 1) for step in s.steps if "tilt" in step
                ],
            }
            for s in segments
        ],
    }
    (out_dir / f"{task_id}.json").write_text(json.dumps(summary, indent=1))
    filmed = sum(s.filmed for s in segments)
    steps = sum(len(s.steps) for s in segments)
    verdict = "passed" if not failures else "FAILED"
    print(f"  checks {verdict}: {steps} control steps, {filmed} of them filmed")
    for failure in failures:
        print(f"    - {failure}")
    print(f"  {_shown(sheet)}, {_shown(out_dir / f'{task_id}.json')}")
    return failures


def _shown(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


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
    parser.add_argument(
        "--tilt",
        type=float,
        help="override the recipe's camera drop towards the horizon",
    )
    parser.add_argument("--crop", help="override the recipe's crop, w:h:x:y of 960x702")
    parser.add_argument(
        "--motion", help="film this motion alone, instead of the recipe's `motions`"
    )
    parser.add_argument(
        "--scene", help="film this scene's cut alone (the defaults if it has none)"
    )
    parser.add_argument(
        "--keep-frames", type=Path, help="also keep the PNG sequence here"
    )
    parser.add_argument(
        "--shot",
        type=Path,
        help="write the framing to this PNG instead of filming a GIF",
    )
    parser.add_argument(
        "--software",
        action="store_true",
        help="no GPU: render through Mesa and slow the page's clock to keep up",
    )
    parser.add_argument(
        "--chromium", help="a Chromium binary to launch instead of Playwright's own"
    )
    parser.add_argument(
        "--check-seconds",
        type=float,
        default=CHECK_SECONDS,
        help=f"how long the app runs on after the clip, checked (default {CHECK_SECONDS:g})",
    )
    parser.add_argument(
        "--no-check", action="store_true", help="film without checking the run"
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
    browser = {
        "clock": SOFTWARE_CLOCK if args.software else 1.0,
        "chromium": args.chromium,
    }
    check_seconds = None if args.no_check else args.check_seconds
    failed: list[str] = []

    for task_id in tasks:
        overrides = {
            key: value
            for key, value in (
                ("seconds", args.seconds),
                ("orbit", args.orbit),
                ("tilt", args.tilt),
                ("crop", args.crop),
                ("motions", (args.motion,) if args.motion else None),
            )
            if value is not None
        }
        recipe = PREVIEWS.get(task_id, Preview())
        cuts = recipe if isinstance(recipe, tuple) else (recipe,)
        if args.scene:
            cuts = tuple(c for c in cuts if c.scene == args.scene) or (
                Preview(scene=args.scene),
            )
        cuts = tuple(dataclasses.replace(cut, **overrides) for cut in cuts)

        try:
            app_dir = ensure_built(task_id, args.dist)
        except Exception as error:
            # A task that cannot build here (a gated download) must not stop the rest.
            print(f"[{task_id}] build failed: {error}")
            failed.append(task_id)
            continue
        server = serve(app_dir)

        def url(cut: Preview) -> str:
            query = "&".join(
                q for q in (cut.scene and f"scene={cut.scene}", cut.query) if q
            )
            return f"http://127.0.0.1:{server.server_address[1]}/?{query}"

        try:
            if args.shot:
                cut = cuts[0]
                print(
                    f"[{task_id}] framing {url(cut)} "
                    f"(orbit {cut.orbit:g}, tilt {cut.tilt:g}, crop {cut.crop})"
                )
                film(
                    url(cut),
                    cut,
                    shot=args.shot,
                    motion=cut.motions[0] if cut.motions else None,
                    **browser,
                )
                crop_png(args.shot, cut.crop)
                print(f"  {args.shot}")
                continue
            print(f"[{task_id}] filming {sum(c.seconds for c in cuts):g}s")
            rate = control_rate(app_dir)
            gif = args.out_dir / f"{task_id}.gif"
            with tempfile.TemporaryDirectory() as tmp:
                frames_dir = (
                    args.keep_frames / task_id if args.keep_frames else Path(tmp)
                )
                # One segment per cut and motion, appended into one sequence; a cut
                # without `motions` is a single segment of whatever its setup left running.
                segments: list[Segment] = []
                for cut in cuts:
                    motions = cut.motions or (None,)
                    for motion in motions:
                        segments.append(
                            film(
                                url(cut),
                                cut,
                                frames_dir=frames_dir,
                                expected_rate=rate,
                                motion=motion,
                                seconds=cut.seconds / len(motions),
                                start_index=segments[-1].next_index if segments else 0,
                                check_seconds=check_seconds,
                                **browser,
                            )
                        )
                to_gif(frames_dir, gif, segments)
            if check_seconds is not None and report(
                task_id,
                segments,
                1 / rate,
                rate,
                per_step_graphs(app_dir),
                gif,
                args.dist / "preview",
            ):
                failed.append(task_id)
        finally:
            server.shutdown()

    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
