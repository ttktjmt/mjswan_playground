# LEAP in-hand rotation — working prototype

A running browser demo of [Msornerrrr/in-hand-rotation-mjlab](https://github.com/Msornerrrr/in-hand-rotation-mjlab),
built to test the conclusions in [`../../in-hand-rotation-feasibility.md`](../../in-hand-rotation-feasibility.md).
It is **not** wired into the package or CI: it needs mjswan changes that are not
released yet (the patch beside it), so `registry.py` does not know about it.

![the demo running](leap_inhand_rotation.gif)

About 160 s of headless-Chromium wall clock at 16× speed. Measured in the browser, not
eyeballed: the cube's yaw moves ≈ −4.0 rad over 26 s (≈0.15 rad/s), and the silhouette's
principal axis spans 178° over the run — against 0.6° when it was stuck.

## What the mjswan patch adds

`mjswan-relative-joint-position.patch` applies to mjswan `main`. Three of the four
pieces are mjlab parity — features mjlab already has that mjswan did not, none of them
specific to this task. The fourth is not, and is marked as such.

| Change | mjlab parity? |
|---|---|
| `RelativeJointPositionActionCfg` (Python cfg + `joint_position_relative` in `applyAction.ts`) | **Yes** — a direct port of mjlab's own action term, which mjswan did not carry even as a compat stub. |
| `joint_pos_target` in `slotReader.ts` | **Yes** — `Entity.data.joint_pos_target` is a plain mjlab field; any task observing its own position command reads it. Recorded by the action layer, since MuJoCo keeps no such array (`ctrl` holds a post-bias target or a torque). |
| The same field cleared on reset | **Yes** — mjlab's `EntityData.reset` zeroes it. |
| `relative_to="command"` on that term | **No.** mjlab has only the measured-position form. This is the integrator variant (`q_cmd += delta`), the standard delta controller in dexterous manipulation. See below for why it had to exist. |
| All of `MujocoCfg` reaching `spec.option`, not just the flags | **Yes** — a straight gap. See below. |
| `resolve_pd_gains` covering the new term | n/a — part of implementing it. |

The target recording is done by every position term, so `joint_position` and
`joint_position_reference` gain the observation too, not just the new one.

`npm test` in `src/mjswan/template`: **373 passing, 11 of them new.**

### `MujocoCfg` was being dropped on the floor

`apply_mjlab_sim_options` transferred `disableflags`/`enableflags` and **nothing else**:
`timestep`, `integrator`, `cone`, `impratio`, `solver`, `iterations`, `tolerance`,
`ls_iterations`, `ls_tolerance`, `ccd_iterations` and `gravity` were all silently
dropped, so the browser ran MuJoCo's defaults under the task's name — Euler instead of
implicitfast, a **pyramidal** cone instead of elliptic, `impratio` 1 instead of 10, and
a 0.002 timestep instead of 0.005 (which the runtime then compensated for by choosing
decimation 25).

For contact-rich manipulation the cone and `impratio` are not details: they are what
makes the fingers grip rather than slip. This is generic — any mjlab task with
non-default sim options is affected — and it is why `husky_skater/main.py` writes those
same fields into its spec by hand, with a comment explaining that they have to travel
in the spec. That workaround is no longer needed.


### Why `relative_to="command"` is not optional here

The two forms are different controllers, not variants of one. Under contact the measured
position lags the command, so an integrator can hold a target the measured-position form
never reaches. `rollout_check.py` runs the ONNX policy in the real mjlab env for 400
steps (20 s) on each:

| action term | cube yaw | rate | dropped |
|---|---|---|---|
| `relative_to="command"` (upstream's) | **−6.10 rad** (0.97 turn) | **−0.305 rad/s** | no |
| `relative_to="position"` (mjlab's) | 0.02 rad | 0.001 rad/s | no |

Upstream's training target for `rotation_progress` was 0.20 rad/s, so the integrator
clears it and mjlab's own term does not rotate the cube at all. The first version of
this prototype ran mjlab's term and looked broken on video for exactly this reason.

### The PD gains never reached the browser

mjlab's ideal-PD actuators compile to `<general>` actuators with no gain or bias
params — motors, taking a torque. mjswan therefore runs the PD itself and reads the
gains off the *action* term, which `resolve_pd_gains` fills in from the entity's
actuator configs. It matched on `JointPositionActionCfg` and
`ReferenceJointPositionActionCfg` only, so the new term got `kp = kd = 0` and every
`ctrl` was zero. The hand still held the cube — on joint friction alone — which is
exactly why this was invisible on video.

### The other bugs that made it look broken

`events={}`. mjlab applies an entity's `init_state` through reset **events**, not
automatically — with none, the hand sat at all-zero joints and the cube at the world
origin. Upstream's three reset events, with their randomisation ranges zeroed, are now
stated in `leap_inhand_task.py`.

A related one, invisible on video: the hand's `init_state.joint_pos` had been taken from
the grasp cache's recorded pose. Upstream's cache reset writes the **cube's** size and
pose and nothing else ("Robot joints are not touched"), so the hand always starts at the
fixed `GRASP_INIT_JOINT_POS` — which is also what `joint_pos_rel` subtracts. Using the
cache pose biased 160 of the policy's 320 inputs.

## Running it

```sh
# mjswan checked out beside this directory, patch applied, installed editable
git -C mjswan apply mjswan-relative-joint-position.patch
uv pip install -e ./mjswan "mjlab==1.5.3"

python export_leap_policy.py          # ckpt -> leap_actor.onnx (320 -> 16)
python rollout_check.py --steps 400   # sanity: does the policy rotate the cube?
python build_leap_demo.py             # -> dist-leap/
python capture_leap.py --seconds 150  # headless run -> video/*.webm
python make_gif.py --seconds 10       # -> leap_inhand_rotation.gif
```

`leap_compat.py` and the upstream clone are expected beside these scripts.

## What each file does

| File | Role |
|---|---|
| `leap_compat.py` | Three shims (`DelayedActuatorCfg`, `update_assets`, `TerrainImporterCfg`) so upstream's **robot definition** imports under mjlab 1.5.3. Its task config is not imported. |
| `leap_inhand_task.py` | The task, restated against current mjlab: scene, reset events, the actor's two observation terms, the action, one termination. Registers itself with mjlab's registry so `add_scene_mjlab` can drive it. |
| `export_leap_policy.py` | Actor + empirical normalizer → ONNX, checked against PyTorch. |
| `rollout_check.py` | Rolls the ONNX policy out in the real mjlab env and measures cube yaw, for either action term. The right place to debug a policy — the browser is not. |
| `build_leap_demo.py` | `Builder` → `add_scene_mjlab` → `add_policy` → `dist-leap/`. |
| `capture_leap.py` | Serves the build with COOP/COEP, drives it in Chromium, records video. |
| `probe_browser.py` | Reads cube yaw, action and integrator target out of the running page. Needs the debug hook described below. |
| `cmp_obs.py` | Puts the browser's observation vector next to mjlab's, element by element. |
| `make_gif.py` | webm → GIF of a target length. Playwright's bundled ffmpeg muxes only webm, so ffmpeg decodes to PNGs and Pillow does the palette. |

## What is still dropped, deliberately

Domain randomization (observation noise and delay, cube mass/size/friction, actuator
gains and delays), the grasp-cache resampler (the demo bakes one grasp at the nominal
cube size), the stateful pose-deviation termination, and the asymmetric critic.

Rollout fidelity against upstream has **not** been measured term by term — contact-rich
manipulation diverges from any per-step parity, so the honest claim is the yaw rate
above, not trajectory equality. The browser turns the cube at ≈0.15 rad/s against
mjlab's 0.305: the same behaviour, at about half the rate. The likely cause is mjlab's
actuator command delay, which the browser does not model; that has not been confirmed.

`mjswan.compile.run_parity` does pass exactly (`max_abs_diff = 0.0`) on every
observation, termination and event graph — worth knowing, because it passed just as
cleanly while the demo was completely broken. Graph parity says the traced maths is
right; it says nothing about what the browser feeds those graphs, or what the action
layer does with the result.

## Reproducing the debugging

`probe_browser.py` and `cmp_obs.py` need a hook the patch deliberately does not carry —
production builds strip `console.log`, so there is otherwise no way to see inside a
running page. To re-attach it, in `src/engine/createEngine.ts`:

```ts
const runtime = new mjswanRuntime(mujoco, element, options.termSeed);
(window as unknown as Record<string, unknown>).__mjswanRuntime = runtime;
return new Engine(runtime);
```

plus a `debugProbe()` on `mjswanRuntime` returning `mjData.qpos`, the policy's last
actions, and the action term's `commandTarget`. Whether mjswan should ship something
like this behind a flag is a separate question, but the three bugs above were all found
this way and none of them were visible any other way.
