# LEAP in-hand rotation — working prototype

A running browser demo of [Msornerrrr/in-hand-rotation-mjlab](https://github.com/Msornerrrr/in-hand-rotation-mjlab),
built to test the conclusions in [`../../in-hand-rotation-feasibility.md`](../../in-hand-rotation-feasibility.md).
It is **not** wired into the package or CI: it needs mjswan changes that are not
released yet (the patch beside it), so `registry.py` does not know about it.

![the demo running](leap_inhand_rotation.gif)

About 160 s of headless-Chromium wall clock at 16× speed. The policy holds the cube and
rotates it for the whole run — it never drops.

## What the mjswan patch adds

`mjswan-relative-joint-position.patch` applies to mjswan `main`. All three pieces are
mjlab parity — features mjlab already has that mjswan did not, none of them specific to
this task:

| Change | Why it is generic |
|---|---|
| `RelativeJointPositionActionCfg` (Python cfg + `joint_position_relative` in `applyAction.ts`) | A direct port of mjlab's own action term. mjswan did not carry it even as a compat stub. |
| `joint_pos_target` in `slotReader.ts` | `Entity.data.joint_pos_target` is a plain mjlab field; any task observing its own position command reads it. Recorded by the action layer, since MuJoCo keeps no such array — `ctrl` holds a post-bias target or a torque. |
| The same field cleared on reset | mjlab's `EntityData.reset` zeroes it; without this the first frame of an episode sees the previous one's command. |

The recording is served by every position term, so `joint_position` and
`joint_position_reference` gain the observation too, not just the new one.

`npm test` in `src/mjswan/template`: 367 passing, 5 of them new.

## Running it

```sh
# mjswan checked out beside this directory, patch applied, installed editable
git -C mjswan apply mjswan-relative-joint-position.patch
uv pip install -e ./mjswan "mjlab==1.5.3"

python export_leap_policy.py     # ckpt -> leap_actor.onnx (320 -> 16)
python build_leap_demo.py        # -> dist-leap/
python capture_leap.py --seconds 150   # headless run -> video/*.webm
python make_gif.py --seconds 10       # -> leap_inhand_rotation.gif
```

`leap_compat.py` and the upstream clone are expected beside these scripts.

## What each file does

| File | Role |
|---|---|
| `leap_compat.py` | Two shims (`DelayedActuatorCfg`, `update_assets`) so upstream's **robot definition** imports under mjlab 1.5.3. Nothing else from upstream is imported. |
| `leap_inhand_task.py` | The task, restated against current mjlab: scene, the actor's two observation terms, the action, one termination. Registers itself with mjlab's task registry so `add_scene_mjlab` can drive it. |
| `export_leap_policy.py` | Actor + empirical normalizer → ONNX, checked against PyTorch. |
| `build_leap_demo.py` | `Builder` → `add_scene_mjlab` → `add_policy` → `dist-leap/`. |
| `capture_leap.py` | Serves the build with COOP/COEP, drives it in Chromium, records video. |
| `make_gif.py` | webm → GIF of a target length. Playwright's bundled ffmpeg muxes only webm, so ffmpeg decodes to PNGs and Pillow does the palette. |

## The one fidelity gap

Upstream integrates the joint target **on its own previous command**
(`q_cmd += delta`, clamped to the soft limits). mjlab's generic term — the one this
runs — re-bases on the **measured** position (`q_cmd = q + delta`). Under contact the
measurement lags the command, so the integrator can hold a target the relative term
cannot, and the two are not the same controller.

The demo shows the policy still works: it grips and rotates for 90 s without dropping.
It is not a faithful reproduction of upstream, and the rotation rate has not been
measured against the upstream env. Closing that gap means a command-space variant of
the term — reasonable as a flag on the same class, but it is *not* something mjlab has,
so it is a separate decision from this patch.

Also dropped, all deliberately: domain randomization, the grasp-cache resampler (the
demo bakes one grasp at the nominal cube size), the stateful pose-deviation
termination, and the asymmetric critic.
