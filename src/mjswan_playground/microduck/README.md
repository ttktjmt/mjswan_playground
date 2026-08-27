# MicroDuck (`microduck`)

Source: https://github.com/pollen-robotics/microduck_rl (Apache-2.0, **3D models
CC BY-SA-NC** — see [License](#license)) ·
robot and policies from https://github.com/pollen-robotics/microduck (Apache-2.0)

A 25 cm, 800 g bipedal duck, and every policy its onboard daemon ships — walking with a
commandable head, standing with trunk-pose control, sitting down and getting back up,
touching the ground with its mouth, kicking a ball with either foot, rolling forward over
its own head, and skating on passive wheels.

On the robot one daemon hot-swaps these behind a shared 61-value observation contract.
Each one gets its own scene here, for the reason in
[What differs from upstream](#what-differs-from-upstream).

## Run

```sh
uv run mjswan-playground run microduck
```

The build clones both repositories into `.cache/` at pinned commits; set
`MJSWAN_MICRODUCK_RL_ROOT` / `MJSWAN_MICRODUCK_ROOT` to point at checkouts you already
have.

| From upstream | Used as |
|---|---|
| `robot/microduck/scene.xml`, `scene_rollers.xml`, `scene_ball.xml` | the scenes — MJCF as exported from Onshape, `<position>` actuators and all |
| its `STAND` keyframe | the reset pose, the default the actions offset from, and what `joint_pos_rel` subtracts (upstream's `DEFAULT_POSE`) |
| the actuator block's joint order | `policy_joint_names` — the 14 servos in the order the runtime indexes them |
| `policies/*.onnx` (9 files, from the robot's repo) | the policies, normalizers baked in, exactly the files `robotd` loads |
| the env configs' command ranges and terminations | the slider bounds and the fall reset, per policy |
| `scripts/infer_policy.py` | the reference: the same ONNX files driven against the same XML in plain MuJoCo |

## The nine

| Scene | Policy | Twist slot (3) | Head (4) | Body (6) | Resets on a 70° tilt |
|---|---|---|---|---|---|
| Walk | `alpha_walking` | forward / sideways / turn sliders | sliders | zero | yes |
| Stand & Pose | `alpha_stand` | zero | sliders | height / roll / pitch sliders, x·y·yaw zero | no |
| Sit / Stand | `alpha_sitstand` | a Sit checkbox in slot 0 | sliders | zero | no |
| Ground Pick | `alpha_ground_pick` | a 4 s phase clock | zero | zero | yes |
| Roulade | `roulade` | zero | zero | zero | no |
| Ball Kick (right) | `ball_kick_right` | zero | zero | zero | yes |
| Ball Kick (left) | `ball_kick_left` | zero | zero | zero | yes |
| Roller Skate | `roller` | throttle + heading-error sliders | zero | zero | yes |
| Roller Crouch | `roller_crouch` | a 4 s phase clock | zero | zero | yes |

"Zero" is not a shortcut: every policy reads the same 13-wide command block, and an env
that does not drive a slot pads it with zeros rather than dropping it
(`zero_command_padding`). That padding is what lets one runtime swap any of these in.

## What the policies read

All nine are `obs[1, 61] -> actions[1, 14]`, no history:

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | `root_link_ang_vel_b` |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `joint_pos` | 14 | `joint_pos_rel`, against the `STAND` pose |
| `joint_vel` | 14 | `joint_vel_rel` |
| `actions` | 14 | the previous action |
| command | 13 | `[twist(3), head_pose(4), body_pose(6)]` |

Upstream reads the first slot off a `gyro` sensor on the `imu` site rather than the root
body. The site sits on `trunk_base` at identity rotation, so the two are the same vector;
`imu_ang_vel` carries no `noise` attribute either, so neither reading is perturbed.

The action is a joint-position delta at 50 Hz: `ctrl = STAND + action`, `action_scale` 1.0,
straight into the XML's own `<position>` actuators (`kp` 0.55, `kv` 0, `forcerange` ±0.96
N·m). Nothing here configures a PD — the gains are in the model, which is where the
browser reads them.

## Why the scenes are XML, not an mjlab task

Unlike [`pacman`](../pacman/README.md) and [`wbc`](../wbc/README.md), this task does not
import upstream's env configs. Every one of them drives the robot through
[BAM](https://github.com/Rhoban/bam): `edit_spec` replaces the position actuators with
torque motors and zeros the joint friction and damping, and a torch `compute()` supplies
the XL330's voltage control law, back-EMF, Coulomb/Stribeck friction and a 3–6 step
command delay each step, under per-env randomization. The browser has no counterpart, and
mjswan resolves PD gains off `IdealPdActuatorCfg` alone, which `BamActuatorCfg` is not.

BAM is a training-time model. The MJCF already carries the position actuators the real
servos' firmware runs, and upstream's own `infer_policy.py` drives these exact ONNX files
against these exact XMLs in plain CPU MuJoCo. That is the deployment this task reproduces,
so the scenes compile from the XML — [`husky`](../husky/README.md)'s shape, not pacman's.
It also means the pinned `mjlab` version never has to agree with upstream's.

## What differs from upstream

- **One policy per scene.** mjswan 0.9.3 writes a scene's fused observation graph to
  `obs/<group>.onnx`, one path for every policy on the scene, so two policies reading
  different command slots would overwrite each other's graph. These nine need five
  layouts, so they get nine scenes off four specs: upstream's three XMLs, with the kick
  one mirrored per foot.
- **The actuator is the XML's, not BAM's** — see above. The joint `damping` (0.053),
  `frictionloss` (0.0048) and `armature` (0.0018) the XML carries stay as exported;
  under BAM the first two are zeroed and recomputed in torch.
- **No domain randomization.** Battery voltage and its sag under load, command delay,
  friction magnitude, CoM offsets, encoder bias, pushes: all training-time, none applied
  by the robot either.
- **No episode timeout.** Upstream ends an episode at 20 s; an interactive demo has no
  episode budget. The 70° fall reset is upstream's own `fell_over`, kept exactly where
  upstream keeps it — the standing, sitting and roulade policies delete it, because they
  are the ones that start or end on the ground.
- **A padded slot reads zero**, where upstream resamples a narrow "keep the input neurons
  alive" range in it — ±5 mm / ±0.05 rad of body pose under the walking policy, ±0.01 m/s
  of twist under the standing one. Zero is inside every one of those ranges.
- **The pick and crouch clock never exits.** Upstream's runtime runs the cycle to phase
  0.7 and hands back to walking; here it keeps turning, which is what the training
  command itself does (`phase = (phase + dt / period) % 1`).
- **The kick ball is placed, not jittered.** `reset_ball_in_front_of_foot` draws ±15 mm
  per axis around its `offset` at every reset — the randomization that makes a ball-blind
  swing robust rather than aimed. A scene bakes one placement, so it bakes the one that
  function's docstring derives, `(0.08, ∓0.042)`: "the toe tip at x≈0.034, so (0.08,
  -0.042) puts a 35mm-radius ball ~1cm in front of the toe". Its `offset` default is
  0.09, the centre of the draw rather than a placement meant to stand alone — at 0.09 the
  right-foot swing passes a stationary ball without touching it, which is the jitter's
  job to cover.
- **The roller heading slider is `infer_policy.py`'s, not the env config's.** At the
  pinned commit the roller env clips the heading slot to 0 ("straight-line focus"), while
  the inference script that runs this checkpoint exposes ±1.0 rad. The slider follows the
  script and defaults to 0; the shipped `roller.onnx` predates the pinned config, and its
  training run is not recorded (see below).
- **`STAND` is the only keyframe.** Upstream's scenes list `INIT` first, then `STAND`,
  `SIT`, `FOLD`. The browser resets to the first keyframe and so does the tracing env,
  whose `default_joint_pos` is what `joint_pos_rel` bakes in, so `INIT`'s zero pose would
  land in every observation. The root height is `infer_policy.py`'s spawn (0.125 m, or
  0.1385 m on wheels), 5 mm above the keyframe's own.
- **Wheel-bearing friction is written into the spec** (`passive_*`, 0.003), which upstream
  does at inference time because "non-zero frictionloss in the XML breaks training".

## Provenance of the checkpoints

The nine ONNX files are vendored into the robot's repo from `apirrone/microduck_runtime`
at commit `5f3b314`, dereferencing symlinks that pointed at particular training runs
(`alpha_walking.onnx` was `BEST_alpha_walking_rough.onnx`). The names are **roles**, not
runs: which env config and which commit each was trained against is not recorded upstream.
So the pairings in the table above are read off those names, the deploy repo's own role
table, and what `infer_policy.py` loads each one as — not off a training manifest. The
61-value observation contract they were trained to is documented and version-checked at
load by `robotd`, and that is what this task builds.

Four registered task families have no shipped checkpoint at all (Swizzle, RollerSlope,
RollerStandUp, Spin), as do all fifteen Backlash variants; they are absent here for that
reason.

## Fidelity

Against upstream's own constants:

- **The default pose** matches `infer_policy.py`'s `DEFAULT_POSE` to `4e-05` — the residue
  of upstream rounding its copy to four decimals; this one is read out of the keyframe.
- **The joint order** is identical to the order upstream's runtime indexes
  (`jnt_qposadr[actuator_trnid]`), which also drops the rollers model's four passive wheel
  hinges on its own.
- **`base_ang_vel`**: upstream reads a `gyro` on the `imu` site, this task reads mjlab's
  `root_link_ang_vel_b`. Over 24 randomized states they agree to `9.7e-07` — the site is
  on the root body at identity rotation — and a 15 s walk is identical either way.
- **`joint_pos`**: upstream's actor reads `joint_pos_rel(biased=True)`, the
  encoder-bias-randomized variant. The bias is a training event; at zero bias it is
  `joint_pos_rel`, which is what this task reads.
- **The observation** traces to exactly 61 values and **the action** to 14, with every
  term at `scale=1.0` — upstream sets no scale on any term the microduck reads, and its
  action scale is an explicit 1.0 (overriding mjlab's 0.5).
- **The simulation settings** are mjlab's for the velocity task, written into the spec
  because the XML carries no `<option>` and the browser compiles from it: timestep 0.005
  with `control_dt` 0.02 (50 Hz, `decimation` 4), `implicitfast`, 10 solver and 20
  line-search iterations. Upstream's inference script sets only the timestep and so runs
  MuJoCo's XML defaults (Euler, 100/50) — over the rollouts below the two are
  indistinguishable to three decimals.

Behavior, each policy driven against this task's spec, joint order and observation layout
in plain CPU MuJoCo from the `STAND` reset, 12 s unless noted:

| Policy | What it does |
|---|---|
| Walk, zero command | stands — trunk 0.116 m, tilt ≤ 0.6° |
| Walk, forward | 0.129 m/s at a 0.3 command, 0.168 at 0.4; below ≈0.25 it stands |
| Walk, turn | 0.39 rad/s in place at a 1.0 command, 0.62 with 0.3 forward; below ≈0.8 it holds |
| Stand & Pose | stands — 0.116 m, tilt ≤ 1.4° |
| Sit / Stand, Sit off | 0.116 m; upstream's `stand_z` is 0.115 |
| Sit / Stand, Sit on | settles at 0.062 m; upstream's `sit_z` is 0.060 |
| Ground Pick | cycles — trunk down to 0.084 m, pitching through 34° |
| Roulade | rolls through 164° of tilt, 0.44 m forward, ends upright at 0.118 m |
| Ball Kick (right / left) | the ball travels 3.1 m / 3.9 m |
| Roller Skate, throttle 0.4 | skates 4.4 m in 12 s, curving; tilt ≤ 14° |
| Roller Crouch | cycles — trunk down to 0.068 m |

No NaNs in any of them, and nothing falls that is not meant to (the roulade's 164° is the
roll, which is why upstream deletes the fall termination for it, and so does this task).

**The walk tracks its command loosely** — a deadband to ≈0.25 m/s, then roughly 40% of
command; turning is the same shape. That is the checkpoint under this actuator, not a
wiring error: the observation, the joint order, the action scale and the control rate are
each checked above, and the rollout is upstream's own inference control path. What it is
not is BAM, which is what the policy trained against, and at this scale upstream puts most
of the sim-to-real gap in exactly that. The sliders span the ranges the curricula reach,
so the useful part of the forward slider is its upper half.

In headless Chromium all nine scenes come up and run 15 s with no page or console
errors, each showing exactly the controls its policy drives and nothing else: the walk
its twist and head sliders, stand its head and trunk-pose sliders, sit/stand its Sit
checkbox, the roller scene a throttle and a heading, and the four machine-driven ones no
command panel at all. The pick is mid-crouch with its mouth at the floor and the roulade
mid-tuck when the frame is grabbed, so the traced phase clock and the zero-command
policies are both live browser-side.

Also checked: the built bundle is 9 scenes, 45 data files, 73 MB, largest file 7.8 MB —
inside mjswan Cloud's 64 files / 200 MB / 50 MB-per-file limits, with room for the
`.mjz` growth a mesh decimation pass would undo.

## License

The code and the policies are Apache-2.0. **The 3D model files are not**: `microduck_rl`
licenses them under Creative Commons **BY-SA-NC**, and they are 21 MB of the 26 MB of STL
this task compiles into each `scene.mjz`. NonCommercial and ShareAlike both bear on
redistributing that bundle — publishing it to a host, mjswan Cloud included — in a way
Apache-2.0 upstream assets do not. Building and running locally is what this directory
supports; clear the publication question with the authors first.
