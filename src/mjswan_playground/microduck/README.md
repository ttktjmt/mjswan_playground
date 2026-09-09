# Microduck (`microduck`)

Source: https://github.com/pollen-robotics/microduck_rl (Apache-2.0, **3D models
CC BY-SA-NC** — see [License](#license)) ·
robot and policies from https://github.com/pollen-robotics/microduck (Apache-2.0)

A 25 cm, 800 g bipedal duck, and every policy its onboard daemon ships — walking with a
commandable head, standing with trunk-pose control, sit ↔ stand, touching the ground with
its mouth, kicking a ball with either foot, a forward roll, and skating on passive wheels.
On the robot one daemon hot-swaps them behind a shared 61-value observation contract; here
they hot-swap the same way, on the four scenes the XMLs actually differ in.

## Run

```sh
uv run msp run microduck
```

The build clones both repositories into `.cache/` at pinned commits.

| From upstream | Used as |
|---|---|
| `robot/microduck/scene.xml`, `scene_rollers.xml`, `scene_ball.xml` | the scenes — MJCF as exported from Onshape, `<position>` actuators and all |
| its `STAND` keyframe | the reset pose, what actions offset from, and what `joint_pos_rel` subtracts (upstream's `DEFAULT_POSE`) |
| the actuator block's joint order | `policy_joint_names` — the 14 servos in the order the runtime indexes them |
| `policies/*.onnx` (9 files, from the robot's repo) | the policies, normalizers baked in, exactly the files `robotd` loads |
| the env configs' command ranges and terminations | the slider bounds and the fall reset, per policy |
| `scripts/infer_policy.py` | the reference: the same ONNX files driven against the same XML in plain MuJoCo |

## The nine policies

| Scene | Policy | Twist slot (3) | Head (4) | Body (6) | Resets on a 70° tilt |
|---|---|---|---|---|---|
| Duck | Walk — `alpha_walking` | forward / sideways / turn sliders | sliders | zero | yes |
| Duck | Stand & Pose — `alpha_stand` | zero | sliders | height / roll / pitch sliders, x·y·yaw zero | no |
| Duck | Sit / Stand — `alpha_sitstand` | a Sit checkbox in slot 0 | sliders | zero | no |
| Duck | Ground Pick — `alpha_ground_pick` | a 4 s phase clock | zero | zero | yes |
| Duck | Roulade — `roulade` | zero | zero | zero | no |
| Ball (right foot) | Kick — `ball_kick_right` | zero | zero | zero | yes |
| Ball (left foot) | Kick — `ball_kick_left` | zero | zero | zero | yes |
| Rollers | Skate — `roller` | throttle + heading-error sliders | zero | zero | yes |
| Rollers | Crouch — `roller_crouch` | a 4 s phase clock | zero | zero | yes |

Four scenes, not nine: mjswan traces each policy's terms into its own `mdp/<policy>/`, so
policies only split where the scene does — the ball, baked in at one of two placements,
and the wheels.

"Zero" is not a shortcut: every policy reads the same 13-wide command block, and an env
that does not drive a slot pads it rather than dropping it (`zero_command_padding`). That
padding is what lets one runtime swap any of these in.

## What the policies read

All nine are `obs[1, 61] -> actions[1, 14]`, no history, every term at `scale=1.0`:

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | `root_link_ang_vel_b`, where upstream reads a `gyro` on the `imu` site — that site is on the root body at identity rotation, and the two agree to `9.7e-07` over 24 randomized states |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `joint_pos` | 14 | `joint_pos_rel`, against the `STAND` pose |
| `joint_vel` | 14 | `joint_vel_rel` |
| `actions` | 14 | the previous action |
| command | 13 | `[twist(3), head_pose(4), body_pose(6)]` |

The action is a joint-position delta at 50 Hz: `ctrl = STAND + action`, `action_scale`
1.0, straight into the XML's own `<position>` actuators (`kp` 0.55, `kv` 0, `forcerange`
±0.96 N·m). Nothing here configures a PD — the gains are in the model, which is where the
browser reads them.

## Why the scenes are XML, not an mjlab task

Unlike [`pacman`](../pacman/README.md) and [`wbc`](../wbc/README.md), this task does not
import upstream's env configs, because every one of them drives the robot through
[BAM](https://github.com/Rhoban/bam) instead of a MuJoCo actuator
(`FrictionDRBamActuatorCfg`, a fitted XL330 "m6" model). BAM swaps the XML's `<position>`
actuators for torque motors, zeroes `dof_frictionloss`, and computes each step in torch:
a firmware position loop (`kp_fw` 200) whose demand is limited by the voltage actually
available — a per-env battery of 6.5–8.2 V, less a load-dependent sag of up to
`0.2·Σ|τ|`, floored at 6.0 V — then a friction budget subtracted off the result (Coulomb
+ Stribeck + load-dependent + viscous), all behind a 3–6 step command delay. The browser
has no counterpart, and mjswan resolves PD gains off `IdealPdActuatorCfg` alone, which
`BamActuatorCfg` is not.

But BAM is a training-time model. The MJCF already carries the position actuators the
real servos' firmware runs, and upstream's own `infer_policy.py` drives these exact ONNX
files against these exact XMLs in plain CPU MuJoCo. That is the deployment this task
reproduces, so the scenes compile from the XML — [`husky`](../husky/README.md)'s shape,
not pacman's. It also means the pinned `mjlab` version never has to agree with upstream's.

## What differs from upstream

- **The actuator is the XML's, not BAM's** — above. The joint `damping` (0.053),
  `frictionloss` (0.0048) and `armature` (0.0018) stay as exported; under BAM the first
  two are zeroed and recomputed in torch.
- **No domain randomization, no episode timeout.** Voltage sag, command delay, friction,
  CoM offsets, encoder bias and pushes are all training-time, and none is applied by the
  robot either; a 20 s episode budget means nothing to an interactive demo. The 70° fall
  reset is upstream's own `fell_over`, kept on the policies upstream keeps it on.
- **A padded slot reads zero**, where upstream resamples a narrow "keep the input neurons
  alive" range — ±5 mm / ±0.05 rad of body pose, ±0.01 m/s of twist. Zero is inside every
  one of those.
- **The pick and crouch clock never exits.** Upstream's runtime runs the cycle to phase
  0.7 and hands back to walking; here it keeps turning, which is what the training command
  itself does (`phase = (phase + dt / period) % 1`).
- **The kick ball is placed, not jittered**, at `(0.08, ∓0.042)` — the placement
  `reset_ball_in_front_of_foot`'s docstring derives, not its `offset` default of 0.09.
  That default is only the centre of a ±15 mm per-axis draw, the randomization that makes
  a ball-blind swing robust rather than aimed; a scene bakes one placement, and at 0.09
  the right-foot swing passes a stationary ball.
- **The roller heading slider is `infer_policy.py`'s ±1.0 rad**, not the pinned env
  config's clip to 0 ("straight-line focus") — the shipped `roller.onnx` predates that
  config. It defaults to 0.
- **`STAND` is the only keyframe.** The browser and the tracing env both take the first
  one, so upstream's leading `INIT` would land its zero pose in every `joint_pos_rel`. The
  root height is `infer_policy.py`'s spawn, 0.125 m (0.1385 m on wheels).
- **Wheel-bearing friction is written into the spec** (`passive_*`, 0.003), which upstream
  does at inference time because "non-zero frictionloss in the XML breaks training".

## Provenance of the checkpoints

The names are **roles**, not runs. The nine files were vendored into the robot's repo from
`apirrone/microduck_runtime` at commit `5f3b314`, dereferencing symlinks that pointed at
particular training runs (`alpha_walking.onnx` was `BEST_alpha_walking_rough.onnx`); which
env config each was trained against is not recorded upstream. The pairings above are read
off those names, the deploy repo's own role table and what `infer_policy.py` loads each
one as — not off a training manifest. What *is* documented and version-checked at load by
`robotd` is the 61-value observation contract, and that is what this task builds.

Four registered task families ship no checkpoint at all (Swizzle, RollerSlope,
RollerStandUp, Spin), as do all fifteen Backlash variants; they are absent for that reason.

## The one known gap

**The walk tracks its command loosely** — a deadband to ≈0.25 m/s, then roughly 40% of
command; turning is the same shape. The observation, the joint order, the action scale and
the control rate each check out, and this is upstream's own inference control path, so it
is the checkpoint under this actuator rather than a wiring fault. What it is not is BAM,
which is what the policy trained against, and at this scale upstream puts most of the
sim-to-real gap in exactly that.

## License

The code and the policies are Apache-2.0. **The 3D model files are not**: `microduck_rl`
licenses them under Creative Commons **BY-SA-NC**, and they are 21 MB of the 26 MB of STL
this task compiles into each `scene.mjz`. NonCommercial and ShareAlike both bear on
redistributing that bundle — publishing it to a host, mjswan Cloud included — in a way
Apache-2.0 upstream assets do not. The microduck scenes already on mjswan Cloud are up
there under permission granted directly by the author
([@antoinepirrone](https://x.com/antoinepirrone/status/2093233465483292889)); that
permission does not travel with the files, so publish your own copy only under the terms
of the license.
