# Microduck Playground (`microduckpg`)

<img src="../../../assets/microduckpg.gif" width="480" alt="Microduck Playground preview"/>

Source: https://github.com/Vottivott/microduck-playground (Apache-2.0, **3D files
CC BY-NC-SA 4.0**, see [License](#license)) ·
policies from the Hub, one repository per experiment (Apache-2.0):
[running](https://huggingface.co/HannesVonEssen/microduck-running), [swing](https://huggingface.co/HannesVonEssen/microduck-swing), [basketball](https://huggingface.co/HannesVonEssen/microduck-basketball),
[stilts](https://huggingface.co/HannesVonEssen/microduck-stilts), [climb](https://huggingface.co/HannesVonEssen/microduck-climb), [chimney-climb](https://huggingface.co/HannesVonEssen/microduck-chimney-climb),
[long-jump](https://huggingface.co/HannesVonEssen/microduck-long-jump), [backflip](https://huggingface.co/HannesVonEssen/microduck-backflip)

Every experiment microduck-playground publishes a policy for, on the 25 cm Microduck:

| Scene | Policies | What it does |
|---|---|---|
| [Running](#running) | Run | sprints at about 1.65 m/s, with a Forward slider from 0 to 2.2 m/s |
| [Swing](#swing) | Pump | pumps itself on a swing from rest to about 160° |
| [Basketball](#basketball) | Balance | balances on a basketball it cannot sense, with an LSTM policy |
| [Stilts 10 cm to 2.0 m](#stilts) | Walk | walks on stilts, eight scenes with one policy per height |
| [Desk Climb](#desk-climb) | Climb, Get up | climbs a ladder onto a desk and stands up there |
| [Chimney Climb](#chimney-climb) | Enter, climb, exit; Climb; Exit | braces up a 12.5 cm gap to a 3 m platform, exits and stands |
| [Long Jump](#long-jump) | Jump | jumps a 0.30 m gap onto a block 0.25 m lower |
| [Backflip](#backflip) | Backflip | backflips off a 0.8 m platform onto a mat |

## Run

```sh
uv run msp run microduckpg
```

The first build clones microduck-playground into `.cache/` at a pinned commit (or reads
the checkout `MJSWAN_MICRODUCKPG_ROOT` points at) and downloads each policy from
the Hub at a pinned revision. Each experiment is a module beside `main.py`, its
`add_scenes` listed in `EXPERIMENTS` in the order the scenes appear.

The whole build is 175 MB, over the 100 MB that mjswan Cloud takes, so `PARTS` splits it
into three uploads, built by their own IDs (`uv run msp build microduck-moves`):

| Part | Scenes | On mjswan Cloud |
|---|---|---|
| `microduck-moves` | Running, Swing, Basketball, Stilts 1.0 m | [mjswan.com/s/_klQQmL](https://mjswan.com/s/_klQQmL) |
| `microduck-parkour` | Desk Climb, Chimney Climb, Long Jump, Backflip | [mjswan.com/s/mqwst_Y](https://mjswan.com/s/mqwst_Y) |
| `microduck-stilts` | Stilts at every height | [mjswan.com/s/qK0Uwr7](https://mjswan.com/s/qK0Uwr7) |

## What every scene shares

This is the same robot, servo order and 61-value observation contract as
[`microduck`](../microduck/README.md), so every scene follows that task's shape: it
compiles from upstream's MJCF rather than from its env configs, which drive the servos
through BAM, a torch actuator model the browser cannot run. Every policy drives the XML's
own `<position>` servos (`kp` 0.55, ±0.96 N·m), and:

- **Training's servo command delay becomes a filter.** BAM delays each command by 3 to 6
  physics steps; each scene gives every servo a first-order `filterexact` filter instead,
  tuned per experiment in browser-order replays (sections below), starting settled at
  the reset pose.
- **`joint_vel` reads one control step back** (`history_steps=(1,)`), the fixed lag every
  experiment trains with.
- **No domain randomization or observation noise**, except where a section says so.
- **Collision meshes are copies.** mjswan's renderer turns the vertices of every mesh a
  drawn geom uses to y-up in place, in the model the physics also reads, so a collision
  geom sharing a visual mesh would collide with a hull rotated 90°. Each scene gives its
  collision geoms their own mesh copies (`_common.own_collision_meshes`).
- **Upstream's `FULL_COLLISION` is restated** (`_common.full_collision`) where an
  experiment trains with it, since mjlab 1.6.0 can no longer build it. Under mjlab 1.3.0
  it left unnamed collision hulls colliding; the restatement disables them, which removes
  the battery support hull from Stilts and Desk Climb and two leg hulls from Stilts.

## Running

### What the policy reads

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | `root_link_ang_vel_b` |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `joint_pos` | 14 | `joint_pos_rel`, against the `STAND` pose |
| `joint_vel` | 14 | `joint_vel_rel`, one control step back |
| `actions` | 14 | the previous action |
| twist | 3 | the Forward slider, then lateral and yaw at 0 |
| head, body | 4 + 6 | zeros |

### What differs from upstream

- **The servos are the XML's `<position>` actuators** (`kp` 0.55, ±0.96 N·m), not BAM,
  as in `microduck`.
- **The command delay is a filter.** Training delays every servo command by 3 to 6
  physics steps (15 to 30 ms) inside BAM. The browser has no actuator delay, so each
  servo gets a first-order `filterexact` filter with a 30 ms time constant, starting
  settled at `STAND`. Without it the policy falls within 2 s; with 20 ms it still falls a
  few times a minute in plain MuJoCo.
- **`joint_vel` reads one control step back** (`history_steps=(1,)`), the fixed lag
  training applies (`delay_min_lag = delay_max_lag = 1`) for the servo firmware's
  velocity estimate. The 0 to 1 step random lag on the gyro and gravity is dropped:
  zero is inside its range.
- **No domain randomization, no episode timeout**, no observation noise. The 70° fall
  reset is the velocity env's `fell_over`, which training keeps.
- **Lateral, yaw, head and body read zero**, where training samples ±0.02 m/s and
  ±0.05 rad/s of twist and small head and body ranges to keep those inputs alive.
- **The Forward slider starts at 1.0 m/s**, the play config's speed. The policy runs at
  its own pace whatever the command above a walk: the manifest's evaluations reach about
  1.65 m/s at a 2.2 m/s command.
- **`STAND` is the only keyframe**, with the root at 0.125 m, as in `microduck`.

### How it behaves

It runs straight from the reset and keeps running. Heading drifts over several seconds,
which the policy's manifest names as a known limit ("Heading and lateral drift remain
substantial"); there is no yaw command to correct it. The policy was never run on
hardware.

## Swing

Policy from https://huggingface.co/HannesVonEssen/microduck-swing (Apache-2.0) ·
upstream `experiments/swing` (`Mjlab-SwingPump-MicroDuck`)

Microduck strapped into a seat that hangs from two 380 mm elastic cords. It starts still
at the bottom and pumps itself up with its head and legs, with no phase clock and no
scripted push: 150° of full span by about 26 s, then about 160°, the trunk swinging to
±80° from the vertical. There is nothing to steer; Reset starts it over from rest.

| From upstream | Used as |
|---|---|
| `get_swing_spec` in `src/mjlab_microduck/robot/microduck_constants.py`, run from the checkout on `robot/microduck/scene.xml` | the scene: the seat (158 g of meshes fixed to the trunk), the visual-only A-frame and the two tension-only spatial-tendon cords (2000 N/m from 380 mm, limit 395 mm), on the `robot_allcollisions.xml` robot with a floor and lights |
| its `SWING_SEATED_FRAME` and `SWING_BOTTOM_TRUNK_Z` | the reset pose (root at 0.2828 m, cords at their 382 mm hanging length, knees ±1.35 rad), what actions offset from, and what `joint_pos_rel` subtracts |
| `src/mjlab_microduck/tasks/microduck_swing_env_cfg.py` | the actor terms, the 0.7 action scale and the camera |
| `swing_plane_heading_observation` in `src/mjlab_microduck/tasks/mdp.py` | the cue in the twist slot, rewritten as the traceable `swing_plane_cue` |
| `policy.onnx` on the Hub at `e7a603f` (`alpha050.pt`, normalizer and the [-1, 1] action clip baked in) | the policy, `obs[1, 61] -> actions[1, 14]` at 50 Hz |

### What the policy reads

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | `root_link_ang_vel_b` |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `joint_pos` | 14 | `joint_pos_rel`, against `SWING_SEATED_FRAME` |
| `joint_vel` | 14 | `joint_vel_rel`, one control step back |
| `actions` | 14 | the previous action |
| twist | 3 | `swing_plane_cue`: `[0, R[0, 1], R[2, 1]]` from `root_link_quat_w`, the trunk's y axis leaving the swing plane |
| head, body | 4 + 6 | zeros |

### What differs from upstream

- **A friction stand-in for BAM.** The XML's `<position>` servos (`kp` 0.55, ±0.96 N·m)
  leave out BAM's load-dependent Coulomb friction. That friction is about 0.27 × the
  motor torque, so up to about 0.17 N·m with the servos saturated, as they are through
  most of the pump. With the XML's 0.0048 N·m the swing climbs to about 180° and
  slackens the cords to 365 mm at the top, below upstream's 370 mm gate. Capping the
  torque at BAM's 0.64 N·m current limit does not bring it down. Every hinge gets a
  constant `frictionloss` of 0.1 N·m instead, which brings the span back to about 160°.
- **The command delay is the shared 30 ms filter.** BAM delays each command by 3 to 6
  physics steps; the scene uses the same first-order filter as Running. With the
  friction stand-in, 20 ms also holds and pumps up faster (150° at 21 s) to 163°.
  30 ms takes 26 s and settles near 162°.
- **The IMU lag is dropped.** Upstream play keeps a random 0 to 1 step lag on the gyro
  and gravity: mjlab 1.3.0 gates only the noise on `enable_corruption`, not the lag.
  mjswan has fixed lags only, and zero is inside that range. Replayed in Python with the
  random lag, the scene still holds (10 runs of 300 s, no seat flip). `joint_vel` keeps
  its fixed one-step lag.
- **No 6° IMU misalignment, encoder bias, observation noise or battery-voltage
  randomization.**
- **Collisions are the XML's.** mjlab 1.3.0's `FULL_COLLISION` disables the geoms it does
  not match by looking their names up, and `""` finds one visual geom. So the nine
  unnamed hulls of `robot_allcollisions.xml` (trunk, hips, legs, jaw) collided in
  training, and the scene keeps them: each leg rests on the trunk about a third of the
  time. The feet already carry its condim 3 and friction 1. Its priority 1 is dropped,
  since both sides of every foot contact have the same parameters.
- **No terminations and no time limit.** Training stops at 24 s or on NaN, and the
  evaluation runs 36 s with neither. A swing past 70° is not a fall, so the velocity
  env's `fell_over` stays out. Reset is the panel's.
- **The cue's `nan_to_num` is dropped**: it acts only on a non-finite quaternion.
- **`SWING_SEATED_FRAME` is the only keyframe**, named `STAND` like the other scenes'.
  `scene.xml`'s `INIT`, `STAND`, `SIT` and `FOLD` go.
- **The camera is pulled back** from the env's 1.15 m to 1.5 m, so a browser frame holds
  the whole arc.
- **The cords draw red**: mjswan gives spatial tendons a fixed material and ignores
  `tendon_rgba` (upstream's cream).

### How it behaves

Replayed in browser order on the built scene (`scene.mjz`, the traced graphs and the Hub
ONNX, reset to the keyframe), over 20 runs with ±0.002 rad of initial joint offset:

- It reaches 150° of full span at 25 to 27 s.
- The median span at 36 s is 159.5° (158.9 to 159.9°), against the published 163.0°
  median over 100 BAM seeds.
- It then holds about 162° for 300 s (10 runs, no seat flip, no NaN).
- 14 of the 20 runs pass upstream's strict 36 s gates (lateral ≤ 20 mm,
  alignment ≤ 0.05, cords 370 to 394 mm), against the published 71 of 100. The misses
  are lateral drift up to 32 mm, alignment up to 0.073 and cord slack to 368 mm.
- A 0.3 N·s shove on the trunk in x and in y at 25 s is absorbed in all 10 runs.

In Chromium the trunk rose to 0.687 m at the top of each swing, the replay's height to
the millimetre, and kept that through the 25 s checked. The policy has not run on
hardware (its manifest: "sim-only-hardware-candidate").

## Basketball

The duck stands on the apex of a free-rolling size-7 basketball (0.24 m, 0.62 kg) and
keeps it under its feet, with a one-layer LSTM policy that never sees the ball. Forward,
Sideways and Turn sliders feed its twist command, but it barely follows them (see below).
Upstream calls the release (b11, iteration 6999) an experimental hardware-test candidate;
it has never run on a robot.

Policy from https://huggingface.co/HannesVonEssen/microduck-basketball (Apache-2.0).

| From upstream | Used as |
|---|---|
| `src/mjlab_microduck/robot/microduck/scene.xml` | the scene: `robot_allcollisions.xml`, the model the basketball env loads (`get_standup_spec`), on a floor; its `STAND` keyframe is the reset pose, what actions offset from and what `joint_pos_rel` subtracts |
| `src/mjlab_microduck/robot/basketball.py` | the release's cream colourway |
| `src/mjlab_microduck/tasks/microduck_basketball_env_cfg.py` | the ball (`_ball_spec`), the twist ranges the sliders span, and the 55° fall tilt |
| `src/mjlab_microduck/robot/assets/basketball/basketball.{obj,png}` | the ball's look, generated by the repo's `scripts/make_basketball_assets.py` |
| `src/mjlab_microduck/tasks/mdp.py` | `reset_basketball`'s spawn and `basketball_fell`'s height margin |
| `policy.onnx` on the Hub at `6e61a73` (normalizer baked in) | the policy, `obs[1, 61]`, `h`, `c[1, 1, 256] -> actions[1, 14]`, `h`, `c` at 50 Hz, wrapped at build time (below) |

### What the policy reads

The same 61 values as Running, with all three twist slots driven:

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | `root_link_ang_vel_b` |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `joint_pos` | 14 | `joint_pos_rel`, against `STAND` |
| `joint_vel` | 14 | `joint_vel_rel`, one control step back |
| `actions` | 14 | the previous action |
| twist | 3 | the Forward, Sideways and Turn sliders, all at 0 to start |
| head, body | 4 + 6 | zeros: the actor is blind, so the body slot carries no ball state |

mjswan feeds a recurrent policy one carry tensor, while this LSTM keeps two 256-wide
states. A wrapper built at build time joins `h` and `c` into one `[1, 512]` carry. It
zero-pads the `[1, 128]` zero carry mjswan starts with and zeroes the state while
`is_init` holds. Each reset or policy switch clears the memory, and slider changes keep
it, as upstream's deploy contract asks. In onnxruntime the wrapped and original graphs
agree exactly over 400 steps and three resets.

### What differs from upstream

- **The servos are the XML's `<position>` actuators**, not BAM. Each gets a 20 ms
  `filterexact` filter for BAM's 3 to 6 physics-step command delay, which is shorter
  than Running's 30 ms. In a replay of the built scene under upstream's eval stress
  (twice the training command ranges, ±0.09 m/s pushes every 0.5 to 1 s, 60 s), 20 of
  20 seeds survive at 10 and 20 ms, 18 at 25 ms and 7 at 30 ms. At 40 ms all 10 seeds
  fall within 2 to 14 s, even unpushed.
- **The fall check reads the trunk, not the ball.** Upstream's `basketball_fell` ends an
  episode when the trunk is under 5 cm above the ball's top, more than 0.16 m off its
  centre sideways, or tilted past 55°. The single-entity trace env holds no ball, so the
  port keeps the 55° tilt (`bad_orientation`). The height margin becomes a 0.29 m trunk
  height (`root_height_below_minimum`), the same rule while the ball rolls on the floor.
  The sideways check is left out. In upstream's own eval it alone caused 3 of 92 falls,
  and a duck that far off then trips one of the other two within a few steps.
- **Collisions stay as the XML has them.** mjlab 1.3.0's `FULL_COLLISION` disabled none
  of the unnamed shell, leg and hip hulls, so they still hit the ball as in training.
  Its foot priority is left out: the ball's own priority decides every foot-ball
  contact, and foot-floor contacts come out the same either way.
- **Each collision hull gets its own copy of its mesh.** mjswan's renderer turns every
  drawn mesh's vertices in place, in the model the physics reads. The soles share
  their mesh with the visible soles, so they would collide turned 90°. Without the
  copies the duck falls off within 3 s, every episode.
- **One deterministic reset**: the duck faces +x with `STAND` joints, the ball at rest
  under it. Upstream draws xy, yaw, tilt and joint noise; the policy trained on random
  yaw, so its heading does not matter.
- **No domain randomization, pushes, observation noise, IMU misalignment or encoder
  bias, and no episode timeout.** The 0 to 1 step gyro and gravity lag is dropped, since
  zero is inside its range; `joint_vel` keeps its fixed one-step lag.
- **The ball is pinned at upstream's defaults**, where upstream reads its radius from
  `MICRODUCK_BB_BALL_RADIUS`. Its free joint is named `ball_free`, and its centre site,
  which nothing here reads, is left out. The training curriculum's ball hold is at 0 in
  play, a free ball, so there is none.

### How it behaves

It balances indefinitely. In a replay of the built scene, the browser's own reset ran
600 s, and 20 of 20 seeds with the play env's reset noise ran 120 s, both at zero
command and under random training-range commands. The trunk tilted 1.5° on average (at
most 4.4°) and stayed within 4 mm of the ball's apex on average (at most 23 mm). In
Chromium it ran 26 s at zero command, and 46 s at Forward 0.15 and Turn 0.5, without a
reset.

It barely follows the sliders. Whatever the command, it drifts sideways at about
0.045 m/s, so the ball wanders some 2.6 m a minute. Forward moves it about 0.01 m/s, and
Sideways leaves the drift where it is. All three sliders mostly set a slow spin
instead, Turn in the opposite sense: +0.5 rad/s turns the duck at -0.15 rad/s and -0.5
at +0.23. Forward +0.15 spins it at -0.10 rad/s and -0.15 at +0.21, against +0.005 at
zero. These numbers are with the XML servos standing in for BAM, so they do not show
how the policy steers on the actuator it trained with. Upstream reports a 1.26 rad/s
yaw-tracking error without a sign, which the replay matches (1.26). The policy only
balances: it does not climb onto the ball or get up from the floor, and a fall resets
it onto the apex.

The ball's 159k-face mesh adds about 4.8 MB to a 12.7 MB scene.

## Stilts

Policies from https://huggingface.co/HannesVonEssen/microduck-stilts (Apache-2.0) ·
servo friction model from https://github.com/Rhoban/bam (Apache-2.0)

The duck walks on stilts, one scene per height: 10, 15, 20, 25 and 50 cm, then 1.0, 1.4
and 2.0 m. Each height runs the policy trained for it. Forward, Sideways and Turn sliders
set the command, and Forward starts at the 0.15 m/s the policies were released at.

| From upstream | Used as |
|---|---|
| `src/mjlab_microduck/robot/stilt_constants.py` | the scenes: its `get_stilt_walk_spec(height_cm, blend=0.5)` runs at build time on `scene_walk.xml` and puts a stilt under each ankle, with a rounded 17 x 22 mm tip and 12 g + 1 g/cm of mass |
| `src/mjlab_microduck/tasks/microduck_stilt_env_cfg.py` (`Mjlab-Stilt-Flat-MicroDuck`) | the sliders: its training ranges (Forward -0.12 to 0.25 m/s, Sideways ±0.06 m/s, Turn ±0.35 rad/s) and its 0.15 m/s play speed as Forward's default; also the reset raised by the stilt length |
| `scene_walk.xml`'s `STAND` keyframe | the reset pose, what actions offset from, and what `joint_pos_rel` subtracts, as in Running |
| `<h>cm/policy.onnx` on the Hub at `371b7e0` (iterations 2,200 to 6,500, normalizer baked in) | the eight policies, `obs[1, 61] -> actions[1, 14]` at 50 Hz, trained at upstream's `c5fcc50`, whose stilt, env and robot files match the pinned commit's |
| `params/xl330/m6.json` in Rhoban/bam at `62bd8ce`, the commit upstream's `uv.lock` pins | the servo friction constants, copied into `stilts.py` |
| `LICENSE-HARDWARE` | each scene's 3D-model attribution, as in Running |

### What the policy reads

Running's 61 values in the same order. The only difference is the twist slot: all three
values come from the sliders, `[Forward, Sideways, Turn]`, where Running reads lateral and
yaw as zero. Head and body read zeros.

### What differs from upstream

- **BAM's gearbox friction is a control-rate event.** Training drives the servos through
  BAM, which rewrites each servo's `frictionloss` every physics step from the motor
  torque and the external load, to several times the XML's 0.0048 N·m while walking.
  Without it, the XML servos drop the 1.0, 1.4 and 2.0 m ducks within 10 s.
  - A traced `mode="interval"` event, `servo_load_friction`, writes BAM's motor-side
    terms (base, Stribeck and 0.267 times the servo's own torque) into
    `dof_frictionloss` once per control step.
  - The servo joints get BAM's stiff `solref_friction` and `solimp_friction`.
  - The terms on the external load are left out: they need `qfrc_bias` and
    `qfrc_constraint`, which mjswan does not serve.
  - The event shows as a **Servo gearbox friction** checkbox. Unchecked, the friction
    freezes where it was and the 2.0 m duck falls within seconds.
- **The command delay is a 25 ms filter**, 5 ms shorter than Running's. Training delays
  each command by 3 to 6 physics steps (15 to 30 ms) inside BAM. Every height walks with
  filters from 2.5 to 30 ms, but from 35 ms the 1.4 m duck falls, so 25 ms keeps a margin
  on both sides.
- **No current limit, voltage sag or random command delay** (all BAM's). There is no
  observation noise, IMU misalignment, encoder bias or random 0 to 1 step IMU lag either.
  `joint_vel` keeps its fixed one-step lag (`history_steps=(1,)`).
- **No domain randomization and no episode timeout.** Training randomizes servo friction
  and armature (±10 %), trunk and head CoM (±3 mm), trunk mass (±5 %), foot friction (0.7
  to 1.3) and supply voltage. The stilt env already drops pushes. The 70° fall reset is
  the velocity env's `fell_over`.
- **The reset is fixed.** The duck starts in `STAND` at x = y = 0 facing +x, with the root
  0.125 m plus the stilt length up. Upstream draws x and y within ±0.5 m, any yaw, and
  ±5 mm of height. From 50 cm up, the stilts cross below the duck and start 2 cm into
  each other, as in training.
- **`FULL_COLLISION` is rebuilt by hand** (`_common.full_collision`), since this mjlab no
  longer builds upstream's. Only the stilts collide, with the floor and with each other.

### How it behaves

- Every height walks forward from the reset and keeps walking: near the commanded speed
  up to 1.4 m, slower at 2.0 m.
- The heading drifts at every height, so the duck walks in arcs. The drift is the
  policies' own (they drift under BAM too), but the port adds to it at 1.4 and 2.0 m,
  where the duck walks in a tight circle.
- At zero command it steps in place and slowly turns.
- Turn steers at 10 to 25 cm, erratically at 50 cm and 1.0 m, and barely at 1.4 and
  2.0 m, where a full left turn at most straightens the walk. Sideways is weak everywhere.
- The 2.0 m gait is bistable: a nudge or a new command can switch it between a slow
  circling walk and a faster, straighter one.
- Once walking, the tall ducks shrug off a 0.1 m/s shove, but the first two seconds after
  a reset are knife-edge. A 0.001 rad joint offset at the start topples the 1.0 to 2.0 m
  ducks in two of five draws, under BAM too. They rely on the exact reset, and dragging
  one early can topple it.
- Upstream calls the 50 cm to 2.0 m policies extreme simulation results, and none of the
  eight was run on hardware.

## Desk Climb

The duck starts on the floor in front of a 27-tread alternating ladder, climbs it to a desk
whose top is 66 cm up, and tumbles onto the desktop on its back; once a foot is on the desk
and its trunk is past the edge, a second policy stands it up there. **Climb** runs the whole
sequence from a slightly randomized floor spawn and starts over on its own; **Get up**
starts from one of upstream's recorded handoffs on the desk.

Policies from https://huggingface.co/HannesVonEssen/microduck-climb (Apache-2.0) ·
ladder and desk from upstream's `experiments/desk-climb` (Apache-2.0 code; the ladder's
hardware design is CC BY-NC-SA 4.0, like the robot's meshes) ·
servo friction model from https://github.com/Rhoban/bam (Apache-2.0)

| From upstream | Used as |
|---|---|
| `experiments/desk-climb/source/src/mjlab_microduck/robot/microduck/scene.xml` | the robot (the experiment's own, with a passive mouth hinge), its floor and lights; its keyframes, which predate the hinge, are dropped |
| `STAND` in `src/mjlab_microduck/robot/microduck/scene_walk.xml` | the reset pose, what actions offset from, and what `joint_pos_rel` subtracts |
| `robot/floor_desk.py` and `floor_desk_assets/geometry.json`, patched by `training/geometry_patch.py` and the modules it imports, with `training/run.py`'s `ENDING_ROLE=above` and `LADDER_SHIFT=.06` | the rails, spine, clamps, bridges and desk (`spec_for`), as static geoms |
| `robot/ladder.py`'s `tread_layout` and `make_tread_spec`, moved as the resets of `geometry_base` and `geometry_shift` move them | the 27 live treads |
| `tasks/mdp.py`'s `reset_floor_desk` and `reset_stair_ladder` | the floor spawn and its noise |
| `training/evaluate_sequence.py` (`TRIGGER_MODE=supported_root`, `SWITCH_MARGIN=.04`), `RUNTIME.md`, `runtime/policy_pair.py` | when the get-up takes over and how: its smoothing, its lower gain, the previous raw action both actors read |
| `models/climber.onnx` and `models/getup.onnx` on the Hub at `1572387` (normalizers baked in) | the two actors, `obs[1, 61] -> actions[1, 14]` at 50 Hz, wired into one recurrent graph at build time |
| `evidence/switches.json`, entry 9 | where Get up starts |
| `params/xl330/m6.json` in Rhoban/bam at `62bd8ce`, the commit upstream's `uv.lock` pins | the servos' gearbox friction |

### What the policies read

Both actors take Running's 61 values with all 13 command slots at zero (upstream's
`blind_stair_zero_slots`). The graph built around them reads:

| Input | Width | Source |
|---|---|---|
| `actor` | 34 | `base_ang_vel`, `projected_gravity`, `joint_pos` against `STAND`, `joint_vel` one control step back |
| `supervisor` | 1 | 1.0 while a foot touches the desktop (two MuJoCo contact sensors) with the trunk 4 cm past the desk's near edge, 5 cm inside its other edges and above the desktop: upstream's simulator-only switch, which neither actor sees |
| `adapt_hx` | 128 | mjswan's recurrent carry: the switch latch, the previous raw action (the actors' slots 34 to 47) and the previous executed offset |

### What differs from upstream

- **One graph for both policies.** Picking a policy in the browser restarts the sim, so
  the climber, the get-up and the handoff are one recurrent ONNX built at build time. It
  latches the switch, feeds both actors the previous raw output, smooths the get-up's
  output from the last executed climbing offset (0.7 legs, 0.5 neck and head) as
  `RUNTIME.md` asks, and stands in for the get-up's 0.8 servo gain by pulling each target a
  fifth of the way back to the measured joint position. Fed upstream's recorded
  observations, it matches `runtime/policy_pair.py` exactly.
- **The servos** are the XML's `<position>` actuators (`kp` 0.55, ±0.96 N·m), not BAM.
  BAM's XL330 gearbox friction is written into each servo's `frictionloss` every control
  step from its own torque and speed, with BAM's stiffer friction constraint; the
  friction's terms on the external load are left out, as the browser serves neither
  `qfrc_bias` nor `qfrc_constraint`. The 3 to 6 physics step command delay is a 15 ms
  filter, half of Running's. Without the friction few dives reach the desktop: 9 of 128
  replayed attempts stand on the desk with Running's servos, 9 of 50 in Chromium with a
  0.64 N·m cap and a 20 ms filter. With it, 23 to 40 of 64 stand for filters from 10 to
  20 ms, 4 at 22.5 ms and none from 25 ms, where the dive off the top tread lands short.
  BAM's 1.75 A current limit (0.64 N·m) is left out: with the friction it drops a sixth
  to a quarter of the attempts off the ladder.
- **Spawn noise** is upstream's play reset as the ladder sees it: x exact, y ±1 cm
  (`reset_stair_ladder` places the ladder from the robot's own xy noise, so only
  `ladder_y_noise` moves one against the other), yaw ±5°, pitch ±2°, roll ±1°, servos
  ±0.03 rad clamped to their limits rather than 0.02 rad inside them. The root draw is a
  term of its own: the trace env's default root pose is the origin, where mjlab's
  `reset_root_state_uniform` would put it in the floor.
- **No domain randomization** (mass, CoM, armature, BAM friction scale, battery voltage
  and sag), no observation noise, IMU misalignment, IMU lag or encoder bias.
- **Treads 27 to 29 and the landing are left out**: `geometry_previous` parks them under
  the floor. The stage's hulls collide hidden under visual twins, as mjswan's renderer
  turns the vertices of every mesh it draws.
- **Episodes end.** Upstream evaluates 60 s without resets and only records its ladder
  flags (`bad_orientation`, `fallen`, `foot_fling`, `overstep`, `airborne`), which fire
  during the normal dive. Here an attempt ends when its trunk tips past 65° below the
  desktop (`fell`), at 60 s, or at 30 s when its trunk is still short of the switch line:
  draped over the desk edge, stuck on the ladder or back on the floor. In 256 replayed
  attempts neither cut one that would have stood.
- **Get up** starts with the servo filters at `STAND` and a zero carry: the browser cannot
  restore the climbing offset upstream seeds the smoothing with. It shares Climb's
  terminations rather than `train_getup.py`'s `below_table`.

### How it behaves

Over 105 attempts in Chromium (three 30-minute sessions), 95 reach the top of the ladder,
89 hand off to the get-up (median 18 s in, against upstream's 11.6 s), 40 (38%) stand on
the desk for 10 s and 10 fall, 8 of them within 3 s at the foot of the ladder. Upstream's
own evaluation (GPU and BAM, 4 × 64 attempts of 60 s) hands off 209 of 256 times and
stands 106 (41%). A browser-order replay of the built scene agrees over 256 attempts: 88%
reach the top, 82% hand off, 48% stand, 14% fall. The usual misses are in the get-up:
the duck lands on its back across the desk edge and wriggles onto the desktop over several
seconds, then the get-up leaves it lying there or rolls it back over the edge. The first
attempt after picking Climb is seeded and always the same: it tops the ladder at about
8 s, which the preview films, hands off at 21.8 s and stands from 24 s on. Get up stands
from its recorded handoff within 3 s and holds; replayed from all 50 of upstream's
recorded handoffs, it stands in 24 and falls in 4. The panel's "Servo gearbox friction"
box switches the friction off, which leaves most attempts draped over the desk edge.

## Chimney Climb

The duck side-steps into a 12.5 cm slot, wedges its back against one wall and its feet
against the other, climbs 3 m, works its way out under the overhanging walls onto a
platform and stands up there, about 25 s from the start. Three policies run that one
chain from different starts: **Enter, climb, exit** from the floor, **Climb** standing in
the slot, **Exit** wedged at the top.

Policies from https://huggingface.co/HannesVonEssen/microduck-chimney-climb (Apache-2.0)
at `1dea212` · get-up `alpha_stand.onnx` from https://github.com/pollen-robotics/microduck
(Apache-2.0), the checkout `microduck` already fetches, checked against the SHA-256 the
release pins

| From upstream | Used as |
|---|---|
| `src/mjlab_microduck/robot/microduck/scene.xml` with `robot/long_jump_robot.py`'s hulls and leg-fold pairs | the robot every chimney policy trains on, under `FULL_COLLISION` |
| `robot/corridor_stage.py`, `robot/exit_stage.py` | the slot, its stepped walls and the platform, set up as `scripts/reproduce_chimney_chain.py` does and reflected to -y |
| `scripts/reproduce_chimney_chain.py` | the start, the three handovers, the exit's reflection and the get-up's history |
| `tasks/approach_mdp.py`, `tasks/chimney_mdp.py`, `tasks/exit_mdp.py` | the enter handover test and each policy's body command |
| `tasks/symmetry.py` | the reflection's joint and IMU tables |
| `policy.onnx`, `enter/policy.onnx`, `exit/policy.onnx` on the Hub at `1dea212` | the climb (w17), enter (a11) and exit (x19), normalizer and joint-travel clamp baked in |
| each one's `config.json` | the joint order and offsets: HOME for enter, BRACE for climb and exit |
| the release's `manifest.json` | the get-up's SHA-256 |
| `policies/alpha_stand.onnx` in pollen-robotics/microduck at `590b986` | the get-up, offset by its own `default_joint_pos` |

### What the policy reads

Each policy is one recurrent graph holding the networks from its start on, built at build
time (`in_keys` `actor`, `chain`, `adapt_hx`); it assembles each network's 61 values.

| Group | Term | Width | Source |
|---|---|---|---|
| `actor` | `base_ang_vel`, `projected_gravity` | 3 + 3 | as in Running |
| | `joint_pos` | 14 | `joint_pos_rel` against HOME, shifted in the graph to each network's offset |
| | `joint_vel` | 14 | `joint_vel_rel`, one control step back |
| `chain` | `root_pos`, `root_quat`, `root_vel` | 3 + 4 + 3 | `root_link_pos_w`, `root_link_quat_w`, `root_com_lin_vel_w` |
| | `joint_vel` | 14 | `joint_vel_rel` now, for the enter handover's settle test |
| | `heights` | 3 | both ankles' and the jaw's world height: the feet over the platform, the head's carriage |
| | `on_platform` | 1 | a contact sensor, the robot against the platform |
| `adapt_hx` | carry | 128 | the phase, the step count and the step the feet cleared the platform, the landed steps, the acting network's previous output, last step's target |

Each network sees the 48 values of proprioception, its joint positions against its own
offset and its own previous output, then twist and head at zero and its body command:
enter, the slot's centre in its body frame, its yaw and the width; climb, the width;
exit, reflected, the goal 20 cm past the walls' end; the get-up, zeros.

### What differs from upstream

- **One graph instead of four policies and a script.** Picking a policy in the browser
  restarts the sim, so `reproduce_chimney_chain.py`'s controller runs inside the policy:
  its handovers (enter to climb on `in_handover_pose` and a mean joint speed under 0.35
  rad/s; climb to exit once the feet are 5 cm over the platform, on 22 then 26 degrees of
  tilt with 0.10 then 0.30 m/s of climb, or at 1.2 s; exit to get-up after two steps
  touching the platform past the walls' end, low or tipped), its action-history resets
  and its reflected exit. The networks are rebuilt node for node from their graphs, the
  weights as published: against an independent rewrite of the script on the four
  original files, the targets agree to 6e-5 rad over whole runs. The 1.2 s is counted in
  control steps, where a summed float clock fired a step early.
- **The servos are the XML's `<position>` actuators**, not BAM, and BAM's 15 to 30 ms
  command delay is one control step (20 ms) inside the graph, the first step after a
  reset holding the start pose. Not Running's filter: from standing in the slot, 12
  seeds, the climb reaches the platform 12 times with the delay, 0 times with no delay
  (back on the floor after a median 8 s), once with a 30 ms filter and 9 times with 20
  ms.
- **`joint_vel` reads one control step back**, as training does, the gyro and gravity
  none. **No domain randomization, observation noise, IMU misalignment or encoder bias,
  and no terminations:** the chain runs until a reset, as upstream's does.
- **The platform's near edge is 7 cm clear of the slot**, `exit_stage`'s default and the
  exit's training value, where the video's is flush. Flush, the climb catches under the
  platform's lip at 2.8 m in 5 to 6 of 48 runs, against 2 here.
- **The platform is 1 m wide and runs 60 cm further**, where the release's is 50 by 85
  cm: there the get-up walks off it in 4 to 5 of 48 runs, here in none. The exit still
  aims 20 cm past the walls' end.
- **The start is `place_outside`'s pose without its spread** (1 cm of x, 0.2 rad of
  yaw): one keyframe, so every reset runs the same chain.
- **The get-up keeps the chain's official contract**: its history starts at its pose,
  its offset is its own `default_joint_pos` and its unbounded output goes through the
  joint-range clamp. Enter (a11) trained with unbounded actions and runs bounded, as in
  upstream's chain.
- **The Climb and Exit policies are the port's**: the same chain from standing in the
  slot, the state enter hands over, and from the middle of `reset_exit_state`'s wedged
  spawn, reflected.

### How it behaves

In Chromium the chain from the keyframe is deterministic and completes: the duck climbs
from 1.6 s, is above 3 m at 18.6 s and stands on the platform from 25 s, still standing
at 43 s. Climb and Exit complete there too. The braced climb tips the trunk up to 83
degrees from upright, so the preview films with `upright=False`.

Replayed in the browser's order, the build ends standing on the platform at 45 s in 21
of 24 runs with the start moved by up to 2 mm and 2 mrad, and in 20 of 24 over
upstream's own start spread. It hands over at a median 1.6, 20.7 and 23.6 s, against
1.74, 25.08 and 27.80 s in upstream's selected run; upstream completed 2 of 4
exploratory seeds. The misses: the exit losing its brace and falling (3 of 48, twice out
of the slot's open end), the get-up stuck lying down (2), the climb caught under the
platform's lip (2). Climb completes 23 of 24 and Exit 24 of 24 from moved starts, Exit
16 of 24 over the exit's whole training spawn.

The climb never slows near the top as upstream's does: every handover to the exit comes
at the 1.2 s limit, the trunk 30 to 34 cm over the platform where the exit trained from
9.5 to 23.5 cm. None of these policies was run on hardware.

## Long Jump

The Microduck stands near the edge of a block 35 cm high, hops back a few centimetres,
crouches, jumps a 30 cm gap and lands on a second block 25 cm lower, then stands there.
One policy does the whole jump and has no controls: it is blind, and the course reaches
it as a constant that matches the blocks baked into the scene. Press reset to jump again.

| From upstream | Used as |
|---|---|
| `src/mjlab_microduck/robot/microduck/scene.xml` | the robot (`robot_allcollisions.xml` and its 14 `<position>` servos) and the floor |
| `src/mjlab_microduck/robot/long_jump_robot.py` | run at build time: `add_full_collision_geoms`, a convex hull on every visible part, and `add_leg_fold_pairs`, 24 thigh/shin and hip/thigh contact pairs |
| `FULL_COLLISION` in `microduck_constants.py` | restated as `_common.full_collision`, since it no longer builds under this mjlab |
| `src/mjlab_microduck/robot/platform_stage.py` | the two blocks: size, colour, contact parameters, and where they sit for a gap and a drop |
| `src/mjlab_microduck/tasks/platform_jump_mdp.py` | restated: `pj_command_obs` (the course input), the standing branch of `reset_platform_jump_state` (the spawn) and `pj_fell` (the resets) |
| `policy.onnx` on the [Hub](https://huggingface.co/HannesVonEssen/microduck-long-jump) at `fdcdc78` (iteration 15,250, normalizer and joint-travel clamp baked in) | the policy, `obs[1, 61] -> actions[1, 14]` at 50 Hz |
| `config.json` there | the joint order and offsets (`HOME_FRAME`), and the release profile's course: `MICRODUCK_PJ_GAP=0.30`, `MICRODUCK_PJ_DROP=0.25` |

### What the policy reads

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | `root_link_ang_vel_b` |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `joint_pos` | 14 | `joint_pos_rel`, against `HOME_FRAME` |
| `joint_vel` | 14 | `joint_vel_rel`, one control step back |
| `actions` | 14 | the previous action |
| twist, head | 3 + 4 | zeros |
| `course` | 6 | `[gap / 0.3, drop / 0.3, 0, 0, 0, 0]`, here `[1.0, 0.833, 0, 0, 0, 0]` |

### What differs from upstream

- **The servos are the XML's `<position>` actuators** with a 30 ms `filterexact` filter
  standing in for BAM's 3 to 6 physics-step command delay, as in Running. In plain MuJoCo
  the jump needs 20 to 40 ms: with no filter or 10 ms it falls short into the pit, with
  50 ms or more it falls.
- **One course, fixed.** Upstream moves both blocks, mocap bodies, to each episode's gap
  and drop. Here they are static geoms at the release profile's 30 cm gap and 25 cm drop,
  and the course input is the matching constant: mjswan cannot bind one value to both a
  block's pose and an observation, so a slider would show one course and tell the policy
  another. With the input zeroed the robot falls short.
- **One spawn**, the middle of the play config's draws: the trunk 8.5 cm behind the edge
  of A and 11.8 cm above it, level, facing B, with no joint noise. Upstream draws the edge
  distance from 3 to 10 cm, ±2 cm sideways, ±5° of yaw, ±3° of tilt and 0.05 rad of joint
  noise.
- **The resets are tilt and ankle height.** `pj_fell` ends an episode past 70° of tilt,
  when the head touches anything, or when a sole touches the floor, through contact
  sensors the trace env does not have. Here `fell_over` is the same 70° test and `in_pit`
  fires when either ankle body is below 6 cm: it sits 2.3 cm above the sole, so it reads
  about 2 cm on the floor and at least 11 cm on B.
- **Head contact is not a reset.** `fell_over` usually follows 0.05 to 0.9 s later; in 4
  of 200 slightly perturbed replays the robot recovered and stood on B instead.
- **No timeout** (4 s in training, 12 s in the release render): the policy keeps
  standing on B.
- **Twist reads zero**, where training samples a ±0.01 m/s, ±0.05 rad/s nuisance
  command; head reads zero as in training. No domain randomization, observation noise,
  encoder bias or IMU misalignment, and the 0 to 1 step random lag on the gyro and
  gravity is dropped: 0 is inside it.
- **`HOME_FRAME` from `config.json`** is the reset pose and the action offset, rather
  than `scene.xml`'s `STAND`, which it rounds to 4 places.

### How it behaves

Filmed in Chromium, the reset plays out as in native MuJoCo: a small hop back around
0.1 s, takeoff at about 0.35 to 0.4 s, touchdown on B at 0.70 s with the trunk dipping to
16 cm, then standing at 21 cm. Over 24 s the browser's trunk height stayed within 4 mm
of a native replay of the built scene, with no reset.

Replays of the built scene in the browser's order, 12 s each unless stated:

| Start | Lands on B | Still standing |
|---|---|---|
| the reset, with 1e-7 relative noise on every action | 100/100 | 100/100, and 40/40 at 40 s |
| 1 mrad of joint noise | 400/400 | 308/400 (77%; batches of 200 at 72% and 82%) |
| the play config's reset draws | 197/200 | 149/200 (75%) |

The failures topple 1.0 to 2.0 s in, after landing, or for the three short jumps tip
into the pit at 0.65 s; either ends in a reset and another jump. Upstream reports 29 of
32 episodes surviving 12 s with BAM and full randomization. Standing on B is past the
4 s training episode: the robot turns slowly, about 60° in 40 s, and stays on B. The
policy was never run on hardware.

## Backflip

Policy from https://huggingface.co/HannesVonEssen/microduck-backflip (Apache-2.0)

Microduck stands on a platform 0.8 m above a crash mat, with its back to the edge. It bounces twice, jumps backwards, turns over once in the air and lands on the mat. There are no controls. Every 12 s the episode times out and the same flip plays again. Upstream calls this a simulation-only policy: **landings may break the robot**, and it was never run on hardware.

| From upstream | Used as |
|---|---|
| `src/mjlab_microduck/robot/microduck/scene.xml` | the robot with every collision mesh, the floor, and its `STAND` keyframe: the reset pose, what actions offset from and what `joint_pos_rel` subtracts |
| `src/mjlab_microduck/robot/long_jump_robot.py` (`add_full_collision_geoms`, `add_leg_fold_pairs`) | a convex hull on every visible part, plus the thigh/shin and hip/thigh contact pairs: the robot the flip env trains on |
| `src/mjlab_microduck/robot/flip_stage.py` | the platform at `platform_center(0.8)` and the mat at `mat_center()`: boxes, friction and contact softness |
| `src/mjlab_microduck/tasks/flip_mdp.py` (`reset_flip_state`, `flip_command_obs`) | the spawn and the body command slot |
| `scripts/parkour_release.py` (`PROFILES["backflip"]`) | the release profile: backwards, a 0.8 m drop, no mat mesh, hops not fatal |
| `policy.onnx` on the Hub at `0aff1c7` (run f21, iteration 12000, normalizer and joint-range clamp baked in) | the policy, `obs[1, 61] -> actions[1, 14]` at 50 Hz |

### What the policy reads

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | `root_link_ang_vel_b` |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `joint_pos` | 14 | `joint_pos_rel`, against the `STAND` pose |
| `joint_vel` | 14 | `joint_vel_rel`, one control step back |
| `actions` | 14 | the previous action, already clamped to the joint range |
| `command` | 3 | zeros |
| `head_command` | 4 | zeros |
| `body_command` | 6 | `[0.5, -1, 0, 0, 0, 0]`: `flip_command_obs` for a 0.8 m drop, `(0.8 - 0.7) / 0.2`, done backwards |

### What differs from upstream

- **The servos are the XML's `<position>` actuators with a 45 ms filter**, not BAM with a 15 to 30 ms command delay. 160 randomized starts give these results:
  - 30 ms, running's value: lands the flip just as well, but only about half the starts are still standing at 12 s.
  - 40 ms: 109/160 still standing at 12 s.
  - 45 ms: 144/160 still standing at 12 s.
  - MuJoCo's own actuator `delay` did worse, alone or with a filter.
- **One fixed start**, at the centre of `reset_flip_state`'s draws: 7.5 cm from the edge, square to it, joints at `STAND`. There is no joint noise, domain randomization, observation noise, or IMU lag and misalignment.
- **`fell` is a stateless stand-in.** It fires when the trunk is tipped past 60° within 0.2 m of the mat top or lying on the platform, or when the trunk sinks 0.1 m below the mat top. Upstream's `flip_fell` is a contact phase machine with a latch: head on any surface, a non-foot part touching the mat or floor in flight, or lying down after landing. Its contact sensors do not trace into this scene. As a result, a landing that touches the mat with the right shin servo goes on here, while upstream ends it as a crash. The start this scene uses lands that way, then stands.
- **12 s episodes**, the release render's length, not training's 3.5 s. `out_of_terrain_bounds` and `nan_state` are dropped, and `hopped` is off, as in the release profile.
- **The stage is fixed world geometry** rather than mocap bodies moved at every reset, and the mat is its collision box, as in the release profile.
- **Collision is mjlab 1.3.0's `FULL_COLLISION`, applied by hand**, which this mjlab can no longer build: only `*_collision` geoms collide, condim 1, with the feet at condim 3, priority 1 and friction 1. Each hull gets a copy of its mesh, so mjswan's renderer does not turn it.

### How it behaves

From the reset it bounces twice, leaves the platform at 0.48 s, turns 318° in the air, lands on the mat at 0.88 s and stands until the 12 s time out restarts it. In Chromium the same flip repeats.

The same built scene was run outside the browser in the browser's order, from 160 randomized starts drawn as upstream's play env draws them:

| Result | Starts |
|---|---|
| Finish the flip upright on the mat | 156 |
| Touch down on the feet alone | 116 |
| Still standing at 12 s | 144 |

The four failed flips turn only 220° to 280°. The other falls are drift-offs from the mat edge between 5 and 12 s, which upstream documents too ("not a stable idle policy"); its own 32-run evaluation had 24 reach 12 s.

## License

The code and the policies are Apache-2.0. **The 3D files are not**: microduck-playground
licenses its hardware design files, the robot meshes from pollen-robotics/microduck_rl
among them, under Creative Commons BY-NC-SA 4.0 (`LICENSE-HARDWARE`, `NOTICE`), as
`microduck` does. NonCommercial and ShareAlike both bear on redistributing the compiled
scene, so the build declares `LICENSE-HARDWARE` as a scene attribution and `mjswan
publish` warns on it. Publishing a copy is the author's call under those terms.
