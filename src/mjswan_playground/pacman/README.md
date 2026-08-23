# PAC-MAN Dodgeball (`pacman`)

Source: https://github.com/lzyang2000/perceptive_cbf_rl ·
project page: https://lzyang2000.github.io/perceptive_cbf_rl

> Lizhi Yang, Junheng Li, Aaron D. Ames.
> [PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball](https://arxiv.org/abs/2607.28623).
> arXiv:2607.28623, 2026.

A Unitree G1 that dodges thrown balls from a head-mounted depth camera alone, trained with
a per-link control-barrier-function reward (Link-CBF) and an adversarial motion prior. Both
checkpoints the paper deploys, each in the environment it was trained in:

| Scene | Policy | Reads |
|---|---|---|
| `Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat` | `Link-CBF Dodge` — the deployed dodge policy, 19 of 20 hand throws dodged on hardware | 960: proprio + a 16×9 ball-only depth image |
| `Unitree-G1-AMP-Flat` | `AMP Walk` — the locomotion half of the hardware walk↔dodge mode switch | 384: proprio only, steered by sliders |

The dodge scene opens first and throws a ball every 1–4 s. The policy sees it as the depth
image and nothing else — no ball position, no velocity, no oracle.

## Run

```sh
uv run mjswan-playground run pacman
```

The build clones the PAC-MAN repository into `.cache/` at a pinned commit; set
`MJSWAN_PACMAN_ROOT` to point at a checkout you already have.

Unlike [`husky-skater`](../husky_skater/README.md), this task **imports** upstream. The
scenes are not shipped XML: the G1's actuators, collision geometry and terrain are
assembled in Python by the repo's own mjlab task configs, so the checkout goes on
`sys.path` and `add_scene_mjlab` builds each scene from its registered task (see
[`upstream.py`](upstream.py) — this repository has a `src/` of its own, and upstream runs
from source under that same top-level name).

| From upstream | Used as |
|---|---|
| its two registered mjlab tasks (`play=True`) | the scenes, the proprio observations, the action terms, the terminations, the startup randomization, the viewer, and every number the throw uses |
| `deploy/ckpts/dodge_link_cbf.onnx`, `deploy/ckpts/walk_policy.onnx` | the policies — the same checkpoints the robot ran, their observation normalizers baked in |
| `deploy/common/g1_deploy_constants.py` | the joint order, the rest pose, and `DEFAULT_FRAME_OFFSETS`, the depth stack's look-back |

Its license is [MIT](https://github.com/lzyang2000/perceptive_cbf_rl/blob/main/LICENSE).
The retargeted motion clips the AMP prior trained against carry the source dataset's own
terms; this demo ships none of them.

## What the policies read

Both start from the same proprio group — six of mjlab's own MDP functions at
`history_length=4`, and nothing else, no per-term scaling:

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | the `imu_ang_vel` sensor |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `command` | 3 | the `twist` command |
| `joint_pos` / `joint_vel` | 29 each | `joint_pos_rel` / `joint_vel_rel` |
| `actions` | 29 | the previous action |

96 per frame × 4 frames = 384, the walk checkpoint's whole input. The dodge checkpoint
takes 576 more: the 144-pixel image at look-back offsets `(0, 3, 8, 18)`, newest frame
first.

mjlab flattens a term's history **oldest frame first**, and so does mjswan's
`history_length` since 0.9.2, so the proprio terms carry their count across as a count.
The image's window is neither dense nor chronological, so it names its offsets instead.
Upstream's dodge actor reads its two groups `("actor", "depth")` concatenated; mjswan
feeds one vector per ONNX input, so the task adapts upstream's own group and appends the
image as one more term rather than restating its six.

### Why the dodge scene has no velocity sliders

The policy still reads a three-number velocity command — that term is upstream's, and it
is in the vector whether or not anyone can move it. What it reads is **zero**, with no
controls attached, because that is the deployed dodge mode: on hardware the dodge policy
ignores the operator's velocity outright (`deploy/policy/dodge_policy.py`). Upstream's
sim-play command is a goal tracker wrapped in a ball-avoiding CBF filter — the goal pinned
0.5 m behind the robot so it backpedals — and that is a training-time construct the robot
never runs. The walk scene, where steering *is* the point, keeps the sliders.

## The image, without a camera

In training the image comes from the head camera's depth **and segmentation** channels:
mjlab renders both, keeps the pixels whose geom is the ball, and sets the rest to `far`.
That is the representation the hardware's EfficientTAM segmenter produces from a ZED, so
the sim-to-real gap collapses into the perception layer.

A browser has no such render, and mjswan's slot reader serves entity fields, sensor
windows, raycasts and contacts — not rendered frames. But with a fixed camera and one
sphere, a ball-only image *is* its ray-sphere intersections, which is what upstream's own
hand-written browser demo computes
([`web/src/depth.js`](https://github.com/lzyang2000/perceptive_cbf_rl/blob/web-demo/web/src/depth.js)).
[`terms.ball_depth`](terms.py) does the same in torch, so it traces to ONNX like any other
term: two `framepos` / `framequat` sensors on the camera give its pose, the ball's root
position is an entity slot, and 144 × 3² rays are min-pooled per pixel.

Two details decide whether the numbers match, and both were measured against upstream's
rendered frames rather than assumed:

- **Perpendicular depth, not ray distance.** Projecting the hit onto the optical axis
  agrees with the rendered frame to a median of 0.017 m; the distance along the ray is off
  by 0.121 m at the median and 0.45 m at the edge of the 85° field of view, growing with
  the pixel's angle off the axis. (Upstream's browser demo stores the ray distance, on the
  documented assumption that this is what `mujoco_warp` reports.) It is also what a stereo
  camera's depth map carries, which is what the deployed policy reads.
- **3×3 sub-rays, min-pooled.** One ray per pixel centre misses 42 % of the pixels the
  renderer marks as ball — a 0.076 m ball subtends 3° where a pixel spans 5.3°. Min-pooling
  sub-rays is also what the robot does: the ZED's full-resolution depth is masked and then
  min-pooled down to 9×16, so a sub-pixel ball still registers.

Self-occlusion is not modelled: an arm in front of the ball segments as the arm in
training, while here the ball stays visible. That only hands the policy a cleaner view than
it trained on, and it trained with per-pixel and whole-ball dropout. Upstream's demo makes
the same trade.

## What differs from upstream

- **The throw is an interval event, not a step event.** Upstream's `throw_ball_on_dwell`
  runs every step and holds a per-env countdown; mjswan's event modes are startup, reset
  and interval. [`terms.throw_ball`](terms.py) is the same launch geometry — read out of
  upstream's own params, both threat types mixed 50/50 — on the 1–4 s interval its play
  config already throws at. Its aim jitter draws uniformly rather than normally, because
  `sample_uniform` is what the build-time RNG spy records.
- **No reference-state initialization.** Upstream resets the robot into a frame of its AMP
  dataset (`init_motion_loader` installs the clips, `reset_from_motion` samples them). The
  loader writes nothing to the simulation, so there is no graph to trace, and the sampler
  has no dataset without it — both are dropped, and a reset lands on the entity's
  `KNEES_BENT_KEYFRAME`, the pose the action offsets are relative to anyway. In the dodge
  scene the reset also parks the ball aside, which is upstream's own `reset_dodge_state`,
  traced as it is.
- **One ball size.** Upstream randomizes the radius per episode (7.5–12.5 cm) through
  mjlab's `dr.geom_size`, which mjswan has no browser-side writer for. The demo keeps the
  entity's own 7.62 cm — the size of the real dodgeball the hardware policy faced.
- **A hit is the contact sensor's verdict alone.** Upstream ORs in a velocity-discontinuity
  check for a ball that tunnels through a link in one step; it compares against the
  previous step's ball velocity, which it keeps on the env, so there is no graph to trace.
  `collapsed_crouch` goes for the same reason — a sustained-low-posture counter — and
  `bad_base_height` (0.45 m) still ends a fall immediately.
- **One randomized robot, not a population.** The play configs' startup randomization
  survives the build — foot friction (0.3–1.2, shared across the foot geoms) and a
  ±2.5 cm / ±3 cm CoM offset on `torso_link`, drawn once from the browser's seeded PRNG,
  plus the encoder bias the runtime applies from the policy config. Upstream draws them per
  env across thousands; a browser session gets one draw and keeps it.
- **`randomize_terrain` is a no-op**, as mjswan records in the bundle: it re-draws which
  sub-terrain an env spawns on, and the browser has one baked plane.
- **The walk scene is the operator's.** `UniformVelocityCommand` resamples a twist every
  3–8 s, zeroes 5 % of envs into standing and puts a quarter under heading control; the
  browser gets three sliders over the play config's own ranges instead (forward
  −1.5…3.0 m/s, starting at 0.5), which is the same 3-vector the policy reads.

## Fidelity

- **The depth image**, against upstream's rendered `BallOnlyDepthObs` over 400 control
  steps of live throws: every pixel the renderer marks as ball is marked here too
  (218/218), and on those pixels the depth agrees to a median of 0.016 m and a maximum of
  0.078 m — about one ball radius, the residue of sampling a sphere at different points
  inside a pixel. It also lights up ~1.2 pixels per frame the renderer does not, and
  registers the ball at ranges where the renderer's single sample misses it: that is the
  min-pool, i.e. the deployed behavior.
- **Proprio**: the five traced terms reproduce the live mjlab env exactly — `max |Δ| = 0`
  over 32 control steps, every term, in both scenes (`mjswan.compile.run_parity`). So do
  the two event graphs the dodge scene writes the ball with: the reset that parks it and
  the throw that launches it, `max |Δ| = 0` over 16 fresh draws each.
- **The articulation the checkpoints were exported against**: the action scale mjswan
  resolves from the model matches upstream's deployed `ACTION_SCALE` to `1.7e-08`
  (float32 rounding), the rest pose matches `DEFAULT_POS` exactly, and the joint order is
  identical to `POLICY_JOINT_NAMES`.
- **Behavior**: fed these observations in a live mjlab env, the dodge checkpoint stays
  upright through 30 s of throws — base height 0.72 m on average, never below 0.51 — and is
  hit once in roughly twelve throws. One rollout is not a benchmark (upstream's
  `dodge_benchmark.py` is), but the policy behaves like the one in the paper rather than
  like a policy reading a broken image. In headless Chromium it stands between throws,
  braces and sidesteps as the ball arrives, and the episode resets when one connects; the
  walk policy walks upright for 24 s at the default 0.5 m/s. No console errors either way.

## mjswan

Needs `mjswan >= 0.9.2b0`, for the three fidelity fixes in
[#101](https://github.com/ttktjmt/mjswan/pull/101) — all three found here:

- an event's root write went to the model's first free joint, so every throw launched the
  *robot* and left the ball parked. The write now lands on the entity it was made on:
  this task's throw serializes as `entity: "ball"`, resolved in the browser through the
  ball's own joint-name prefix.
- a task's *subclass* of an mjlab observation group was not adapted at all, and upstream's
  dodge group is one, so the build failed on the first mjswan-only field it read;
- mjlab's dense observation history reached the runtime reversed. `history_length` now
  means mjlab's chronological stack outright, which is why the proprio terms above carry
  a count rather than the offsets an earlier engine needed.
