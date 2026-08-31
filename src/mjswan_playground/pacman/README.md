# PAC-MAN Dodgeball (`pacman`)

Source: https://github.com/lzyang2000/perceptive_cbf_rl ·
project page: https://lzyang2000.github.io/perceptive_cbf_rl

> Lizhi Yang, Junheng Li, Aaron D. Ames.
> [PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball](https://arxiv.org/abs/2607.28623).
> arXiv:2607.28623, 2026.

A Unitree G1 that dodges thrown balls from a head-mounted depth camera alone, trained
with a per-link control-barrier-function reward (Link-CBF) and an adversarial motion
prior. Both checkpoints the paper deploys, each in the scene it was trained in:

| Scene | Policy | Reads |
|---|---|---|
| `Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat` | `Link-CBF Dodge` — the deployed dodge policy, 19 of 20 hand throws dodged on hardware | 960: proprio + a 16×9 ball-only depth image |
| `Unitree-G1-AMP-Flat` | `AMP Walk` — the locomotion half of the hardware walk↔dodge mode switch | 384: proprio only, steered by sliders |

The dodge scene opens first and throws a ball every 1–4 s. The policy sees the ball as
that depth image and nothing else — no position, no velocity, no oracle.

## Run

```sh
uv run msp run pacman
```

The build clones the PAC-MAN repository into `.cache/` at a pinned commit; set
`MJSWAN_PACMAN_ROOT` to point at a checkout you already have.

Unlike [`husky`](../husky/README.md), this task **imports** upstream. The scenes are not
shipped XML — the robot, its collision geometry and the terrain are assembled in Python by
the repo's own mjlab task configs — so the checkout goes on `sys.path` and
`add_scene_mjlab` builds each scene from its registered task ([`upstream.py`](upstream.py)).

| From upstream | Used as |
|---|---|
| its two registered mjlab tasks (`play=True`) | the scenes, the proprio observations, the action terms, the terminations, the startup randomization, the viewer, and every number the throw uses |
| `deploy/ckpts/dodge_link_cbf.onnx`, `deploy/ckpts/walk_policy.onnx` | the policies — the same checkpoints the robot ran, normalizers baked in |
| `deploy/common/g1_deploy_constants.py` | the joint order, the rest pose, and `DEFAULT_FRAME_OFFSETS`, the depth stack's look-back |

Its license is [MIT](https://github.com/lzyang2000/perceptive_cbf_rl/blob/main/LICENSE).
The retargeted motion clips the AMP prior trained against carry the source dataset's own
terms; this demo ships none of them.

## What the policies read

Both start from the same proprio group — six of mjlab's own MDP functions at
`history_length=4`, no per-term scaling:

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | the `imu_ang_vel` sensor |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `command` | 3 | the `twist` command |
| `joint_pos` / `joint_vel` | 29 each | `joint_pos_rel` / `joint_vel_rel` |
| `actions` | 29 | the previous action |

96 per frame × 4 frames = 384, the walk checkpoint's whole input. The dodge checkpoint
takes 576 more: the 144-pixel image at look-back offsets `(0, 3, 8, 18)`, newest frame
first. The proprio terms carry a plain frame count because mjlab and mjswan both stack
history oldest frame first; the image's window is neither dense nor chronological, so it
names its offsets instead. Upstream's dodge actor reads its two groups concatenated, while
mjswan feeds one vector per ONNX input — so the task adapts upstream's own group and
appends the image as one more term.

## The image, without a camera

In training the image is the head camera's depth **and segmentation** channels: mjlab
renders both, keeps the pixels whose geom is the ball, and sets the rest to `far`. That is
what the hardware's EfficientTAM segmenter produces from a ZED, so the sim-to-real gap
collapses into the perception layer.

A browser has no renderer, and mjswan's slot reader serves entity fields, sensors,
raycasts and contacts — not frames. But with a fixed camera and one sphere, a ball-only
image *is* its ray-sphere intersections, which is what upstream's own browser demo computes
([`web/src/depth.js`](https://github.com/lzyang2000/perceptive_cbf_rl/blob/web-demo/web/src/depth.js)).
[`terms.ball_depth`](terms.py) does the same in torch, so it traces to ONNX like any other
term: two `framepos` / `framequat` sensors give the camera pose, the ball's root position
is an entity slot, and 144 × 3² rays are min-pooled per pixel.

Two details decide whether the numbers match, both measured against upstream's rendered
frames rather than assumed:

- **Perpendicular depth, not ray distance.** Projecting the hit onto the optical axis
  agrees with the rendered frame to a median of 0.017 m; the distance along the ray (what
  upstream's browser demo stores) is off by 0.121 m at the median and 0.45 m at the edge
  of the 85° field of view. Perpendicular depth is also what a stereo depth map carries,
  i.e. what the deployed policy reads.
- **3×3 sub-rays, min-pooled.** One ray per pixel centre misses 42 % of the pixels the
  renderer marks as ball — a 0.076 m ball subtends 3° where a pixel spans 5.3°. The robot
  min-pools too: the ZED's depth is masked at full resolution, then pooled down to 9×16,
  so a sub-pixel ball still registers.

Over 400 control steps of live throws, every pixel the renderer marks as ball is marked
here too (218/218), to a median of 0.016 m and a maximum of 0.078 m — about one ball
radius, the residue of sampling a sphere at different points inside a pixel. It also
lights up ~1.2 pixels per frame the renderer does not, and catches the ball at ranges
where the renderer's single sample misses it: that is the min-pool, i.e. the deployed
behavior.

Self-occlusion is not modelled — an arm in front of the ball segments as the arm in
training, while here the ball stays visible. That only hands the policy a cleaner view
than it trained on (it trained with per-pixel and whole-ball dropout), and upstream's demo
makes the same trade.

## What differs from upstream

- **The throw is an interval event, not a step event.** Upstream's `throw_ball_on_dwell`
  runs every step with a per-env countdown; mjswan's event modes are startup, reset and
  interval. [`terms.throw_ball`](terms.py) keeps the same launch geometry — read out of
  upstream's own params, both threat types mixed 50/50 — on the 1–4 s interval its play
  config already throws at. Its aim jitter draws uniformly rather than normally, because
  `sample_uniform` is what the build-time RNG spy records.
- **The dodge scene has no velocity sliders.** The velocity command is still in the
  vector — it is upstream's term — but it reads **zero**, because that is the deployed
  dodge mode: on hardware the policy ignores the operator's velocity outright
  (`deploy/policy/dodge_policy.py`). Upstream's sim-play command is a goal tracker in a
  ball-avoiding CBF filter, its goal pinned 0.5 m behind the robot so it backpedals — a
  training-time construct the robot never runs.
- **The walk scene is the operator's.** `UniformVelocityCommand` resamples a twist every
  3–8 s, zeroes 5 % of envs into standing and puts a quarter under heading control; the
  browser gets three sliders over the play config's own ranges instead (forward
  −1.5…3.0 m/s, starting at 0.5) — the same 3-vector the policy reads.
- **No reference-state initialization.** Upstream resets the robot into a frame of its AMP
  dataset (`init_motion_loader`, `reset_from_motion`); the loader writes nothing to the
  simulation, so there is no graph to trace and no dataset for the sampler. A reset lands
  on the entity's `KNEES_BENT_KEYFRAME` instead — the pose the action offsets are relative
  to anyway — and still parks the ball aside, which is upstream's own `reset_dodge_state`.
- **One ball size.** Upstream randomizes the radius per episode (7.5–12.5 cm) through
  mjlab's `dr.geom_size`, which mjswan has no browser-side writer for. The demo keeps the
  entity's own 7.62 cm — the real dodgeball the hardware policy faced.
- **A hit is the contact sensor's verdict alone.** Upstream also ORs in a
  velocity-discontinuity check for a ball that tunnels through a link in one step, using
  the previous step's ball velocity kept on the env — no graph to trace. `collapsed_crouch`
  goes for the same reason (a sustained-posture counter); `bad_base_height` (0.45 m) still
  ends a fall immediately.
- **One randomized robot, not a population.** The startup randomization survives the
  build — foot friction (0.3–1.2, shared across the foot geoms), a ±2.5 cm / ±3 cm CoM
  offset on `torso_link`, and the runtime's encoder bias — but the browser draws it once
  from its seeded PRNG and keeps it, where upstream draws per env across thousands.
- **`randomize_terrain` is a no-op**, as mjswan records in the bundle: it re-draws which
  sub-terrain an env spawns on, and the browser has one baked plane.

## How it behaves

Fed these observations in a live mjlab env, the dodge checkpoint stays upright through 30 s
of throws — base height 0.72 m on average, never below 0.51 — and is hit once in roughly
twelve throws. One rollout is not a benchmark (upstream's `dodge_benchmark.py` is), but it
behaves like the policy in the paper, not like one reading a broken image. In headless
Chromium it stands between throws, braces and sidesteps as the ball arrives, and resets
when one connects; the walk policy walks upright for 24 s at the default 0.5 m/s.
