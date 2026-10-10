# G1 Velocity Walking (`unitreerl`)

<img src="../../../assets/unitreerl.gif" width="480" alt="G1 Velocity Walking preview"/>

Source: https://github.com/unitreerobotics/unitree_rl_mjlab (Apache-2.0; the G1 model is
Unitree's, BSD-3-Clause)

A Unitree G1 (29 DoF) walking to velocity commands with the checkpoint Unitree ships for
deployment in its own mjlab repository. The command is resampled every 3 to 8 s, as
upstream's play config draws it, and mjlab's joystick panel can take over.

## Run

```sh
uv run msp run unitreerl
```

The build clones the repository into `.cache/` at a pinned commit; set
`MJSWAN_UNITREERL_ROOT` to point at a checkout you already have. Upstream runs from source
as the top-level package `src`, so [`upstream.py`](upstream.py) puts the checkout first on
`sys.path` and imports `src.tasks.velocity.config.g1`, which registers the task.

| From `unitree_rl_mjlab` | Used as |
|---|---|
| `Unitree-G1-Flat` (`play=True`) | the scene, the observations, the action term, the terminations and the startup randomization |
| `deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx` | the policy (98 → 29), already exported; its mjlab metadata gives the joint order and the rest pose |

That checkpoint is the one upstream's deploy config pairs with `params/deploy.yaml`. It
reads no height scan, so it belongs to the flat task.

## What the policy reads

Seven terms, one frame, no history:

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | the `imu_ang_vel` sensor |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `command` | 3 | the `twist` command |
| `phase` | 2 | sin and cos of a 0.6 s gait clock, zero while the command is below 0.1 |
| `joint_pos` / `joint_vel` | 29 each | `joint_pos_rel` / `joint_vel_rel` |
| `actions` | 29 | the previous action |

## What differs from upstream

- **The gait clock is a command term.** Upstream's `phase` reads `env.episode_length_buf`,
  which the browser does not serve. [`_gait_clock.py`](../_gait_clock.py) counts control
  steps in a `gait_clock` command that restarts on reset, as in `bipedhrl`, and the
  observation reads its `step_count` and the twist's `vel_command_b`. The joystick leaves
  the standing gate to the resampled command, as `bipedhrl`'s README describes.
- **mjlab 1.6.0.** Upstream pins `mjlab==1.2.0`. Its robot constants build `CollisionCfg`
  without the fields 1.6.0 made required, and its G1 asset loader calls
  `mjlab.utils.os.update_assets`, which 1.6.0 removed, so `upstream.py` supplies both
  while importing it. The task config and the G1 constants are upstream's own modules, so
  they are the same under either version.
- **No encoder bias.** mjlab's play config draws a ±0.015 rad bias per joint at startup;
  the browser applies the policy config's `encoder_bias` instead, and the checkpoint
  carries none.
- **One randomized robot, not a population.** The CoM offset on `torso_link` and the foot
  friction (0.3 to 1.6) are drawn once from the browser's seeded PRNG, where mjlab draws
  them per env.
- **No terrain re-draw.** `randomize_terrain` picks a sub-terrain per reset; the flat task
  has one plane, so it does nothing in either.
