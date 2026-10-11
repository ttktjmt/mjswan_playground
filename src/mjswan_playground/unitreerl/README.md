# Unitree RL mjlab (`unitreerl`)

<img src="../../../assets/unitreerl.gif" width="480" alt="Unitree RL mjlab preview"/>

Source: https://github.com/unitreerobotics/unitree_rl_mjlab (Apache-2.0; the G1 model is
Unitree's, BSD-3-Clause) · the dance clip is LAFAN1's `dance1_subject2` retargeted to the
G1 (**CC BY-NC-ND 4.0**, see [License](#license))

Unitree's own mjlab repository, running every policy it ships a checkpoint for, on a
Unitree G1 (29 DoF) and two scenes:

- **Unitree-G1-Flat** walks to velocity commands, resampled every 3 to 8 s as upstream's
  play config draws them; mjlab's joystick panel can take over.
- **Unitree-G1-Tracking-No-State-Estimation** dances LAFAN1's `dance1_subject2`, a 131 s
  clip, from its first frame.

Every other task upstream registers (the G1's rough and state-estimating variants, the
23-DoF G1, and the Go2, A2, As2, H1-2, H2 and R1) ships no policy, so it has no scene here.

## Run

```sh
uv run msp run unitreerl
```

The build clones the repository into `.cache/` at a pinned commit; set
`MJSWAN_UNITREERL_ROOT` to point at a checkout you already have. Upstream runs from source
as the top-level package `src`, so [`upstream.py`](upstream.py) puts the checkout first on
`sys.path` and imports `src.tasks`, which registers every task.

| From `unitree_rl_mjlab` | Used as |
|---|---|
| `Unitree-G1-Flat`, `Unitree-G1-Tracking-No-State-Estimation` (`play=True`) | each scene, its observations, action term, commands, terminations and startup randomization |
| `deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx` | the walk, already exported; its mjlab metadata gives the joint order and the rest pose |
| `deploy/robots/g1/config/policy/mimic/dance1_subject2/exported/policy.onnx` | the dance, already exported, without metadata |
| `deploy/robots/g1/config/policy/*/params/deploy.yaml` | the dance's rest pose, and the joint map it shares with the walk, so the walk's joint order serves both |
| `deploy/robots/g1/config/policy/mimic/dance1_subject2/params/dance1_subject2.npz` | the clip, at 50 Hz |

| Scene | Policy | Checkpoint | Input → action | Rate |
|---|---|---|---|---|
| Unitree-G1-Flat | v0 | `velocity/v0` | 98 → 29 | 50 Hz |
| Unitree-G1-Tracking-No-State-Estimation | dance1_subject2 | `mimic/dance1_subject2` | 154 → 29 | 50 Hz |

## What the policies read

**Walk**: seven terms, one frame, no history:

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | the `imu_ang_vel` sensor |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `command` | 3 | the `twist` command |
| `phase` | 2 | sin and cos of a 0.6 s gait clock, zero while the command is below 0.1 |
| `joint_pos` / `joint_vel` | 29 each | `joint_pos_rel` / `joint_vel_rel` |
| `actions` | 29 | the previous action |

**Dance**: six terms, one frame, and no base velocity or position, which the robot does
not estimate:

| Term | Width | Source |
|---|---|---|
| `command` | 58 | the clip's joint positions and velocities at the current frame |
| `motion_anchor_ori_b` | 6 | the clip's torso orientation in the robot's torso frame |
| `base_ang_vel` | 3 | the `imu_ang_vel` sensor |
| `joint_pos` / `joint_vel` | 29 each | `joint_pos_rel` with the encoder bias / `joint_vel_rel` |
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
  while importing it. The walk's config and G1 constants are upstream's own modules. The
  dance runs on mjlab's own G1 and tracking command, which 1.6.0 changes only in how the
  G1 loads its meshes, an `imu_upvector` sensor no term reads, and a refresh of the
  robot's state after the clip wraps.
- **The dance loses the clip partway, as it does in mjlab.** Upstream's policy does not
  track all 131 s: in mjlab's own play env, its `ee_body_pos` termination ends the
  episode at 26.9 s for two seeds out of three and at 34.3 s for the third, and in the
  browser at 26.4 s. Each then starts the clip over from its first frame.
- **No encoder bias.** mjlab's play config draws a ±0.015 rad bias per joint at startup;
  the browser applies the policy config's `encoder_bias` instead, and neither checkpoint
  carries one.
- **One randomized robot, not a population.** The CoM offset on `torso_link` and the foot
  friction are drawn once from the browser's seeded PRNG, where mjlab draws them per env.
- **No terrain re-draw.** The walk's `randomize_terrain` picks a sub-terrain per reset;
  the flat task has one plane, so it does nothing in either.

## License

The code, both checkpoints and the G1 model are Apache-2.0, under upstream's `LICENCE`,
which the build ships as the project's `LICENSE`. The G1's meshes are byte for byte those
of mjlab's G1, from MuJoCo Menagerie's `unitree_g1`, which the build credits to Unitree
under BSD-3-Clause beside each scene.

The dance clip is `dance1_subject2` from Ubisoft La Forge's
[LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset), licensed
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), as retargeted to
the G1 in the [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).
It may be shared for noncommercial use, unmodified and with credit, so publishing the
dance scene is its author's call.
