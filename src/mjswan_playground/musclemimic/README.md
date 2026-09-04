# MuscleMimic full body (`musclemimic`)

Source: https://github.com/Vittorio-Caggiano/myosuite4 (Apache-2.0) ·
policy and clips from https://github.com/amathislab/musclemimic

> Chengkun Li, Cheryl Wang, Bianca Ziliotto, Merkourios Simos, Jozsef Kovecses,
> Guillaume Durandau, Alexander Mathis.
> [Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale](https://arxiv.org/abs/2603.25544).
> arXiv:2603.25544, 2026.

A 354-muscle, 89-DoF musculoskeletal body tracking a retargeted walking clip, driven by
the public checkpoint [`amathislab/mm-10m-2`](https://huggingface.co/amathislab/mm-10m-2)
(2.05e9 steps). The scene is myosuite's mjlab task `myoMimicFullbody-v0`; the policy reads
upstream MuscleMimic's own 2418-wide observation, which `upstream.py` reproduces in torch
from myosuite's numpy builder.

## Run

> **Requires access to a private repository.** The `myoMimic*` mjlab tasks and
> `FullbodyObsAdapter` live only in [myosuite4](https://github.com/Vittorio-Caggiano/myosuite4),
> which is private; the public `myosuite` release carries neither. `--extra musclemimic`
> therefore resolves only for an account with access. The policy and the clips are public.

```sh
uv sync --extra musclemimic
hf auth login          # the clip dataset is gated, auto-approved
uv run msp run musclemimic
```

The first build downloads the checkpoint (140 MB) and the clip, converts the actor to
float16 ONNX (19 MB) into `.cache/`, and checks it against myosuite's forward pass.

## What the policy reads

`obs[1, 2418] -> actions[1, 354]`. Five blocks are raw sim fields read every step; the
sixth is a 484-row table baked from the clip and indexed by sim time.

| Block | Width | Source |
|---|---:|---|
| joint positions (root z + quat, then the rest) | 87 | `qpos` |
| joint velocities | 88 | `qvel` |
| per muscle: length, velocity, force, ctrl, activation | 1770 | `actuator_*`, `ctrl`, `act` |
| foot touch sensor sums | 4 | `sensordata` |
| mimic sites relative to the pelvis: pos, angles, vel | 192 | `site_xpos`, `site_xmat`, `cvel`, `subtree_com` |
| clip lookahead (5 steps, stride 20) + motion phase | 277 | baked table, indexed by `time` |

Actions are written to `ctrl` as they come, clipped to the actuator's `[-1, 1]`
(`action_mode="direct"`). The 17 mimic sites and the four touch sensors exist in the
model under mjlab's `entity/` prefixes.

| Compared | max \|Δ\| |
|---|---:|
| exported actor vs myosuite's numpy forward pass | 6.7e-06 |
| float16 vs float32 weights (600-step rollout identical) | 1.8e-02 |
| torch observation vs myosuite's numpy builder, 40 steps | 4.4e-07 |
| traced graph vs the torch observation, 24 steps | 2.4e-07 |

## What differs from upstream

- **Every episode starts on clip frame 0.** Training draws a random start frame and the
  browser resets `mjData.time` to 0, so `terms.py` pins the clip phase to sim time and
  replaces the random reset with a frame-0 one. It also rotates the clip's body-frame root
  angular velocity into the world frame mjlab's `write_root_state_to_sim` expects, which
  myosuite's own reset does not.
- **The termination is upstream's**, mean relative-site deviation or root deviation over
  1.0 m, not myosuite's `mimic_deviation`, which compares the root against the *mean of the
  target sites* with a 0.3 m limit and would end the episode at 1.8 s. The episode ends
  where the 484-frame clip wraps, at 4.84 s.
- **The actor is float16.** 38 MB in float32; hosts such as Cloudflare Pages cap a file at
  25 MiB. Actions move by at most 0.018 and the rollout does not change. Int8 would give
  9.6 MB but moves them by 0.22 and worsens tracking.
- **Browser behaviour is not asserted.** Parity proves each graph matches mjlab under
  `mujoco_warp`; the browser runs MuJoCo's WASM build.
