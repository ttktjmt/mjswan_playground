# MuJoCo Playground (`reaf`)

<img src="../../../assets/reaf.gif" width="480" alt="MuJoCo Playground preview"/>

Source: [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (Apache-2.0),
built on [reaf-sim](https://github.com/google-deepmind/reaf-sim) (Apache-2.0) ·
policies from [ttktjmt/so101-pick-cube](https://huggingface.co/ttktjmt/so101-pick-cube) and
[ttktjmt/so101-pick-cube-vision](https://huggingface.co/ttktjmt/so101-pick-cube-vision),
trained with mjswan's
[`examples/train/so101_pick_cube.py`](https://github.com/ttktjmt/mjswan/blob/main/examples/train/so101_pick_cube.py) ·
the SO-101 from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Apache-2.0)

MuJoCo Playground's RL environments, one scene each, named after the environment's ID in
Playground's `env_registry.py`. Each environment is also a part of its own, the build
that reaf-sim's documentation embeds on that environment's page.

| Scene | Part | What it does |
|---|---|---|
| `SO101PickCube` | `reaf-so101-pick-cube` | The SO-101 lifts a cube to a target that moves every 4 to 7 s, reading where the cube is. |
| `SO101PickCubeVision` | `reaf-so101-pick-cube-vision` | The same, finding the cube through its wrist camera, at 10 Hz. |

## Run

```sh
uv run msp run reaf                    # every environment
uv run msp run reaf-so101-pick-cube    # one of them
```

The first build fetches the SO-101 from MuJoCo Menagerie at a pinned commit into
`.cache/reaf/`, and each policy from the Hugging Face Hub at a pinned revision.

## Train

[`train.py`](train.py) trains any of Playground's environments with Playground's own
environment and PPO settings and exports the policy its scene runs, with a contract and a
reference rollout to check the scene against. It runs in Playground's environment
(MuJoCo 4), not this package's.
[`train.ipynb`](train.ipynb) runs it on a free Colab GPU and keeps every checkpoint on the
Hugging Face Hub, so a run that outlasts a session resumes in the next one:
[open it in Colab](https://colab.research.google.com/github/ttktjmt/mjswan_playground/blob/main/src/mjswan_playground/reaf/train.ipynb).

| From | Used as |
|---|---|
| Playground `tasks/manipulation/pick_cube` | the cube, the target and when it moves, the observations, the action scale, the resets |
| Playground `robots/so101/sim` | the servo dynamics identified on the real arm |
| Playground `_src/objects/assets/primitive/plane.xml` | the floor |
| Menagerie `robotstudio_so101` | the arm |
| Hub `ttktjmt/so101-pick-cube` at `fcdf077` | `SO101PickCube`'s policy, trained on Playground 1.0.1 |
| Hub `ttktjmt/so101-pick-cube-vision` at `c1b4d11` | `SO101PickCubeVision`'s policy, 1800 of its 3000 iterations |

## What differs from upstream

- **Restated, not imported.** Playground builds its environments with reaf-sim on
  MuJoCo 4 and PyTorch, while mjswan runs MuJoCo 3.11, so each environment is written again
  as an mjlab task whose terms compute Playground's observations. The port follows
  mjswan's `examples/demo/playground.py`.
- **One physics step later.** Playground reads the cube and the gripper from `mj_step`'s
  kinematics, one physics step older than these; on its reference rollout they are up
  to 4 mm apart, which the policy does not notice.
- **The wrist camera is drawn by three.js**, not Playground's ray tracer, so its shading
  differs from what the vision policy trained on.
- **No domain randomization or observation noise**: both are training-only.
