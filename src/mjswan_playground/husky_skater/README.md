# HUSKY Skateboarding (`husky-skater`)

Source: https://github.com/TeleHuman/humanoid_skateboarding ·
project page: https://husky-humanoid.github.io

> Jinrui Han, Dewei Wang, Chenyun Zhang, Xinzhe Liu, Ping Luo, Chenjia Bai, Xuelong Li.
> [HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control](https://arxiv.org/abs/2602.03205).
> Robotics: Science and Systems (RSS), 2026. arXiv:2602.03205.

A Unitree G1 riding a skateboard, pushing and steering under whole-body control.

## Run

```sh
uv run mjswan-playground run husky-skater
```

The build clones the HUSKY repository into `.cache/` at a pinned
commit; set `MJSWAN_HUSKY_ROOT` to point at a checkout you already have. The package
itself is never installed or imported — everything the demo needs is data it ships:

| From upstream | Used as |
|---|---|
| `test_scene/mjlab_scene.xml` | the scene, generated from the task's own `scene_cfg` |
| `src/.../xmls/g1.xml` | the robot alone, for the tracing env |
| `ckpts/test.onnx` | the policy |
| the actuators' `gainprm` / `forcerange` | the action scale (`0.25 * effort_limit / stiffness`) |
| the scene's `init_state` keyframe | the default joint pose the action offsets from |

Its license is [CC BY-NC 4.0](https://github.com/TeleHuman/humanoid_skateboarding/blob/main/LICENSE-CC-BY-NC-4.0.md). This project uses those assets, and publishes the demo to mjswan Cloud, with the HUSKY authors' permission. Please review the license when using this for your own work.

## How the observations map

The task's `policy` group is eight terms with `history_length=5`, wired in
[`main.py`](main.py). Six are mjlab's own MDP functions, traced to ONNX exactly as any
other task's are. The two in [`terms.py`](terms.py) read state the tracer cannot serve —
`heading` is an attribute of the task's env subclass, and `phase` is a gait clock the env
keeps in a step counter, which becomes a traced command term. See that module for why.

## What differs from upstream

- **The command is the operator's, not the task's.** `SkateUniformVelocityCommand`
  resamples a speed and heading on a 20 s timer and re-references the heading whenever
  the robot leaves a push. The browser gets the two sliders its evaluation script binds
  to the arrow keys instead — the same two numbers the policy reads, chosen by hand.
- **A fall resets the episode.** Upstream's play config keeps only a 60 s timeout,
  which suits a scripted evaluation; the demo carries the training config's `fell_over`
  check (`bad_orientation` past 70°) so a fall puts the robot back on the board.
- **No domain randomization.** The task's startup events perturb the CoM and the
  friction of the board, wheels and feet per env; the browser runs one env, and the
  demo shows the nominal robot.

## Fidelity

Checked against the observation vector upstream's `test_scene/sim.py` assembles, in
lockstep on a shared simulation: `max |Δ| = 1.2e-07` over 500 control steps (float32
rounding), across push and both steering directions.
