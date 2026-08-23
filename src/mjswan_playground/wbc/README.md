# WBC-Mjlab G1 (`wbc`)

Source: https://github.com/wbc-mjlab/wbc-mjlab (Apache-2.0) ·
policy and clips from https://github.com/wbc-mjlab/wbc-g1-deploy (Apache-2.0) ·
the authors' own browser demo: https://wbc-mjlab.github.io/wbc-demo/

One whole-body tracking policy, many skills. Pick a clip — idle, walk, run, sprint,
dance, a fight combo, a flip — and the same controller tracks it.

## Run

```sh
uv sync --extra wbc             # wbc-mjlab registers the Wbc-* tasks with mjlab
uv run mjswan-playground run wbc
```

`wbc-mjlab` registers its tasks through mjlab's `mjlab.tasks` entry point, so
`add_scene_mjlab("Wbc-G1")` finds the task and **every term set defaults off its env
config**. This task only supplies what that config does not carry — the trained policy
and the clips, which live in the deploy repo, cloned into `.cache/`;
set `MJSWAN_WBC_DEPLOY_ROOT` to use a checkout you already have.

| From `wbc-g1-deploy` | Used as |
|---|---|
| `config/policy/wbc/params/policy.onnx` | the policy (132 → 29), already exported |
| `config/policy/wbc/params/config.yaml` | joint order and default pose — the same contract the C++ runtime reads |
| `config/clips/*.npz` + `manifest.yaml` | the motion library and which clip opens |

Three things this task needed and mjswan now carries: PD gains resolved off mjlab's
ideal-PD actuators (`resolve_pd_gains`), a reference-residual action term
(`ReferenceJointPositionActionCfg`, `q_cmd = q_ref(t) + scale · a`), and anchor-frame
reference slots on `TrackingCommand` — with which wbc's own observation functions trace
unmodified, so there is no second copy of the math here.

## What differs from upstream

Two training-only events are dropped: `assistive_wrench`, the helper wrench that carries
early training, and `pull_robot`, the disturbance sampler. Neither is applied by the
deploy runtime either.

## Fidelity

`mjswan.compile.run_parity`, term by term against the live `Wbc-G1` env.
