"""SuperMimic: two muscle-driven humans tracking motion-capture clips with the public
checkpoints. See ``README.md``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mjswan
import onnx
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

from . import upstream

#: Scene name -> the body it runs.
SCENES = {"MS-Human-700": upstream.MS_HUMAN_700, "MyoFullBody": upstream.MYOFULLBODY}


def play_env_cfg(body: upstream.Body, root: Path) -> tuple[Any, Path]:
    """The body's play config as the browser runs it, and its decoder."""
    from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics

    from . import terms

    env_cfg = load_env_cfg(body.task_id, play=True)
    onnx_path, tables = upstream.ensure_policy(body, root, env_cfg)
    command = env_cfg.commands["motion"]
    (sensor,) = env_cfg.scene.sensors
    # The last substep's reading, which the touch observation reads (`terms`).
    sensor.history_length = 1
    contract = terms.build_terms(
        Scene(env_cfg.scene, device="cpu").spec.compile(),
        SkeletonKinematics(
            command.body_names,
            "cpu",
            env_cfg.scene.entities["robot"].spec_fn().compile(),
        ),
        tables,
        tracked=command.body_names,
        touch=sensor.name,
        thresholds={
            name: term.params["threshold"]
            for name, term in env_cfg.terminations.items()
            if "threshold" in term.params
        },
    )

    env_cfg.commands["motion"] = terms.ClipLibraryCommandCfg(init=tables.init)
    env_cfg.observations = {
        "latent": ObservationGroupCfg(
            terms={"latent": ObservationTermCfg(func=contract.latent)},
            enable_corruption=False,
        ),
        "policy": ObservationGroupCfg(
            terms={
                name: ObservationTermCfg(func=func)
                for name, func in contract.observations.items()
            },
            enable_corruption=False,
        ),
    }
    env_cfg.terminations = {
        name: TerminationTermCfg(func=func, time_out=name == "motion_end")
        for name, func in contract.terminations.items()
    }
    env_cfg.actions["muscles"] = terms.muscle_action(env_cfg.actions["muscles"])
    # Every reward reads upstream's motion command, and the browser computes none.
    env_cfg.rewards = {}
    return env_cfg, onnx_path


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)

    builder = mjswan.Builder()
    # The checkpoints state no license.
    project = builder.add_project(name="SuperMimic")
    for name, body in SCENES.items():
        env_cfg, onnx_path = play_env_cfg(body, root)
        scene = project.add_scene_mjlab(body.task_id, name=name, env_cfg=env_cfg)
        scene.add_policy(
            Path(body.checkpoint).stem,
            onnx.load(onnx_path),
            in_keys=["latent", "policy"],
        )
    return builder
