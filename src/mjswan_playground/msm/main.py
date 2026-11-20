"""SuperMimic: two muscle-driven humans tracking motion-capture clips with the public
checkpoints. See ``README.md``."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import mjswan
import mujoco
import onnx
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.spec_config import MaterialCfg

from mjswan_playground._deps import license_file

from . import style100, terms, upstream

#: Scene name -> the body it runs.
SCENES = {"MS-Human-700": upstream.MS_HUMAN_700, "MyoFullBody": upstream.MYOFULLBODY}

#: The checkpoints state no license, so each scene carries where its policy comes from.
WEIGHTS_NOTICE = """\
The policy: {checkpoint}
from https://huggingface.co/{repo_id} at {revision}.
Its decoder is exported to ONNX with float16 weights, as upstream's browser demo ships it.
muscle-supermimic: https://github.com/kaibohex/muscle-supermimic.
The model card states no license and calls the checkpoints private.
"""

#: Parts built on their own for mjswan Cloud, which takes 100 MB: one body each.
PARTS = {"msm-mshuman": ("MS-Human-700",), "msm-myofullbody": ("MyoFullBody",)}

ATLAS_REPO_ID = "myohub/myosuite-assets"
ATLAS_REVISION = "afa8427188c8bd054925200c70db6757908dee45"
#: The free joint's qpos the atlas is modelled at, myoMimicFullbody-v0's zero pose: a
#: quarter turn about z from upstream's.
ATLAS_ROOT = (-0.025, 0.1, 1.0, 1.0, 0.0, 0.0, 0.0)
ATLAS_NOTICE = f"""\
MyoSuite's muscle atlas, muscles/fullbody_atlas_light.glb, licensed under CC BY 4.0,
from https://huggingface.co/datasets/{ATLAS_REPO_ID} at {ATLAS_REVISION[:7]}.
Bundled here unchanged.
Muscle geometry and names: BodyParts3D, (c) The Database Center for Life Science (DBCLS),
licensed under CC BY 4.0. Mitsuhashi N, et al. BodyParts3D: 3D structure database for
anatomical concepts. Nucleic Acids Res. 2009;37:D782-5. doi:10.1093/nar/gkn613
Obtained through human-atlas (https://github.com/ashemag/human-atlas), whose anatomy
data is CC BY 4.0 too.
Changed by MyoSuite: posed on its full-body skeleton, each muscle weighted to the bones
it attaches to, the quadriceps slimmed toward the femur, the bones chained along the body
tree, and decimated for the light copy.
"""


def ivory_bones(env_cfg: Any) -> None:
    """MyoSuite's ivory on the bones: linear (0.80, 0.73, 0.60), stored as sRGB."""
    robot = env_cfg.scene.entities["robot"]
    bones = [
        re.escape(g.name) + "$"
        for g in robot.spec_fn().geoms
        if g.type == mujoco.mjtGeom.mjGEOM_MESH
    ]
    robot.materials += (
        MaterialCfg(
            "ivory_bone", rgba=(0.91, 0.87, 0.80, 1.0), geom_names_expr=tuple(bones)
        ),
    )


def add_atlas(scene: mjswan.SceneHandle) -> None:
    from huggingface_hub import hf_hub_download

    atlas = hf_hub_download(
        ATLAS_REPO_ID,
        "muscles/fullbody_atlas_light.glb",
        repo_type="dataset",
        revision=ATLAS_REVISION,
    )
    scene.add_skin(
        "Muscles",
        mjswan.ModelledSkinConfig(
            atlas,
            rest_pose={"root": ATLAS_ROOT},
            hide_tendons=True,
            # MyoSuite's muscle material: linear (0.42, 0.11, 0.10), stored as sRGB.
            material=mjswan.SolidShading(
                color=(0.68, 0.37, 0.35),
                roughness=0.32,
                clearcoat=0.4,
                clearcoat_roughness=0.12,
            ),
            license="CC-BY-4.0",
            notice=ATLAS_NOTICE,
        ),
    )


def play_env_cfg(
    body: upstream.Body, root: Path
) -> tuple[Any, Path, list[upstream.Clip], list[str]]:
    """The body's play config as the browser runs it, its decoder, its clips and its
    joints in the clips' order."""
    from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics

    env_cfg = load_env_cfg(body.task_id, play=True)
    onnx_path, clips = upstream.ensure_policy(body, root, env_cfg)
    tracked = env_cfg.commands["motion"].body_names
    (sensor,) = env_cfg.scene.sensors
    # Keeps the last substep's reading, which the touch observation reads (`terms`).
    sensor.history_length = 1
    kin = SkeletonKinematics(
        tracked, "cpu", env_cfg.scene.entities["robot"].spec_fn().compile()
    )
    contract = terms.build_terms(
        Scene(env_cfg.scene, device="cpu").spec.compile(),
        kin,
        tracked=tracked,
        touch=sensor.name,
        thresholds={
            name: term.params["threshold"]
            for name, term in env_cfg.terminations.items()
            if "threshold" in term.params
        },
    )

    # mjlab and mjswan write the first tracked body's pose into the free joint, so that
    # body leads: MyoFullBody's pelvis, upstream's first, sits in it, same origin, turned.
    reference = (kin.info.root_name, *tracked[1:])
    env_cfg.commands["motion"] = terms.MotionCommandCfg(
        motion_file=str(clips[0].path),
        anchor_body_name=reference[0],
        body_names=reference,
    )
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
    return env_cfg, onnx_path, clips, kin.info.joint_names


def setup_builder(part: str | None = None) -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)

    builder = mjswan.Builder()
    # The checkpoints state no license.
    project = builder.add_project(name="SuperMimic")
    for name in PARTS[part] if part else SCENES:
        body = SCENES[name]
        env_cfg, onnx_path, clips, joint_names = play_env_cfg(body, root)
        if body is upstream.MYOFULLBODY:
            ivory_bones(env_cfg)
        scene = project.add_scene_mjlab(body.task_id, name=name, env_cfg=env_cfg)
        scene.set_viewer(replace(scene.viewer, color_management="filament"))
        if body is upstream.MYOFULLBODY:
            add_atlas(scene)
            scene.add_attribution(
                "musclemimic_models",
                license=license_file("musclemimic_models", "LICENSE"),
                notice=license_file("musclemimic_models", "NOTICE"),
            )
        # The code the scene's graphs and decoder are made from.
        scene.add_attribution(
            "muscle-supermimic", license=root / "LICENSE", notice=root / "NOTICE"
        )
        scene.add_attribution(
            "muscle-supermimic-checkpoint",
            notice=WEIGHTS_NOTICE.format(
                checkpoint=body.checkpoint,
                repo_id=upstream.POLICY_REPO_ID,
                revision=upstream.POLICY_REVISION[:7],
            ),
        )
        scene.add_attribution(
            "100style",
            license="CC-BY-4.0",
            copyright="Ian Mason",
            notice=style100.notice(mshuman700=body is upstream.MS_HUMAN_700),
        )
        policy = scene.add_policy(
            Path(body.checkpoint).stem,
            onnx.load(onnx_path),
            in_keys=["latent", "policy"],
        )
        command = env_cfg.commands["motion"]
        for clip in clips:
            policy.add_motion(
                name=clip.name,
                source=str(clip.path),
                fps=upstream.FPS,
                anchor_body_name=command.anchor_body_name,
                body_names=command.body_names,
                dataset_joint_names=joint_names,
                default=clip is clips[0],
                frame_fields=upstream.FRAME_FIELDS,
                group=clip.group,
            )
    return builder
