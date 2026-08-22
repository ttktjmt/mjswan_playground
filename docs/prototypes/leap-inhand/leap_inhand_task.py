"""LEAP in-hand cube rotation, rebuilt against mjlab 1.5.3 and mjswan.

The upstream task (github.com/Msornerrrr/in-hand-rotation-mjlab) targets mjlab v1.1 and
carries a training env: domain randomization, a grasp-cache resampler, an asymmetric
critic, rewards, curricula. None of that reaches a browser. What does reach it is the
actor's two observation terms, the action, the scene, and one termination — so those are
what this module states, against current mjlab.

Only the *robot definition* is borrowed from upstream (via `leap_compat`), because its
actuator SysID numbers and collision bitmasks are the calibration the policy trained
against, and retyping them would be a silent-divergence risk.

Two differences from upstream, both deliberate:

- **The action term.** Upstream integrates the joint target on its own previous command
  (`q_cmd += delta`). mjlab's own generic term, `RelativeJointPositionActionCfg`,
  re-bases on the measured position (`q_cmd = q + delta`); it is what mjswan now
  implements, so it is what this runs.
- **No domain randomization.** Observation noise and delay, cube mass/size/friction,
  actuator gains and delays: all training-time, and mjswan drops them by design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import leap_compat

leap_compat.install()

import mujoco  # noqa: E402
import torch  # noqa: E402
from mjlab.entity import EntityCfg  # noqa: E402
from mjlab.envs import ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.envs import mdp as envs_mdp  # noqa: E402
from mjlab.managers.observation_manager import (  # noqa: E402
    ObservationGroupCfg as MjlabObsGroupCfg,
)
from mjlab.managers.observation_manager import (  # noqa: E402
    ObservationTermCfg as MjlabObsTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.managers.termination_manager import TerminationTermCfg  # noqa: E402
from mjlab.rl import (  # noqa: E402
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg  # noqa: E402
from mjlab.sim import MujocoCfg, SimulationCfg  # noqa: E402
from mjlab.terrains import TerrainEntityCfg  # noqa: E402
from mjlab.viewer import ViewerConfig  # noqa: E402

from mjlab.envs.mdp.actions import (  # noqa: E402
    RelativeJointPositionActionCfg,
)
from mjswan.managers.observation_manager import (  # noqa: E402
    ObservationGroupCfg,
    ObservationTermCfg,
)

REPO = Path(__file__).resolve().parent / "in-hand-rotation-mjlab"
GRASP_CACHE = (
    REPO
    / "src/in_hand_rotation_mjlab/tasks/hand_cube/cache/leap_left_custom_grasp_cache.npz"
)

TASK_ID = "Mjswan-Leap-Left-Custom-HandCube-Rotate"
ENTITY = "robot"
OBJECT = "cube"

#: `sim.mujoco.timestep` (0.005) * `decimation` (10) — 20 Hz, the rate the policy
#: trained at.
CONTROL_DT = 0.05
DECIMATION = 10

#: The action is a per-step joint delta of +/- 1/24 rad at a raw action of +/- 1.
DELTA_PER_STEP = 1.0 / 24.0

#: The actor group's `history_length=10`, oldest frame first — mjlab's history buffer is
#: chronological, so the 10 look-back offsets run 9 -> 0.
HISTORY_STEPS = tuple(range(9, -1, -1))

#: `object_fallen` / `cube_fell` in the upstream config.
CUBE_MIN_HEIGHT = 0.2


def joint_pos_commanded(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(ENTITY, joint_names=(".*",)),
) -> torch.Tensor:
    """The position target the action term last wrote, i.e. mjlab's `joint_pos_target`.

    Upstream reads the action term's own integrator state and falls back to this field
    when the term has none. mjlab's `RelativeJointPositionAction` has none, so this *is*
    the fallback path — and being a plain `Entity.data` read, it traces to a slot the
    browser serves rather than to anything action-manager-shaped.
    """
    return env.scene[asset_cfg.name].data.joint_pos_target[:, asset_cfg.joint_ids]


def _grasp(cube_size: float = 0.0375) -> dict:
    """One grasp from upstream's cache: hand pose, cube pose, cube size.

    The browser runs a single environment with one cube, so the reset-time resampling
    over 7,700 grasps and five size buckets collapses to a choice made here. Picking the
    sample nearest the nominal size keeps the demo on the middle of the trained range.
    """
    with np.load(GRASP_CACHE) as cache:
        sizes = cache["cube_size"]
        index = int(np.argmin(np.abs(sizes - cube_size)))
        return {
            "joint_names": [str(n) for n in cache["joint_names"]],
            "joint_pos": cache["joint_pos"][index].astype(float),
            "cube_pose": cache["cube_pose_rel"][index].astype(float),
            "cube_size": float(sizes[index]),
            "index": index,
        }


#: The actor's two terms. Shared verbatim between the mjlab config below (which the
#: tracer builds a live env from) and the mjswan group the policy carries, so the two
#: cannot drift into computing different observations.
_ACTOR_TERMS: dict[str, dict] = {
    "joint_pos": {
        "func": envs_mdp.joint_pos_rel,
        "params": {
            "asset_cfg": SceneEntityCfg(ENTITY, joint_names=(".*",)),
            "biased": True,
        },
    },
    "prev_commanded_joint_pos": {
        "func": joint_pos_commanded,
        "params": {"asset_cfg": SceneEntityCfg(ENTITY, joint_names=(".*",))},
    },
}


def mjswan_actor_group() -> ObservationGroupCfg:
    """The same two terms, with the group's `history_length=10` unrolled into the
    per-term look-back offsets mjswan wants — oldest frame first, matching the order
    mjlab's chronological history buffer flattens into."""
    return ObservationGroupCfg(
        terms={
            name: ObservationTermCfg(**spec, history_steps=HISTORY_STEPS)
            for name, spec in _ACTOR_TERMS.items()
        },
    )


def _cube_spec_fn(size: float, mass: float = 0.1):
    def get_cube_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name=OBJECT)
        body.add_freejoint(name="cube_joint")
        body.add_geom(
            name="cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(size,) * 3,
            mass=mass,
            rgba=(0.85, 0.3, 0.15, 1.0),
        )
        return spec

    return get_cube_spec


def make_env_cfg() -> ManagerBasedRlEnvCfg:
    from in_hand_rotation_mjlab.robots import get_leap_left_custom_hand_cfg

    grasp = _grasp()

    robot_cfg = get_leap_left_custom_hand_cfg()
    # The grasp the cube pose was recorded against — not the open-hand home pose.
    robot_cfg.init_state.joint_pos = {
        name: float(value)
        for name, value in zip(grasp["joint_names"], grasp["joint_pos"])
    }

    cube_pose = grasp["cube_pose"]
    cube_cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(cube_pose[:3]),
            rot=tuple(cube_pose[3:]),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        spec_fn=_cube_spec_fn(grasp["cube_size"]),
    )

    # mjlab's own classes: `add_scene_mjlab` instantiates this config as a live env to
    # trace against, and mjlab's manager rejects mjswan's extra fields. The per-term
    # look-back the browser needs is stated separately, in `mjswan_actor_group`.
    observations = {
        "actor": MjlabObsGroupCfg(
            {name: MjlabObsTermCfg(**spec) for name, spec in _ACTOR_TERMS.items()},
            history_length=len(HISTORY_STEPS),
            flatten_history_dim=True,
        ),
    }

    actions = {
        "joint_pos": RelativeJointPositionActionCfg(
            entity_name=ENTITY,
            actuator_names=(".*",),
            scale=DELTA_PER_STEP,
        ),
    }

    terminations = {
        "cube_fell": TerminationTermCfg(
            func=envs_mdp.root_height_below_minimum,
            params={
                "minimum_height": CUBE_MIN_HEIGHT,
                "asset_cfg": SceneEntityCfg(OBJECT),
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={ENTITY: robot_cfg, OBJECT: cube_cfg},
            num_envs=1,
            env_spacing=0.6,
        ),
        observations=observations,
        actions=actions,
        commands={},
        events={},
        rewards={},
        terminations=terminations,
        curriculum={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name=ENTITY,
            body_name="palm",
            distance=0.45,
            elevation=-25,
            azimuth=110,
        ),
        sim=SimulationCfg(
            nconmax=55,
            njmax=600,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=DECIMATION,
        # The browser resets only when the cube falls; upstream's play config does the
        # same by setting an episode length no run reaches.
        episode_length_s=1.0e9,
        scale_rewards_by_dt=True,
    )


def make_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """What mjswan reads off the runner: the actor's observation group, and the raw
    action bound the policy was rolled out under."""
    model = RslRlModelCfg(
        hidden_dims=(512, 512, 256), activation="elu", obs_normalization=True
    )
    return RslRlOnPolicyRunnerCfg(
        actor=model,
        critic=model,
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.003,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="leap_left_hand_cube_rotate",
        num_steps_per_env=32,
        max_iterations=5_000,
        clip_actions=1.0,
    )


#: The 16 joints in model order, which is also actuator order for this hand — so the
#: policy's action vector needs no reindexing between the two.
JOINT_ORDER = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)


def policy_joint_names() -> list[str]:
    return [f"{ENTITY}/{name}" for name in JOINT_ORDER]


def default_joint_pos() -> list[float]:
    """The grasp the episode starts from, in `JOINT_ORDER`."""
    grasp = _grasp()
    by_name = dict(zip(grasp["joint_names"], grasp["joint_pos"]))
    return [float(by_name[name]) for name in JOINT_ORDER]


def register() -> str:
    from mjlab.tasks.registry import register_mjlab_task

    env_cfg = make_env_cfg()
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg,
        play_env_cfg=make_env_cfg(),
        rl_cfg=make_rl_cfg(),
    )
    return TASK_ID
