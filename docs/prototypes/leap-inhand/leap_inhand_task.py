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
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
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
from mjswan.envs.mdp.actions import (  # noqa: E402
    RelativeJointPositionActionCfg as MjswanRelativeJointPositionActionCfg,
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

#: `EntityArticulationInfoCfg.soft_joint_pos_limit_factor` on the LEAP hand: the
#: integrated target is clamped to the joint range shrunk by this about its midpoint.
SOFT_JOINT_POS_LIMIT_FACTOR = 0.95

#: `object_fallen` / `cube_fell` in the upstream config.
CUBE_MIN_HEIGHT = 0.2

#: `GRASP_INIT_JOINT_POS` from upstream's leap_left_custom config. The hand starts here
#: on every episode and — this is the part that matters — `joint_pos_rel` subtracts it.
#: Upstream's grasp-cache reset writes the *cube's* size and pose and nothing else
#: ("Robot joints are not touched"), so the cache's own recorded hand pose is not the
#: reference the policy was trained against.
GRASP_INIT_JOINT_POS: dict[str, float] = {
    "if_mcp": 0.1, "if_rot": 0.4, "if_pip": 1.3, "if_dip": 0.0,
    "mf_mcp": 0.1, "mf_rot": 0.0, "mf_pip": 1.3, "mf_dip": 0.0,
    "rf_mcp": 0.1, "rf_rot": -0.4, "rf_pip": 1.3, "rf_dip": 0.0,
    "th_cmc": 1.45, "th_axl": -1.5, "th_mcp": 0.579, "th_ipl": 1.37,
}


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
            # The hand pose the grasp settled into. Recorded for reference only: the
            # reset never writes it, so it is not the policy's `default_joint_pos`.
            "settled_joint_pos": cache["joint_pos"][index].astype(float),
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


def _upstream_delta_action_cfg():
    """Upstream's `JointPositionDeltaActionCfg`, loaded straight from its file.

    Not `from in_hand_rotation_mjlab.tasks…import`: the task package's `__init__` walks
    every config module, and those still call the mjlab v1.1 domain-randomization API.
    The action module itself imports nothing from its own package, so a file-path load
    gets the real class with no shims at all.
    """
    import importlib.util
    import sys

    path = (
        leap_compat.IN_HAND_SRC
        / "in_hand_rotation_mjlab/tasks/hand_cube/mdp/actions.py"
    )
    spec = importlib.util.spec_from_file_location("_upstream_hand_cube_actions", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves annotations through `sys.modules[cls.__module__]`, so the
    # module has to be registered before its body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.JointPositionDeltaActionCfg


def _action_cfg(kind: str):
    """`relative` is mjlab's own term; `integrator` is upstream's, which is what the
    policy was trained with. See `mjswan_action_cfg` for the browser side."""
    if kind == "relative":
        return RelativeJointPositionActionCfg(
            entity_name=ENTITY, actuator_names=(".*",), scale=DELTA_PER_STEP
        )
    if kind == "integrator":
        JointPositionDeltaActionCfg = _upstream_delta_action_cfg()
        return JointPositionDeltaActionCfg(
            entity_name=ENTITY,
            actuator_names=(".*",),
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clip_to_joint_limits=True,
            use_soft_joint_pos_limits=True,
            delta_min=-DELTA_PER_STEP,
            delta_max=DELTA_PER_STEP,
            interpolate_decimation=True,
        )
    raise ValueError(f"unknown action kind {kind!r}")


def make_env_cfg(action: str = "integrator") -> ManagerBasedRlEnvCfg:
    from in_hand_rotation_mjlab.robots import get_leap_left_custom_hand_cfg

    grasp = _grasp()

    robot_cfg = get_leap_left_custom_hand_cfg()
    robot_cfg.init_state.joint_pos = dict(GRASP_INIT_JOINT_POS)

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

    actions = {"joint_pos": _action_cfg(action)}

    # mjlab applies an entity's `init_state` through reset *events*, not automatically:
    # with no events the hand sits at all-zero joints and the cube at the world origin.
    # Upstream's three, with every randomisation range zeroed — the browser runs one
    # environment and shows the nominal grasp.
    events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg(ENTITY),
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg(ENTITY, joint_names=(".*",)),
            },
        ),
        "reset_cube_pose": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg(OBJECT),
            },
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
        events=events,
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
    """The pose `joint_pos_rel` is relative to, in `JOINT_ORDER`."""
    return [GRASP_INIT_JOINT_POS[name] for name in JOINT_ORDER]


def mjswan_action_cfg() -> MjswanRelativeJointPositionActionCfg:
    """The browser's action term.

    `relative_to="command"` because that is the controller the policy was trained with:
    the target integrates on its own previous command. mjlab's own measured-position
    form is a different controller and, measured against this policy, does not rotate
    the cube at all (0.001 rad/s over 400 steps, against 0.305 for the integrator).
    """
    return MjswanRelativeJointPositionActionCfg(
        entity_name=ENTITY,
        actuator_names=(".*",),
        scale=DELTA_PER_STEP,
        relative_to="command",
        clip_to_joint_limits=True,
        soft_joint_pos_limit_factor=SOFT_JOINT_POS_LIMIT_FACTOR,
        interpolate_decimation=True,
    )


def register(action: str = "integrator") -> str:
    from mjlab.tasks.registry import register_mjlab_task

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=make_env_cfg(action),
        play_env_cfg=make_env_cfg(action),
        rl_cfg=make_rl_cfg(),
    )
    return TASK_ID
