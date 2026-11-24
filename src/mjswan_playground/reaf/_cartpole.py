"""Playground's ``tasks/dm_control_suite/cartpole`` as an mjlab task: the DeepMind Control
Suite's cart-pole, on Playground's own model."""

from __future__ import annotations

import math

import mjswan
import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.cartpole.cartpole_env_cfg import pole_angle_cos_sin
from mjlab.viewer import ViewerConfig

#: Playground's `tasks/dm_control_suite/assets/cartpole.xml`.
CARTPOLE_XML = """
<mujoco model="cart-pole">
  <option timestep="0.01" integrator="RK4">
    <flag contact="disable" energy="disable"/>
  </option>

  <default>
    <default class="pole">
      <joint type="hinge" axis="0 1 0"  damping="2e-6"/>
      <geom type="capsule" fromto="0 0 0 0 0 1" size="0.045" mass=".1"/>
    </default>
  </default>

  <worldbody>
    <light name="light" pos="0 0 6"/>
    <camera name="fixed" pos="0 -4 1" zaxis="0 -1 0"/>
    <camera name="lookatcart" mode="targetbody" target="cart" pos="0 -2 2"/>
    <geom name="floor" pos="0 0 -.05" size="4 4 .2" type="plane"/>
    <geom name="rail1" type="capsule" pos="0  .07 1" zaxis="1 0 0" size="0.02 2"/>
    <geom name="rail2" type="capsule" pos="0 -.07 1" zaxis="1 0 0" size="0.02 2"/>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" limited="true" axis="1 0 0" range="-1.8 1.8"
        solreflimit=".08 1" damping="5e-4"/>
      <geom name="cart" type="box" size="0.2 0.15 0.1" mass="1"/>
      <body name="pole_1" childclass="pole">
        <joint name="hinge_1"/>
        <geom name="pole_1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="slide" joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1" />
  </actuator>
</mujoco>
"""

#: Playground's reset noise is normal with this deviation; mjlab's is uniform, over the
#: range with the same deviation.
NOISE = 0.01 * math.sqrt(3)
CART = SceneEntityCfg("cartpole", joint_names=("slider",))
HINGE = SceneEntityCfg("cartpole", joint_names=("hinge_1",))


def _spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_string(CARTPOLE_XML)
    # mjlab sets the options from `SimulationCfg`; the model's would only conflict.
    spec.option = mujoco.MjSpec().option
    return spec


def env_cfg(*, swing_up: bool) -> ManagerBasedRlEnvCfg:
    """`CartpoleBalance`, or with `swing_up` `CartpoleSwingUp`, at Playground's 100 Hz.

    The observation is Playground's: the cart's position, the pole's angle as its cosine
    and sine, and both velocities. The action is the motor's control, which MuJoCo clips
    to ±1. mjlab has no RK4, so the physics steps with `implicitfast`.
    """
    # Balance draws Playground's ranges about upright; swing-up its noise about hanging.
    cart_range = (-NOISE, NOISE) if swing_up else (-0.1, 0.1)
    hinge_range = (-NOISE, NOISE) if swing_up else (-0.034, 0.034)
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={
                "cartpole": EntityCfg(
                    spec_fn=_spec,
                    init_state=EntityCfg.InitialStateCfg(
                        joint_pos={
                            "slider": 0.0,
                            "hinge_1": math.pi if swing_up else 0.0,
                        }
                    ),
                    articulation=EntityArticulationInfoCfg(
                        actuators=(XmlActuatorCfg(target_names_expr=("slider",)),)
                    ),
                )
            },
        ),
        observations={
            "actor": ObservationGroupCfg(
                {
                    "cart_pos": ObservationTermCfg(
                        func=mdp.joint_pos_rel, params={"asset_cfg": CART}
                    ),
                    "pole_angle": ObservationTermCfg(
                        func=pole_angle_cos_sin, params={"asset_cfg": HINGE}
                    ),
                    "cart_vel": ObservationTermCfg(
                        func=mdp.joint_vel_rel, params={"asset_cfg": CART}
                    ),
                    "pole_vel": ObservationTermCfg(
                        func=mdp.joint_vel_rel, params={"asset_cfg": HINGE}
                    ),
                }
            )
        },
        actions={
            "effort": JointEffortActionCfg(
                entity_name="cartpole", actuator_names=("slider",), scale=1.0
            )
        },
        events={
            f"reset_{name}": EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": position_range,
                    "velocity_range": (-NOISE, NOISE),
                    "asset_cfg": asset_cfg,
                },
            )
            for name, position_range, asset_cfg in (
                ("cart", cart_range, CART),
                ("pole", hinge_range, HINGE),
            )
        },
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 0.0, 1.0),
            distance=4.5,
            azimuth=90.0,
            elevation=-10.0,
        ),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, disableflags=("contact",))),
        decimation=1,
        episode_length_s=10.0,
    )


def add_scene(
    project: mjswan.ProjectHandle,
    env_id: str,
    *,
    env_cfg: ManagerBasedRlEnvCfg,
    policy: tuple[str, str] | None,
    policy_name: str,
    in_keys: list[str] | None = None,
) -> mjswan.SceneHandle:
    """A scene named after its Playground environment, driven by the policy trained on
    it with `train.py`, from the Hub at a pinned revision, once there is one."""
    scene = project.add_scene_mjlab(env_id, name=env_id, env_cfg=env_cfg)
    if policy is not None:
        repo, revision = policy
        scene.add_policy_hf(repo, revision=revision, name=policy_name, in_keys=in_keys)
    return scene
