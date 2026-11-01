"""Upstream's command, observation, termination and action terms as the browser can run them.

Every upstream term reads its ``MotionCommand``, which turns the body's state into
positions, velocities and muscle forces with torch forward kinematics and a torch replica
of MuJoCo's tendons: thousands of graph nodes, too slow for a browser at 100 Hz, where
MuJoCo computes the same every step. So the observation reads MuJoCo's muscle and body
fields, on the simulated state rather than on upstream's, whose coupled joints sit
exactly on their ``<equality>`` polynomials; and :class:`MotionCommand`, mjlab's, which
mjswan runs as its own tracking command, serves each clip's reference and per-frame
latent.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

import numpy as np
import torch
from mjlab.tasks.tracking import mdp
from mjlab.utils.lab_api.math import quat_apply

from .upstream import FRAME_FIELDS

# A resampling time no episode reaches, so the clip changes on episode reset alone.
_NEVER = 1.0e9


@dataclass(kw_only=True)
class MotionCommandCfg(mdp.MotionCommandCfg):
    """mjlab's tracking command, under the name mjswan runs as its own, serving each
    clip's :data:`~.upstream.FRAME_FIELDS` too."""

    entity_name: str = "robot"
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    sampling_mode: str = "start"
    resampling_time_range: tuple[float, float] = (_NEVER, _NEVER)

    def build(self, env: Any) -> MotionCommand:
        return MotionCommand(self, env)


class MotionCommand(mdp.MotionCommand):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        with np.load(cfg.motion_file) as clip:
            self._latent, self._frames_left = (
                torch.as_tensor(clip[name], device=self.device) for name in FRAME_FIELDS
            )
        self._held = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def latent(self) -> torch.Tensor:
        return self._latent[self.time_steps]

    @property
    def frames_left(self) -> torch.Tensor:
        return self._frames_left[self.time_steps]

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        extras = super().reset(env_ids)
        self._held[env_ids] = True
        return extras

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        # Hold the frame a reset wrote through the next update, as upstream's command and
        # mjswan's do; mjlab's steps past it.
        ids = slice(None) if env_ids is None else env_ids
        self.time_steps[ids] -= self._held[ids].long()
        self._held[ids] = False
        super()._update_command(env_ids)

    def _write_reference_state_to_sim(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_ori: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        """mjlab's, written as mjswan writes it: the joints unclipped, as upstream's reset
        leaves a coupled joint up to 1e-4 rad past its range on its ``<equality>``
        polynomial, and the root's angular velocity, in the root's frame at the first
        frame (see the ponytail in `upstream`), straight into qvel."""
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_ang_vel = quat_apply(root_ori, root_ang_vel)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], -1),
            env_ids=env_ids,
        )
        self.robot.reset(env_ids=env_ids)


def muscle_action(term: Any) -> Any:
    """Upstream's ``MuscleExcitationActionCfg`` under the name of mjswan's muscle action,
    which writes the same ctrl: each excitation, clipped to [0, 1]. mjlab still builds
    upstream's term, which writes it through the tendon's effort target, a path mjswan
    does not run."""
    from muscle_supermimic.tasks.tracking.mdp.actions import MuscleExcitationActionCfg

    @dataclass(kw_only=True)
    class MuscleActivationActionCfg(MuscleExcitationActionCfg):
        action_mode: str = "excitation"

    return MuscleActivationActionCfg(
        **{f.name: getattr(term, f.name) for f in fields(term)}
    )


@dataclass
class Terms:
    observations: dict[str, Callable[[Any], torch.Tensor]]
    """The decoder's observation, term by term in upstream's order."""
    latent: Callable[[Any], torch.Tensor]
    terminations: dict[str, Callable[[Any], torch.Tensor]]
    """Upstream's failures and its clip end, ``(B,)`` bool each."""


def build_terms(
    model: Any,
    kin: Any,
    *,
    tracked: tuple[str, ...],
    touch: str,
    thresholds: dict[str, float],
) -> Terms:
    """The terms for one body.

    ``model`` is the scene as compiled, ``kin`` upstream's ``SkeletonKinematics`` of the
    robot, ``tracked`` upstream's tracked bodies, ``touch`` the foot contact sensor, which
    keeps one substep of history, and ``thresholds`` upstream's for each deviation
    termination.
    """
    from muscle_supermimic.tasks.tracking.mdp.heads import rel_sites, rot_angle
    from muscle_supermimic.utils.rotations import mat_to_quat, quat_to_mat

    def ids(values: Any) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values, dtype=np.int64))

    root = model.joint("robot/root")
    free_q = ids(np.arange(7) + root.qposadr[0])
    free_v = ids(np.arange(6) + root.dofadr[0])
    joints = [model.joint(f"robot/{name}") for name in kin.info.indep_names]
    q_adr = ids([j.qposadr[0] for j in joints])
    v_adr = ids([j.dofadr[0] for j in joints])
    act_adr = ids(model.actuator_actadr)
    bodies = ids([model.body(f"robot/{name}").id for name in tracked])
    body_root = ids(model.body_rootid[bodies.numpy()])

    def command(env: Any) -> Any:
        return env.command_manager.get_term("motion")

    def root_rotation(data: Any) -> torch.Tensor:
        return quat_to_mat(data.qpos[:, free_q[3:]])

    def joints_full(data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return kin.full_joint_state(data.qpos[:, q_adr], data.qvel[:, v_adr])

    def sites(data: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # `mj_objectVelocity` of each body origin: cvel is about its root's subtree com.
        pos = data.xpos[:, bodies]
        cvel = data.cvel[:, bodies]
        angular = cvel[..., :3]
        linear = cvel[..., 3:] + torch.cross(
            angular, pos - data.subtree_com[:, body_root], dim=-1
        )
        rot = data.xmat[:, bodies].reshape(-1, len(tracked), 3, 3)
        return rel_sites(pos, rot, linear, angular)

    def root_pos(env: Any) -> torch.Tensor:
        """Height, and the free joint's quaternion with w >= 0."""
        data = env.sim.data
        return torch.cat(
            [data.qpos[:, free_q[2:3]], mat_to_quat(root_rotation(data))], -1
        )

    def joint_pos(env: Any) -> torch.Tensor:
        return joints_full(env.sim.data)[0]

    def root_vel(env: Any) -> torch.Tensor:
        """The free joint's qvel: linear in the world frame, angular in the root's."""
        return env.sim.data.qvel[:, free_v]

    def joint_vel(env: Any) -> torch.Tensor:
        return joints_full(env.sim.data)[1]

    # `[:]` unwraps the `TorchArray` a live env serves.
    def muscle_length(env: Any) -> torch.Tensor:
        return env.sim.data.actuator_length[:]

    def muscle_velocity(env: Any) -> torch.Tensor:
        return env.sim.data.actuator_velocity[:]

    def muscle_force(env: Any) -> torch.Tensor:
        return env.sim.data.actuator_force[:]

    def muscle_excitation(env: Any) -> torch.Tensor:
        return env.action_manager.action

    def muscle_activation(env: Any) -> torch.Tensor:
        return env.sim.data.act[:, act_adr]

    def touch_force(env: Any) -> torch.Tensor:
        """The vertical force of the ground on each foot body.

        Upstream's terminations read the sensor first in a step, before mjlab's
        ``forward()``, and the observation gets that cached reading, the last substep's;
        after a reset, nothing reads it before the observation does.
        """
        data = env.scene[touch].data
        fresh = env.episode_length_buf.unsqueeze(-1) == 0
        return torch.where(
            fresh, data.force[..., 2], data.force_history[..., 0, 2]
        ).abs()

    def site_rpos(env: Any) -> torch.Tensor:
        return sites(env.sim.data)[0].flatten(1)

    def site_rangles(env: Any) -> torch.Tensor:
        return sites(env.sim.data)[1].flatten(1)

    def site_rvel(env: Any) -> torch.Tensor:
        return sites(env.sim.data)[2].flatten(1)

    def latent(env: Any) -> torch.Tensor:
        return command(env).latent

    # Upstream compares the state after a step with the frame after the observed one;
    # the command serves the observed one until its update, after the terminations.
    def body_pos_dev(env: Any) -> torch.Tensor:
        ref = command(env).body_pos_w
        pos = env.sim.data.xpos[:, bodies]
        deviation = torch.linalg.vector_norm(
            (ref[:, 1:] - ref[:, :1]) - (pos[:, 1:] - pos[:, :1]), dim=-1
        )
        return deviation.mean(-1) > thresholds["body_pos_dev"]

    def root_pos_dev(env: Any) -> torch.Tensor:
        position = env.sim.data.qpos[:, free_q[:3]]
        error = torch.linalg.vector_norm(command(env).anchor_pos_w - position, dim=-1)
        return error > thresholds["root_pos_dev"]

    def root_ori_dev(env: Any) -> torch.Tensor:
        ref = quat_to_mat(command(env).anchor_quat_w)
        return rot_angle(ref, root_rotation(env.sim.data)) > thresholds["root_ori_dev"]

    def motion_end(env: Any) -> torch.Tensor:
        """The step reached the clip's last frame."""
        return command(env).frames_left[:, 0] <= 1

    return Terms(
        observations={
            "root_pos": root_pos,
            "joint_pos": joint_pos,
            "root_vel": root_vel,
            "joint_vel": joint_vel,
            "muscle_length": muscle_length,
            "muscle_velocity": muscle_velocity,
            "muscle_force": muscle_force,
            "muscle_excitation": muscle_excitation,
            "muscle_activation": muscle_activation,
            "touch": touch_force,
            "site_rpos": site_rpos,
            "site_rangles": site_rangles,
            "site_rvel": site_rvel,
        },
        latent=latent,
        terminations={
            "motion_end": motion_end,
            "body_pos_dev": body_pos_dev,
            "root_pos_dev": root_pos_dev,
            "root_ori_dev": root_ori_dev,
        },
    )
