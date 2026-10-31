"""Upstream's command, observation, termination and action terms as the browser can run them.

Every upstream term reads its ``MotionCommand``, which turns the body's state into
positions, velocities and muscle forces with torch forward kinematics and a torch replica
of MuJoCo's tendons: thousands of graph nodes, too slow for a browser at 100 Hz, where
MuJoCo computes the same every step. So the observation reads MuJoCo's muscle and body
fields, on the simulated state rather than on upstream's, whose coupled joints sit
exactly on their ``<equality>`` polynomials; the encoder's latent and the reference the
terminations compare against are per-frame tables indexed by the clip and the step count;
and :class:`ClipLibraryCommand` plays the clips in turn.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply
from mjswan import CommandBinding, register_command

# A resampling time no episode reaches, so the clip changes on episode reset alone.
_NEVER = 1.0e9


@dataclass(kw_only=True)
class ClipLibraryCommandCfg(CommandTermCfg):
    """Each reset starts the next clip at its first frame."""

    init: np.ndarray
    """``(clips, nq + nv)``: the robot's qpos and qvel at each clip's first frame."""
    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (_NEVER, _NEVER)

    def build(self, env: Any) -> ClipLibraryCommand:
        return ClipLibraryCommand(self, env)


class ClipLibraryCommand(CommandTerm):
    cfg: ClipLibraryCommandCfg

    def __init__(self, cfg: ClipLibraryCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        init = torch.as_tensor(cfg.init, dtype=torch.float32, device=self.device)
        nq = 7 + self.robot.num_joints
        self._root_pose = init[:, :7]
        # mjlab writes the root's angular velocity in the world frame; qvel holds it in
        # the root's own.
        ang_vel = quat_apply(self._root_pose[:, 3:], init[:, nq + 3 : nq + 6])
        self._root_vel = torch.cat([init[:, nq : nq + 3], ang_vel], -1)
        self._joint_pos = init[:, 7:nq]
        self._joint_vel = init[:, nq + 6 :]
        # A float: the browser feeds a command's state to graphs as float32. The first
        # reset moves it to clip 0.
        self.clip = torch.full((self.num_envs,), -1.0, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.clip

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.clip = torch.remainder(self.clip + 1.0, float(len(self._root_pose)))
        index = self.clip[env_ids].long()
        self.robot.write_joint_state_to_sim(
            self._joint_pos[index], self._joint_vel[index], env_ids=env_ids
        )
        self.robot.write_root_link_pose_to_sim(self._root_pose[index], env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(
            self._root_vel[index], env_ids=env_ids
        )

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids

    def _update_metrics(self) -> None:
        pass


register_command(
    "ClipLibraryCommandCfg",
    CommandBinding(state_fields=["clip"], command_field="clip"),
)


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
    tables: Any,
    *,
    tracked: tuple[str, ...],
    touch: str,
    thresholds: dict[str, float],
) -> Terms:
    """The terms for one body.

    ``model`` is the scene as compiled, ``kin`` upstream's ``SkeletonKinematics`` of the
    robot, ``tables`` the clip tables of :func:`upstream.ensure_policy`, ``tracked`` the
    command's tracked bodies, ``touch`` the foot contact sensor, which keeps one substep
    of history, and ``thresholds`` upstream's for each deviation termination.
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

    clip_start = torch.as_tensor(tables.clip_start, dtype=torch.long)
    last_clip = len(tables.clip_start) - 1
    clip_len = torch.as_tensor(tables.clip_len, dtype=torch.long)
    latent = torch.as_tensor(tables.latent)
    ref_root_pos = torch.as_tensor(tables.root_pos, dtype=torch.float32)
    ref_root_rot = torch.as_tensor(tables.root_rot, dtype=torch.float32)
    ref_body_rpos = torch.as_tensor(tables.body_rpos, dtype=torch.float32)

    def frame(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """The clip and the frame of it the step count has reached, held at the end."""
        clip = env.command_manager.get_term("motion").clip.long().clamp(0, last_clip)
        steps = env.episode_length_buf.long()
        return clip, clip_start[clip] + torch.minimum(steps, clip_len[clip] - 1)

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

    def latent_term(env: Any) -> torch.Tensor:
        return latent[frame(env)[1]]

    # Upstream compares the state after a step with the frame after the observed one,
    # which the step count has reached by then.
    def body_pos_dev(env: Any) -> torch.Tensor:
        rpos = sites(env.sim.data)[0]
        deviation = torch.linalg.vector_norm(
            ref_body_rpos[frame(env)[1]] - rpos, dim=-1
        )
        return deviation.mean(-1) > thresholds["body_pos_dev"]

    def root_pos_dev(env: Any) -> torch.Tensor:
        position = env.sim.data.qpos[:, free_q[:3]]
        error = torch.linalg.vector_norm(ref_root_pos[frame(env)[1]] - position, dim=-1)
        return error > thresholds["root_pos_dev"]

    def root_ori_dev(env: Any) -> torch.Tensor:
        canonical = root_rotation(env.sim.data) @ kin.R_off
        return (
            rot_angle(ref_root_rot[frame(env)[1]], canonical)
            > thresholds["root_ori_dev"]
        )

    def motion_end(env: Any) -> torch.Tensor:
        clip, _ = frame(env)
        return env.episode_length_buf.long() >= clip_len[clip] - 1

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
        latent=latent_term,
        terminations={
            "motion_end": motion_end,
            "body_pos_dev": body_pos_dev,
            "root_pos_dev": root_pos_dev,
            "root_ori_dev": root_ori_dev,
        },
    )
