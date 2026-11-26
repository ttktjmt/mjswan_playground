"""Retarget a 100STYLE walk onto the G1: the clip `G1WholeBodyTracking` trains on and
plays, in place of LAFAN1's dance, whose license (CC BY-NC-ND 4.0) allows no published
derivative.

    uv run --extra msm python -m mjswan_playground.reaf.retarget_100style

100STYLE (Ian Mason, https://ianxmason.github.io/100style/, CC BY 4.0) is fetched as the
msm task fetches it. GMR (MuscleMimic's fork, gmr_plus, MIT) fits the G1 to each BVH
frame at 30 fps, with GMR's own costs for LAFAN1 onto the G1 (`bvh_lafan1_to_g1.json`),
and scales and offsets made from the two rest poses, as msm makes MyoFullBody's:
100STYLE's T-pose and the G1 with its arms held out level. The clip then starts at the
origin facing +x, its feet on the floor. It is written to `motions/` in the LAFAN1
Retargeting Dataset's layout, which Playground's `csv_to_npz` reads: per frame, the
pelvis's position and xyzw quaternion, then the 29 joints in the G1's order.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from mjswan_playground._deps import CACHE_DIR
from mjswan_playground.msm import style100

#: The neutral forward walk from where `Frame_Cuts.csv` starts its capture.
CLIP = style100.Clip("Neutral", "FW", 807)
SECONDS = 20.0
OUT = Path(__file__).parent / "motions" / "100style_neutral_fw.csv"

#: G1 body -> the BVH joint it follows, then its (position, orientation) costs in GMR's
#: two IK passes: `bvh_lafan1_to_g1.json`'s, LAFAN1's joints named as 100STYLE's.
MATCH: dict[str, tuple[str, tuple[int, int], tuple[int, int]]] = {
    "pelvis": ("Hips", (0, 10), (100, 5)),
    "torso_link": ("Chest4", (0, 100), (0, 10)),
    **{
        f"{side}_{body}": (f"{Side}{joint}", first, second)
        for side, Side in (("left", "Left"), ("right", "Right"))
        for body, joint, first, second in (
            ("hip_yaw_link", "Hip", (0, 10), (10, 5)),
            ("knee_link", "Knee", (0, 10), (10, 5)),
            ("ankle_roll_link", "Ankle", (50, 10), (50, 10)),
            ("shoulder_yaw_link", "Shoulder", (0, 100), (5, 100)),
            ("elbow_link", "Elbow", (0, 10), (10, 5)),
            ("wrist_yaw_link", "Wrist", (0, 10), (10, 5)),
        )
    },
}
#: The G1's arms held out level and straight, as 100STYLE's actor stands at rest.
TPOSE = {
    "left_shoulder_roll_joint": np.pi / 2,
    "right_shoulder_roll_joint": -np.pi / 2,
    "left_elbow_joint": np.pi / 2,
    "right_elbow_joint": np.pi / 2,
}
_CACHE = CACHE_DIR / "reaf" / "100style"


def _g1_xml() -> Path:
    """mjlab's G1, which is Menagerie's `g1_mjlab.xml` that Playground loads."""
    import mjlab.asset_zoo.robots.unitree_g1 as g1

    return Path(g1.__file__).parent / "xmls" / "g1.xml"


def _bvh() -> Path:
    """The clip's BVH, extracted from 100STYLE's archive into msm's cache once."""
    root = style100._CACHE / "bvh"
    if not (root / CLIP.bvh).exists():
        style100._fetch(root, {CLIP.bvh})
    return root / CLIP.bvh


def _tpose(model):
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for joint, angle in TPOSE.items():
        data.qpos[model.joint(joint).qposadr[0]] = angle
    mujoco.mj_kinematics(model, data)
    return data


def ik_config(model, rest: dict[str, np.ndarray]) -> dict:
    """GMR's IK table for the G1 following a BVH with the rest pose ``rest``."""
    from scipy.spatial.transform import Rotation as R

    data = _tpose(model)
    bodies = (
        *MATCH,
        "right_hip_pitch_link",
        "right_shoulder_pitch_link",
        "right_wrist_roll_link",
    )
    pos = {b: data.xpos[model.body(b).id] for b in bodies}
    rot = {b: data.xmat[model.body(b).id].reshape(3, 3) for b in MATCH}

    def length(points: dict, *chain: str) -> float:
        return sum(
            np.linalg.norm(points[a] - points[b]) for a, b in zip(chain, chain[1:])
        )

    # One scale for the legs, which the torso and the root's travel follow, one for the arms.
    leg = length(
        pos, "right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link"
    ) / length(rest, "RightHip", "RightKnee", "RightAnkle")
    arm = length(
        pos, "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_roll_link"
    ) / length(rest, "RightShoulder", "RightElbow", "RightWrist")
    is_arm = ("Shoulder", "Elbow", "Wrist")
    scale = {
        joint: arm if joint.endswith(is_arm) else leg for joint, _, _ in MATCH.values()
    }

    tables: tuple[dict, dict] = ({}, {})
    for body, (joint, *costs) in MATCH.items():
        # At rest a joint's frame is the world's, mapped: the offset turns it into the body's.
        quat = R.from_matrix(style100.BVH_TO_WORLD.T @ rot[body]).as_quat(
            scalar_first=True
        )
        human = scale[joint] * (rest[joint] - rest["Hips"])
        offset = rot[body].T @ ((pos[body] - pos["pelvis"]) - human)
        for table, (pos_cost, rot_cost) in zip(tables, costs):
            table[body] = [joint, pos_cost, rot_cost, offset.tolist(), quat.tolist()]
    return {
        "robot_root_name": "pelvis",
        "human_root_name": "Hips",
        "ground_height": 0.0,
        "human_height_assumption": 1.0,
        "human_scale_table": scale,
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "ik_match_table1": tables[0],
        "ik_match_table2": tables[1],
    }


def _foot_floor(model, data) -> float:
    """The lowest point of the G1's foot capsules."""
    lowest = np.inf
    for g in range(model.ngeom):
        name = model.geom(g).name
        if name.startswith(("left_foot", "right_foot")) and name.endswith("_collision"):
            radius, half = model.geom_size[g][:2]
            axis_z = abs(data.geom_xmat[g].reshape(3, 3)[2, 2])
            lowest = min(lowest, data.geom_xpos[g][2] - radius - half * axis_z)
    return lowest


def retarget() -> Path:
    import mink
    import mujoco
    from scipy.spatial.transform import Rotation as R

    gmr = style100._gmr()
    from general_motion_retargeting import motion_retarget
    from general_motion_retargeting.utils.soma import load_soma_bvh_file

    bvh = _bvh()
    source, _ = load_soma_bvh_file(bvh)
    frames = slice(CLIP.start, CLIP.start + round(SECONDS * style100.BVH_FPS))
    source = source[frames][:: style100.BVH_FPS // style100.FIT_FPS]

    motion_retarget.ROBOT_XML_DICT["unitree_g1"] = _g1_xml()
    model = mujoco.MjModel.from_xml_path(str(_g1_xml()))
    config = _CACHE / "ik_config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(ik_config(model, style100._rest_pose(bvh)), indent=1))
    with contextlib.redirect_stdout(io.StringIO()):
        fit = gmr.GeneralMotionRetargeting(
            src_human="bvh_100style",
            tgt_robot="unitree_g1",
            ik_config_path=str(config),
            verbose=False,
        )
    fit.ik_limits = [mink.ConfigurationLimit(fit.model)]
    fit.setup_retarget_configuration()

    # Start on the first frame's root pose: the G1's, turned and moved with the hips.
    q = fit.model.qpos0.copy()
    hips_pos, hips_quat = source[0]["Hips"]
    turn = (
        R.from_quat(hips_quat, scalar_first=True)
        * R.from_matrix(style100.BVH_TO_WORLD).inv()
    )
    q[:3] = hips_pos * fit.human_scale_table["Hips"]
    q[3:7] = (turn * R.from_quat(q[3:7], scalar_first=True)).as_quat(scalar_first=True)
    fit.configuration.update(q)

    placed = [body for body, (_, _, (pos_cost, _)) in MATCH.items() if pos_cost]
    placed_rows = [k for k, body in enumerate(MATCH) if body in placed]
    qpos, residual = [], []
    for frame in source:
        q, error = fit.retarget(frame)
        qpos.append(q)
        residual.append(error[placed_rows])
    qpos = np.stack(qpos)

    # From the origin, facing +x: the first frame's heading turned away.
    heading = R.from_quat(qpos[0, 3:7], scalar_first=True).as_euler("ZYX")[0]
    unturn = R.from_euler("z", -heading)
    qpos[:, :2] -= qpos[0, :2]
    qpos[:, :3] = unturn.apply(qpos[:, :3])
    qpos[:, 3:7] = (unturn * R.from_quat(qpos[:, 3:7], scalar_first=True)).as_quat(
        scalar_first=True
    )
    # One height, so that the feet stand on the floor at the median frame.
    data = mujoco.MjData(fit.model)
    lowest = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(fit.model, data)
        lowest.append(_foot_floor(fit.model, data))
    qpos[:, 2] -= np.median(lowest)

    worst = np.max(np.stack(residual), axis=0)
    print(
        f"{CLIP.name}, {SECONDS:g} s: onto the G1, at worst "
        + ", ".join(f"{b} {100 * e:.1f} cm" for b, e in zip(placed, worst))
    )
    rows = np.concatenate(
        [
            qpos[:, :3],
            R.from_quat(qpos[:, 3:7], scalar_first=True).as_quat(),
            qpos[:, 7:],
        ],
        axis=1,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT, rows, delimiter=",", fmt="%.6f")
    return OUT


if __name__ == "__main__":
    print(retarget())
