"""100STYLE clips retargeted onto MyoFullBody, the clips the policies track here.

100STYLE (Ian Mason, https://ianxmason.github.io/100style/, CC BY 4.0) is one actor's
Xsens capture of 100 locomotion styles at 60 fps: a BVH per style and gait, Y up, Z
forward, in centimetres, its rest pose a T-pose with every joint frame along the world
axes. GMR (MuscleMimic's fork, gmr_plus, MIT) fits MyoFullBody to it frame by frame.
Its IK table is made here rather than shipped: the costs are gmr_plus's own for SOMA's
BVH onto MyoFullBody, and the scales and offsets align the BVH's rest pose with
MyoFullBody in gmr_plus's T-pose (both shoulders elevated 90 deg), as its shape fitting
does for SMPL-H.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from mjswan_playground._deps import CACHE_DIR, ensure_repo

GMR_URL = "https://github.com/amathislab/gmr_plus"
GMR_COMMIT = "1b4eb9864360fa822b1290f34592da6393a4d496"
ROOT_ENV_VAR = "MJSWAN_100STYLE_ROOT"
#: 100STYLE v1 on Zenodo, and its archive of BVHs.
ZENODO_RECORD = "8127870"
ARCHIVE = "100STYLE.zip"


def notice(*, mshuman700: bool) -> str:
    """What CC BY 4.0 asks a copy of the clips to carry: the credit, and the changes."""
    onto = "MyoFullBody with GMR" + (
        ", and from there onto MS-Human-700" if mshuman700 else ""
    )
    return (
        "The motions are from The 100STYLE Dataset - Ian Mason "
        "(https://ianxmason.github.io/100style/), licensed under CC BY 4.0. They were "
        f"cut, retargeted onto {onto}, and resampled to 100 Hz."
    )


#: 100STYLE's gait codes -> the motion picker's groups.
GAITS = {"FW": "Walk forward", "SW": "Walk sideways"}
#: How long each clip is, in seconds.
CLIP_SECONDS = 8.0
#: 100STYLE's frame rate.
BVH_FPS = 60


@dataclasses.dataclass(frozen=True)
class Clip:
    style: str
    gait: str
    """A key of :data:`GAITS`."""
    start: int
    """The BVH frame it starts at."""

    @property
    def name(self) -> str:
        return f"{self.style}/{self.gait}_{self.start / BVH_FPS:.0f}s"

    @property
    def group(self) -> str:
        return GAITS[self.gait]

    @property
    def bvh(self) -> str:
        return f"{self.style}/{self.style}_{self.gait}.bvh"


#: The clips both policies play from their reset to their end in the browser with no
#: termination but the clip's end. Each starts where ``Frame_Cuts.csv`` says its capture
#: does, but the neutral forward walk's, 8 s on and at 69 s, where it turns least. The
#: policies learned on KIT walks at 0.4 to 0.7 m/s, and backwards only in straight lines,
#: so each of these ends on a deviation: the runs (1 to 1.3 m/s), the backward walks and
#: the other sideways ones, which 100STYLE's actor takes through turns at the edges of a
#: 4.5 m square, and the forward walks of Akimbo, ArmsAboveHead, ArmsFolded, Elated,
#: OnPhoneRight, Proud, Robot, Strutting and Zombie.
CLIPS: tuple[Clip, ...] = (
    Clip("Neutral", "FW", 807),
    Clip("Neutral", "FW", 4140),
    Clip("ArmsBehindBack", "FW", 433),
    Clip("ArmsBySide", "FW", 350),
    Clip("Depressed", "FW", 410),
    Clip("DuckFoot", "FW", 635),
    Clip("HandsInPockets", "FW", 421),
    Clip("LookUp", "FW", 411),
    Clip("Monk", "FW", 450),
    Clip("Old", "FW", 513),
    Clip("PigeonToed", "FW", 563),
    Clip("Neutral", "SW", 354),
)

#: MyoFullBody body -> the BVH joint it follows, then its (position, orientation) costs
#: in GMR's two IK passes: gmr_plus's ``bvh_soma_to_myofullbody.json``, whose first pass
#: weighs the left forearm and hand 40 where the right ones weigh 50; both 50 here.
MATCH: dict[str, tuple[str, tuple[int, int], tuple[int, int]]] = {
    "pelvis": ("Hips", (100, 10), (100, 5)),
    # MyoFullBody keeps its thorax rigid on L1; Chest3 sits at L1's height.
    "lumbar1": ("Chest3", (100, 10), (40, 5)),
    "head": ("Head", (100, 10), (45, 5)),
    **{
        f"{body}_{s}": (f"{side}{joint}", costs, second)
        for s, side in (("r", "Right"), ("l", "Left"))
        for body, joint, costs, second in (
            ("humerus", "Shoulder", (1, 10), (0, 5)),
            ("ulna", "Elbow", (50, 10), (40, 5)),
            ("lunate", "Wrist", (50, 10), (40, 5)),
            ("femur", "Hip", (0, 10), (0, 5)),
            ("tibia", "Knee", (0, 10), (0, 5)),
            ("calcn", "Ankle", (50, 10), (50, 5)),
            ("toes", "Toe", (50, 10), (50, 5)),
        )
    },
}
#: The BVH's axes in MuJoCo's (x forward, y left, z up), as gmr_plus's SOMA loader maps them.
BVH_TO_WORLD = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
#: The rate GMR fits at; the motion library resamples to the control rate.
FIT_FPS = 30
_CACHE = CACHE_DIR / "msm" / "100style"


def _gmr() -> Any:
    root = ensure_repo(
        name="msm/gmr_plus",
        url=GMR_URL,
        commit=GMR_COMMIT,
        marker="general_motion_retargeting/__init__.py",
        root_env_var="MJSWAN_GMR_ROOT",
        # The rest of the repository is robot assets, 1.4 GB of them.
        paths=("general_motion_retargeting",),
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import general_motion_retargeting

    return general_motion_retargeting


def _rest_pose(bvh: Path) -> dict[str, np.ndarray]:
    """Each joint's world position at rest (all rotations zero), in metres."""
    from general_motion_retargeting.utils.soma import _parse_bvh

    parsed = _parse_bvh(bvh)
    pos: list[np.ndarray] = []
    for offset, parent in zip(parsed["offsets"], parsed["parents"]):
        pos.append(offset / 100.0 + (pos[parent] if parent >= 0 else 0.0))
    return {name: BVH_TO_WORLD @ p for name, p in zip(parsed["names"], pos)}


def _tpose(model: Any) -> Any:
    """MyoFullBody in gmr_plus's T-pose, its couplers holding, facing +x."""
    import mujoco
    from muscle_supermimic.assets.robots.model_info import ModelInfo

    info = ModelInfo(model)
    q = np.zeros(len(info.indep_names))
    for name in ("shoulder_elv_r", "shoulder_elv_l"):
        q[info.indep_names.index(name)] = np.pi / 2
    data = mujoco.MjData(model)
    data.qpos[:7] = model.qpos0[:7]
    data.qpos[7:] = info.full_joint_pos(q)[0]
    mujoco.mj_kinematics(model, data)
    return data


def ik_config(model: Any, rest: dict[str, np.ndarray]) -> dict:
    """GMR's IK table for MyoFullBody following a BVH with the rest pose ``rest``."""
    from scipy.spatial.transform import Rotation as R

    data = _tpose(model)
    pos = {b: data.xpos[model.body(b).id] for b in (*MATCH, "talus_r")}
    rot = {b: data.xmat[model.body(b).id].reshape(3, 3) for b in MATCH}

    def length(points: dict, *chain: str) -> float:
        return sum(
            np.linalg.norm(points[a] - points[b]) for a, b in zip(chain, chain[1:])
        )

    # One scale for the legs, which the torso and the root's travel follow, and one for the arms.
    leg = length(pos, "femur_r", "tibia_r", "talus_r") / length(
        rest, "RightHip", "RightKnee", "RightAnkle"
    )
    arm = length(pos, "humerus_r", "ulna_r", "lunate_r") / length(
        rest, "RightShoulder", "RightElbow", "RightWrist"
    )
    scale = {
        joint: arm if body.split("_")[0] in ("humerus", "ulna", "lunate") else leg
        for body, (joint, _, _) in MATCH.items()
    }

    tables: tuple[dict, dict] = ({}, {})
    for body, (joint, *costs) in MATCH.items():
        # At rest a joint's frame is the world's, mapped: the offset turns it into the body's.
        quat = R.from_matrix(BVH_TO_WORLD.T @ rot[body]).as_quat(scalar_first=True)
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


def retarget(
    bvh: Path, out: Path, frames: slice = slice(None)
) -> dict[str, np.ndarray]:
    """One BVH's ``frames`` fitted onto MyoFullBody, saved at ``out`` as a MuscleMimic
    trajectory (what upstream's motion library and retargeter read). Returns, per body
    with a position target, its distance from the target in each frame."""
    import mink
    import mujoco
    from muscle_supermimic.assets.robots import myofullbody
    from muscle_supermimic.assets.robots.model_info import ModelInfo
    from muscle_supermimic.tasks.tracking.scripts.retarget_myofullbody import (
        FEET,
        geom_half_height,
    )
    from scipy.spatial.transform import Rotation as R

    gmr = _gmr()
    from general_motion_retargeting.utils.soma import load_soma_bvh_file

    # The model upstream's MyoFullBody task builds: no finger joints.
    model = myofullbody.compiled_model()
    source, _ = load_soma_bvh_file(bvh)
    if frames.stop is not None and frames.stop > len(source):
        raise ValueError(f"{bvh} has {len(source)} frames, not {frames.stop}")
    source = source[frames][:: BVH_FPS // FIT_FPS]

    config = _CACHE / "ik_config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(ik_config(model, _rest_pose(bvh)), indent=1))
    # It lists the model's joints, bodies and actuators whatever ``verbose`` says.
    with contextlib.redirect_stdout(io.StringIO()):
        fit = gmr.GeneralMotionRetargeting(
            src_human="bvh_100style",
            tgt_robot="myofullbody",
            ik_config_path=str(config),
            verbose=False,
        )
    fit.model = model
    fit.ik_limits = [mink.ConfigurationLimit(model)]
    fit.setup_retarget_configuration()

    # Start on the first frame's root pose: the T-pose's, turned and moved with the hips,
    # and the couplers holding, which qpos0's knee translations do not.
    info = ModelInfo(model)
    q = model.qpos0.copy()
    q[7:] = info.full_joint_pos(np.zeros(len(info.indep_names)))[0]
    hips_pos, hips_quat = source[0]["Hips"]
    turn = R.from_quat(hips_quat, scalar_first=True) * R.from_matrix(BVH_TO_WORLD).inv()
    q[:3] = hips_pos * fit.human_scale_table["Hips"]
    q[3:7] = (turn * R.from_quat(q[3:7], scalar_first=True)).as_quat(scalar_first=True)
    fit.configuration.update(q)

    # The second pass's frame tasks, in MATCH order: only those with a position cost.
    placed = [body for body, (_, _, (pos_cost, _)) in MATCH.items() if pos_cost]
    placed_rows = [k for k, body in enumerate(MATCH) if body in placed]
    qpos, residual = [], []
    for frame in source:
        q, error = fit.retarget(frame)
        qpos.append(q)
        residual.append(error[placed_rows])
    qpos = np.stack(qpos)

    # One height per clip, so that the feet stand on the floor at the median frame.
    data = mujoco.MjData(model)
    geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_contype[g] and model.body(model.geom_bodyid[g]).name in FEET
    ]
    lowest = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        lowest.append(
            min(data.geom_xpos[g, 2] - geom_half_height(model, data, g) for g in geoms)
        )
    qpos[:, 2] -= np.median(lowest)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        qpos=qpos,
        joint_names=np.array([model.joint(j).name for j in range(model.njnt)]),
        jnt_type=model.jnt_type.copy(),
        frequency=float(FIT_FPS),
    )
    return dict(zip(placed, np.stack(residual).T))


def archive() -> Path:
    """A directory laid out as the archive's ``100STYLE/``, with :data:`CLIPS`' BVHs:
    the one ``MJSWAN_100STYLE_ROOT`` points at, else the cache, filled from Zenodo."""
    configured = os.getenv(ROOT_ENV_VAR)
    if configured:
        return Path(configured).expanduser().resolve()
    root = _CACHE / "bvh"
    wanted = {clip.bvh for clip in CLIPS}
    if any(not (root / path).exists() for path in wanted):
        _fetch(root, wanted)
    return root


def _fetch(root: Path, wanted: set[str]) -> None:
    """Extract ``wanted`` from the record's BVH archive, checked against the checksum
    Zenodo lists for it; a Zenodo record's files never change."""
    api = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    with urllib.request.urlopen(api) as response:
        record = json.load(response)
    entry = next(f for f in record["files"] if f["key"] == ARCHIVE)
    algorithm, expected = entry["checksum"].split(":", 1)
    with tempfile.TemporaryFile() as download:
        digest = hashlib.new(algorithm)
        with urllib.request.urlopen(entry["links"]["self"]) as response:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                download.write(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(
                f"{ARCHIVE} has {algorithm} {digest.hexdigest()}, not {expected}."
            )
        with zipfile.ZipFile(download) as bundle:
            for path in sorted(wanted):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(f"100STYLE/{path}"))


def ensure_clips(*, mshuman700: bool) -> list[Path]:
    """:data:`CLIPS` as upstream's motion library reads them, retargeted once into the
    cache: onto MyoFullBody, and from there onto MS-Human-700 with upstream's own
    retargeter."""
    paths, retargeter = [], None
    for clip in CLIPS:
        stem = clip.name.replace("/", "_")
        myofullbody = _CACHE / "myofullbody" / f"{stem}.npz"
        if not myofullbody.exists():
            frames = slice(clip.start, clip.start + round(CLIP_SECONDS * BVH_FPS))
            residual = retarget(archive() / clip.bvh, myofullbody, frames)
            worst = max(residual, key=lambda body: residual[body].max())
            print(
                f"{clip.name}: onto MyoFullBody, {100 * residual[worst].max():.1f} cm "
                f"at worst ({worst})"
            )
        if not mshuman700:
            paths.append(myofullbody)
            continue
        target = _CACHE / "mshuman700" / f"{stem}.npz"
        if not target.exists():
            from muscle_supermimic.tasks.tracking.mdp.motion import load_clip
            from muscle_supermimic.tasks.tracking.scripts.retarget_myofullbody import (
                Retargeter,
            )

            retargeter = retargeter or Retargeter()
            fps, p, quat, q = load_clip(
                str(myofullbody), retargeter.mf_info, retargeter.mf_R_off
            )
            p, q, _ = retargeter.clip(p, quat, q)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                target,
                fps=fps,
                root_pos=p,
                root_quat=quat,
                joint_pos=q,
                joint_names=np.array(retargeter.ms_info.indep_names),
            )
        paths.append(target)
    return paths
