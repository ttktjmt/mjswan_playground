"""The SO-101 as Playground's ``robots/so101/sim`` models it: Menagerie's arm with the
servo dynamics identified on the real one."""

from __future__ import annotations

from pathlib import Path

import mujoco

from ._common import MENAGERIE, fetch, load, name_unnamed_geoms

#: Per joint: kp, kd, armature, frictionloss, identified from real teleop data.
DYNAMICS = {
    "shoulder_pan": (251.5896, 17.5096, 0.1088, 0.0001),
    "shoulder_lift": (265.0958, 28.0620, 0.1472, 0.0),
    "elbow_flex": (243.9844, 25.6669, 0.1224, 0.0),
    "wrist_flex": (223.2804, 25.4543, 0.1133, 0.0),
    "wrist_roll": (218.4066, 23.1058, 0.1146, 0.0),
    "gripper": (223.3508, 34.1633, 0.1985, 0.2517),
}
#: STS3215 stall torque: 30 kg cm at 12 V.
FORCE_LIMIT = 30 * 9.81 / 100

#: Joint positions of the home pose each episode starts from.
HOME = {
    "shoulder_pan": -0.192,
    "shoulder_lift": -0.262,
    "elbow_flex": 0.253,
    "wrist_flex": 1.13,
    "wrist_roll": 0.0793,
    "gripper": 0.80,
}


def spec() -> mujoco.MjSpec:
    arm = name_unnamed_geoms(load(MENAGERIE, "robotstudio_so101/so101.xml"))
    for joint, (kp, kd, armature, frictionloss) in DYNAMICS.items():
        actuator = arm.actuator(joint)
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm[0] = kp
        actuator.biasprm[1] = -kp
        actuator.biasprm[2] = -kd
        actuator.forcelimited = True
        actuator.forcerange = [-FORCE_LIMIT, FORCE_LIMIT]
        arm.joint(joint).armature = armature
        arm.joint(joint).frictionloss = frictionloss
    # mjlab attaches the arm to a scene on default options; others would warn.
    arm.option = mujoco.MjSpec().option
    return arm


def license_file() -> Path:
    return fetch(MENAGERIE, "robotstudio_so101/LICENSE")
