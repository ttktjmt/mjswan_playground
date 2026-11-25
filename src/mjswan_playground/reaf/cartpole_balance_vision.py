"""`CartpoleBalanceVision`: the cart balances the pole from the fixed camera's pixels
alone, its last three frames in grayscale."""

from __future__ import annotations

import math
from typing import Any

import mjswan
import onnx
import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as lift_mdp
from onnx import TensorProto, helper

from . import _cartpole

#: Trained by `train.py` on Playground 1.1.0's `CartpoleBalanceVision`, all 500 iterations.
POLICY: tuple[str, str] | None = (
    "ttktjmt/cartpole-balance-vision",
    "18888ec9c2da560ed0a1256e58c208092fc554fa",
)


def _zeros(env: Any) -> torch.Tensor:
    """Playground's `dummy` group: the MLP's own input, empty beside the image."""
    return torch.zeros(env.num_envs, 1, device=env.device)


def _grayscale(env: Any, sensor_name: str) -> torch.Tensor:
    """Playground's `_obs_grayscale`, for the newest frame: the mean of the three
    channels, centred on 0."""
    return lift_mdp.camera_rgb(env, sensor_name).mean(dim=1, keepdim=True) - 0.5


def _fallen(env: Any) -> torch.Tensor:
    """Playground's termination: the pole past horizontal, or the cart off its rail."""
    slider, hinge = env.scene["cartpole"].data.joint_pos.unbind(-1)
    return (hinge.abs() > math.pi / 2) | (slider.abs() > 3.0)


def env_cfg() -> ManagerBasedRlEnvCfg:
    """`CartpoleBalance` at 50 Hz, seen through the fixed camera at 64 x 64 without
    shadows, and reset once the pole falls or 5 s pass. The observation is the newest
    frame; `stacked_policy` keeps the two before it. three.js draws the frames, not
    Playground's ray tracer, so their shading differs from what the policy trained on."""
    cfg = _cartpole.env_cfg(swing_up=False)
    cfg.scene.sensors = (
        CameraSensorCfg(
            name="fixed",
            camera_name="cartpole/fixed",
            width=64,
            height=64,
            data_types=("rgb",),
            use_shadows=False,
        ),
    )
    cfg.observations = {
        "dummy": ObservationGroupCfg({"zeros": ObservationTermCfg(func=_zeros)}),
        "pixels": ObservationGroupCfg(
            {
                "frame": ObservationTermCfg(
                    func=_grayscale, params={"sensor_name": "fixed"}
                )
            }
        ),
    }
    cfg.terminations["fallen"] = TerminationTermCfg(func=_fallen)
    cfg.decimation = 2
    cfg.episode_length_s = 5.0
    return cfg


def stacked_policy(policy: onnx.ModelProto) -> onnx.ModelProto:
    """Playground's policy with its frame stack carried in `adapt_hx`.

    Playground feeds the policy the last three frames, oldest first. An image term with
    history never runs in mjswan 0.13.1 (its build declares no camera for it), so the
    scene observes the newest frame and this graph keeps the two before it, filling
    them with the newest on an episode's first step, as Playground's history does.
    Inputs: `obs` [1, 1], `frame` [1, 1, 64, 64], `is_init` [1], `adapt_hx` [1, 8192].
    Outputs: `actions` [1, 1], `next_adapt_hx` [1, 8192].
    """
    model = onnx.ModelProto()
    model.CopyFrom(policy)
    graph = model.graph
    stack_input = next(i for i in graph.input if i.name == "policy")
    graph.input.remove(stack_input)
    graph.input.extend(
        [
            helper.make_tensor_value_info("frame", TensorProto.FLOAT, [1, 1, 64, 64]),
            helper.make_tensor_value_info("is_init", TensorProto.BOOL, [1]),
            helper.make_tensor_value_info("adapt_hx", TensorProto.FLOAT, [1, 8192]),
        ]
    )
    graph.initializer.extend(
        [
            helper.make_tensor(
                "stack_hx_shape", TensorProto.INT64, [4], [1, 2, 64, 64]
            ),
            helper.make_tensor("stack_flat_shape", TensorProto.INT64, [2], [1, 8192]),
            helper.make_tensor("stack_from", TensorProto.INT64, [1], [1]),
            helper.make_tensor("stack_to", TensorProto.INT64, [1], [3]),
            helper.make_tensor("stack_axis", TensorProto.INT64, [1], [1]),
        ]
    )
    stack = [
        helper.make_node("Reshape", ["adapt_hx", "stack_hx_shape"], ["stack_held"]),
        helper.make_node("Concat", ["frame", "frame"], ["stack_fill"], axis=1),
        helper.make_node(
            "Where", ["is_init", "stack_fill", "stack_held"], ["stack_prev"]
        ),
        # Playground's own input, now made here: oldest frame first.
        helper.make_node("Concat", ["stack_prev", "frame"], ["policy"], axis=1),
        helper.make_node(
            "Slice",
            ["policy", "stack_from", "stack_to", "stack_axis"],
            ["stack_next"],
        ),
        helper.make_node(
            "Reshape", ["stack_next", "stack_flat_shape"], ["next_adapt_hx"]
        ),
    ]
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(stack + nodes)
    graph.output.append(
        helper.make_tensor_value_info("next_adapt_hx", TensorProto.FLOAT, [1, 8192])
    )
    onnx.checker.check_model(model)
    return model


def add_scenes(project: mjswan.ProjectHandle) -> None:
    scene = project.add_scene_mjlab(
        "CartpoleBalanceVision", name="CartpoleBalanceVision", env_cfg=env_cfg()
    )
    if POLICY is None:
        return
    from huggingface_hub import hf_hub_download

    repo, revision = POLICY
    path = hf_hub_download(repo, "policy.onnx", revision=revision)
    scene.add_policy(
        policy=stacked_policy(onnx.load(path)),
        name="Balance Vision",
        in_keys=["dummy", "pixels", "is_init", "adapt_hx"],
        out_keys=["action", ["next", "adapt_hx"]],
        policy_joint_names=["cartpole/slider"],
        default_joint_pos=[0.0],
        default=True,
    )
