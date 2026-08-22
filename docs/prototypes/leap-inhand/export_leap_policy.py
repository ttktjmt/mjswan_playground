"""Export the upstream actor to ONNX: 320 -> 16, empirical normalizer folded in.

rsl-rl stores the actor as an MLP plus a running mean/std it applies to the observation.
The browser feeds the raw observation vector, so the normalizer has to travel inside the
graph rather than beside it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent / "in-hand-rotation-mjlab"
CHECKPOINT = REPO / "ckpts" / "leap_left_custom_model_4900.pt"
OUT = Path(__file__).resolve().parent / "leap_actor.onnx"

HIDDEN = (512, 512, 256)


class Actor(nn.Module):
    """`(obs - mean) / std` then the MLP, which is what rsl-rl's actor does at play."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, obs_dim: int, act_dim: int):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        layers: list[nn.Module] = []
        prev = obs_dim
        for width in HIDDEN:
            layers += [nn.Linear(prev, width), nn.ELU()]
            prev = width
        layers.append(nn.Linear(prev, act_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.mean) / self.std)


def export(path: Path = OUT) -> Path:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    actor_state = state["actor_state_dict"]
    mean = actor_state["obs_normalizer._mean"]
    std = actor_state["obs_normalizer._std"]
    obs_dim = int(mean.shape[-1])
    act_dim = int(actor_state["mlp.6.bias"].shape[0])

    model = Actor(mean, std, obs_dim, act_dim).eval()
    model.mlp.load_state_dict(
        {k[len("mlp.") :]: v for k, v in actor_state.items() if k.startswith("mlp.")}
    )

    torch.onnx.export(
        model,
        (torch.zeros(1, obs_dim),),
        str(path),
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamo=False,
    )

    # A graph that loads but computes something else is the failure mode worth ruling
    # out here, so check it against the torch module it came from.
    import onnxruntime as ort

    session = ort.InferenceSession(str(path))
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(32):
        sample = (mean.numpy() + std.numpy() * rng.standard_normal((1, obs_dim))).astype(
            np.float32
        )
        got = session.run(None, {"obs": sample})[0]
        want = model(torch.from_numpy(sample)).detach().numpy()
        worst = max(worst, float(np.abs(got - want).max()))

    print(f"exported {path.name}: {obs_dim} -> {act_dim}, {path.stat().st_size:,} bytes")
    print(f"onnxruntime vs torch, max |delta| over 32 samples: {worst:.2e}")
    return path


if __name__ == "__main__":
    export()
    onnx.checker.check_model(onnx.load(str(OUT)))
    print("onnx.checker: ok")
