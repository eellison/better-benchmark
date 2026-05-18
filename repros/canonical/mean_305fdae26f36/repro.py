"""
Standalone repro captured via capture_hook.
Label: tritonbench_mean_dim1_4096x4096
Pattern hash: 305fdae2e5ef
Shape hash: 6f365176
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[4096, 4096]"):
        # No stacktrace found for following nodes
        mean_dim: "f32[4096]" = torch.ops.aten.mean.dim(arg0_1, [-1]);  arg0_1 = None
        return mean_dim


def _default_make_inputs():
    return [
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
