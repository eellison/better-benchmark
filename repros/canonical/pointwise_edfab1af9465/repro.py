"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-6-6-linux.aws.a100_graph30
Pattern hash: edfab1af9465
Shape hash: 1e7e7c93
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
    def forward(self, arg0_1: "f32[256, 1]"):
        # No stacktrace found for following nodes
        tanh_default: "f32[256, 1]" = torch.ops.aten.tanh.default(arg0_1);  arg0_1 = None
        return tanh_default


def _default_make_inputs():
    return [
    torch.randn([256, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
