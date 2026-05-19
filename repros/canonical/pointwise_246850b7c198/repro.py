"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-2-9-linux.aws.h100_graph63
Pattern hash: 246850b7c198
Shape hash: cf89a5e3
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
    def forward(self, convolution_4: "bf16[256, 1, 1, 1]"):
        # No stacktrace found for following nodes
        sigmoid_default: "bf16[256, 1, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_4);  convolution_4 = None
        return sigmoid_default


def _default_make_inputs():
    return [
    torch.randn([256, 1, 1, 1], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
