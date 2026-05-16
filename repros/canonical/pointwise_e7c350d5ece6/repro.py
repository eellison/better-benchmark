"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s4_g10
Pattern hash: e7c350d5ece6
Shape hash: 1de4870b
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_94: "bf16[8, 1280, 1, 1]", arg431_1: "bf16[1000, 1280]"):
        # No stacktrace found for following nodes
        relu_default: "bf16[8, 1280, 1, 1]" = torch.ops.aten.relu.default(convolution_94);  convolution_94 = None
        permute_default: "bf16[1280, 1000]" = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        reshape_default: "bf16[8, 1280]" = torch.ops.aten.reshape.default(relu_default, [8, 1280]);  relu_default = None
        return (permute_default, reshape_default)


def _default_make_inputs():
    return [
    torch.randn([8, 1280, 1, 1], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1000, 1280], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
