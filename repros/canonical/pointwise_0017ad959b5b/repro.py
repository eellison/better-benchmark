"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s7_g10
Pattern hash: 0017ad959b5b
Shape hash: ef4120ef
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "bf16[16, 5]", arg3_1: "bf16[16, 3, 128, 128]"):
        # No stacktrace found for following nodes
        reshape_default: "bf16[16, 5, 1, 1]" = torch.ops.aten.reshape.default(arg1_1, [16, 5, 1, 1]);  arg1_1 = None
        repeat_default: "bf16[16, 5, 128, 128]" = torch.ops.aten.repeat.default(reshape_default, [1, 1, 128, 128]);  reshape_default = None
        cat_default: "bf16[16, 8, 128, 128]" = torch.ops.aten.cat.default([arg3_1, repeat_default], 1);  arg3_1 = repeat_default = None
        return cat_default


def _default_make_inputs():
    return [
    torch.randn([16, 5], dtype=torch.bfloat16, device='cuda'),
    torch.randn([16, 3, 128, 128], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
