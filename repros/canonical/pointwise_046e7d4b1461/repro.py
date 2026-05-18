"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s8_g21
Pattern hash: 046e7d4b1461
Shape hash: ec33362d
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
    def forward(self, arg1_1: "f32[1024]", arg0_1: "f32[1024, 3]", arg3_1: "f32[256, 3]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f16[1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float16);  arg1_1 = None
        convert_element_type_default_1: "f16[1024, 3]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float16);  arg0_1 = None
        convert_element_type_default_2: "f16[256, 3]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float16);  arg3_1 = None
        permute_default: "f16[3, 1024]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (convert_element_type_default, convert_element_type_default_2, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 3], dtype=torch.float32, device='cuda'),
    torch.randn([256, 3], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
