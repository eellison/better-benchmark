"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s8_g21
Pattern hash: 4cbcc9780522
Shape hash: adb46d13
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
    def forward(self, addmm_1: "f16[256, 1024]", arg7_1: "f32[2]", arg6_1: "f32[2, 1024]"):
        # No stacktrace found for following nodes
        relu_default: "f16[256, 1024]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        convert_element_type_default: "f16[2]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float16);  arg7_1 = None
        convert_element_type_default_1: "f16[2, 1024]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float16);  arg6_1 = None
        permute_default: "f16[1024, 2]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (relu_default, convert_element_type_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([256, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([2], dtype=torch.float32, device='cuda'),
    torch.randn([2, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
