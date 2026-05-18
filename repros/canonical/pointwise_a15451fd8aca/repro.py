"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g111
Pattern hash: a15451fd8aca
Shape hash: 382357b5
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
    def forward(self, arg1_1: "f32[2]", arg0_1: "f32[2, 768]", arg2_1: "f32[4, 474, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f16[2]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float16);  arg1_1 = None
        convert_element_type_default_1: "f16[2, 768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float16);  arg0_1 = None
        convert_element_type_default_2: "f16[4, 474, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float16);  arg2_1 = None
        reshape_default: "f16[1896, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_2, _shape_param_0);  convert_element_type_default_2 = _shape_param_0 = None
        permute_default: "f16[768, 2]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (convert_element_type_default, reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([2], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 474, 768], dtype=torch.float32, device='cuda'),
    [1896, 768],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
