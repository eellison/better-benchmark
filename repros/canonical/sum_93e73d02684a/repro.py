"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g160
Pattern hash: 93e73d02684a
Shape hash: e54fd3e7
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
    def forward(self, arg2_1: "f16[1896, 2]", mm: "f16[1896, 768]", mm_1: "f16[2, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(arg2_1, [0], True, dtype = torch.float32);  arg2_1 = None
        reshape_default: "f32[2]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        reshape_default_1: "f16[4, 474, 768]" = torch.ops.aten.reshape.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type_default: "f32[4, 474, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        convert_element_type_default_1: "f32[2, 768]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_default_2: "f32[2]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        _output_to_half_0: "f16[4, 474, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float16);  convert_element_type_default = None
        _output_to_half_1: "f16[2, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float16);  convert_element_type_default_1 = None
        _output_to_half_2: "f16[2]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float16);  convert_element_type_default_2 = None
        return (_output_to_half_0, _output_to_half_1, _output_to_half_2)


def _default_make_inputs():
    return [
    torch.randn([1896, 2], dtype=torch.float16, device='cuda'),
    torch.randn([1896, 768], dtype=torch.float16, device='cuda'),
    torch.randn([2, 768], dtype=torch.float16, device='cuda'),
    [2],  # _shape_param_0
    [4, 474, 768],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
