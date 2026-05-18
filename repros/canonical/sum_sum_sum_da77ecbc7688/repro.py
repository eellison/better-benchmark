"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s8_g27
Pattern hash: da77ecbc7688
Shape hash: ed0cf466
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
    def forward(self, cat: "f16[256, 2]", mm_1: "f16[2, 1024]", where: "f16[256, 1024]", mm_3: "f16[1024, 1024]", where_1: "f16[256, 1024]", mm_4: "f16[1024, 3]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f16[1, 2]" = torch.ops.aten.sum.dim_IntList(cat, [0], True);  cat = None
        reshape_default: "f16[2]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        convert_element_type_default: "f32[2, 1024]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_default_1: "f32[2]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        sum_dim_int_list_1: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        reshape_default_1: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        convert_element_type_default_2: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_default_3: "f32[1024]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        sum_dim_int_list_2: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
        reshape_default_2: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_2);  sum_dim_int_list_2 = _shape_param_2 = None
        convert_element_type_default_4: "f32[1024, 3]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_default_5: "f32[1024]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        _output_to_half_0: "f16[2, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float16);  convert_element_type_default = None
        _output_to_half_1: "f16[2]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float16);  convert_element_type_default_1 = None
        _output_to_half_2: "f16[1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float16);  convert_element_type_default_2 = None
        _output_to_half_3: "f16[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_3, torch.float16);  convert_element_type_default_3 = None
        _output_to_half_4: "f16[1024, 3]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float16);  convert_element_type_default_4 = None
        _output_to_half_5: "f16[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float16);  convert_element_type_default_5 = None
        return (_output_to_half_0, _output_to_half_1, _output_to_half_2, _output_to_half_3, _output_to_half_4, _output_to_half_5)


def _default_make_inputs():
    return [
    torch.randn([256, 2], dtype=torch.float16, device='cuda'),
    torch.randn([2, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([256, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([256, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([1024, 3], dtype=torch.float16, device='cuda'),
    [2],  # _shape_param_0
    [1024],  # _shape_param_1
    [1024],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
