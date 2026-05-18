"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g53
Pattern hash: 7e71e05ce650
Shape hash: c7ee3b25
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
    def forward(self, arg9_1: "f16[4, 1]", mm: "f16[1, 16]", convert_element_type_7: "f16[4, 16]", mm_2: "f16[16, 16]", convert_element_type_12: "f16[4, 16]", mm_4: "f16[16, 16]", convert_element_type_17: "f16[4, 16]", mm_6: "f16[16, 16]", convert_element_type_22: "f16[4, 16]", mm_7: "f16[16, 1]"):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f16[1, 1]" = torch.ops.aten.sum.dim_IntList(arg9_1, [0], True);  arg9_1 = None
        reshape_default: "f16[1]" = torch.ops.aten.reshape.default(sum_dim_int_list, [1]);  sum_dim_int_list = None
        convert_element_type_default: "f32[1, 16]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_default_1: "f32[1]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        sum_dim_int_list_1: "f16[1, 16]" = torch.ops.aten.sum.dim_IntList(convert_element_type_7, [0], True);  convert_element_type_7 = None
        reshape_default_1: "f16[16]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [16]);  sum_dim_int_list_1 = None
        convert_element_type_default_2: "f32[16, 16]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_default_3: "f32[16]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        sum_dim_int_list_2: "f16[1, 16]" = torch.ops.aten.sum.dim_IntList(convert_element_type_12, [0], True);  convert_element_type_12 = None
        reshape_default_2: "f16[16]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [16]);  sum_dim_int_list_2 = None
        convert_element_type_default_4: "f32[16, 16]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_default_5: "f32[16]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        sum_dim_int_list_3: "f16[1, 16]" = torch.ops.aten.sum.dim_IntList(convert_element_type_17, [0], True);  convert_element_type_17 = None
        reshape_default_3: "f16[16]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [16]);  sum_dim_int_list_3 = None
        convert_element_type_default_6: "f32[16, 16]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_default_7: "f32[16]" = torch.ops.prims.convert_element_type.default(reshape_default_3, torch.float32);  reshape_default_3 = None
        sum_dim_int_list_4: "f16[1, 16]" = torch.ops.aten.sum.dim_IntList(convert_element_type_22, [0], True);  convert_element_type_22 = None
        reshape_default_4: "f16[16]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [16]);  sum_dim_int_list_4 = None
        convert_element_type_default_8: "f32[16, 1]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_default_9: "f32[16]" = torch.ops.prims.convert_element_type.default(reshape_default_4, torch.float32);  reshape_default_4 = None
        return (convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_7, convert_element_type_default_8, convert_element_type_default_9)


def _default_make_inputs():
    return [
    torch.randn([4, 1], dtype=torch.float16, device='cuda'),
    torch.randn([1, 16], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16], dtype=torch.float16, device='cuda'),
    torch.randn([16, 16], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16], dtype=torch.float16, device='cuda'),
    torch.randn([16, 16], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16], dtype=torch.float16, device='cuda'),
    torch.randn([16, 16], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16], dtype=torch.float16, device='cuda'),
    torch.randn([16, 1], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
