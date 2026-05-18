"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s3_g26
Pattern hash: c6d2eebde988
Shape hash: ed4c930f
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
    def forward(self, mm: "f16[128, 192]", arg27_1: "f32[192]", arg188_1: "f32[128, 197, 192]", arg191_1: "f32[128, 197, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[128, 192]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        full_default: "f32[128, 197, 192]" = torch.ops.aten.full.default([128, 197, 192], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[128, 197, 192]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default, 1, 0);  full_default = convert_element_type_default = None
        mul_tensor: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(select_scatter_default, arg27_1);  select_scatter_default = arg27_1 = None
        mul_tensor_1: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, 192)
        sum_dim_int_list: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, arg188_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(arg188_1, sum_dim_int_list_1);  arg188_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(arg191_1, sub_tensor_1);  arg191_1 = sub_tensor_1 = None
        convert_element_type_default_1: "f16[128, 197, 192]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float16);  mul_tensor_4 = None
        reshape_default: "f16[25216, 192]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_0);  convert_element_type_default_1 = _shape_param_0 = None
        return reshape_default


def _default_make_inputs():
    return [
    torch.randn([128, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 1], dtype=torch.float32, device='cuda'),
    [25216, 192],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
