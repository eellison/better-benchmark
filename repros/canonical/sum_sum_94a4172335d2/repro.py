"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g53
Pattern hash: 94a4172335d2
Shape hash: fa64cefc
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
    def forward(self, mm_94: "f16[1584, 768]", arg2_1: "f32[768]", arg39_1: "f32[8, 198, 768]", arg260_1: "f32[8, 198, 1]", add_44: "f32[8, 198, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        reshape_default: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(mm_94, _shape_param_0);  mm_94 = _shape_param_0 = None
        convert_element_type_default: "f32[8, 198, 768]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg2_1);  convert_element_type_default = arg2_1 = None
        mul_tensor_1: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg39_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(arg39_1, sum_dim_int_list_1);  arg39_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(arg260_1, sub_tensor_1);  arg260_1 = sub_tensor_1 = None
        add_tensor: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_44, mul_tensor_4);  add_44 = mul_tensor_4 = None
        convert_element_type_default_1: "f16[8, 198, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float16);  add_tensor = None
        reshape_default_1: "f16[1584, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        return reshape_default_1


def _default_make_inputs():
    return [
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    [8, 198, 768],  # _shape_param_0
    [1584, 768],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
