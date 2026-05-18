"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g174
Pattern hash: 48d1f3daac51
Shape hash: cb4d6917
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
    def forward(self, arg277_1: "f32[4, 476, 768]", mul_208: "f32[4, 476, 768]", mm_128: "f16[1904, 768]", mm_130: "f16[1904, 768]", mm_132: "f16[1904, 768]", arg4_1: "f32[768]", arg39_1: "f32[4, 476, 768]", arg263_1: "f32[4, 476, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        add_tensor: "f32[4, 476, 768]" = torch.ops.aten.add.Tensor(arg277_1, mul_208);  arg277_1 = mul_208 = None
        reshape_default: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(mm_128, _shape_param_0);  mm_128 = _shape_param_0 = None
        convert_element_type_default: "f32[4, 476, 768]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        add_tensor_1: "f32[4, 476, 768]" = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default);  add_tensor = convert_element_type_default = None
        reshape_default_1: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(mm_130, _shape_param_1);  mm_130 = _shape_param_1 = None
        convert_element_type_default_1: "f32[4, 476, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        add_tensor_2: "f32[4, 476, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_1);  add_tensor_1 = convert_element_type_default_1 = None
        reshape_default_2: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(mm_132, _shape_param_2);  mm_132 = _shape_param_2 = None
        convert_element_type_default_2: "f32[4, 476, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        add_tensor_3: "f32[4, 476, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_2);  add_tensor_2 = convert_element_type_default_2 = None
        mul_tensor: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_tensor_3, arg4_1);  add_tensor_3 = arg4_1 = None
        mul_tensor_1: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[4, 476, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg39_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 476, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(arg39_1, sum_dim_int_list_1);  arg39_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 476, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 476, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(arg263_1, sub_tensor_1);  arg263_1 = sub_tensor_1 = None
        convert_element_type_default_3: "f16[4, 476, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float16);  mul_tensor_4 = None
        reshape_default_3: "f16[1904, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_3, _shape_param_3);  convert_element_type_default_3 = _shape_param_3 = None
        return reshape_default_3


def _default_make_inputs():
    return [
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 1], dtype=torch.float32, device='cuda'),
    [4, 476, 768],  # _shape_param_0
    [4, 476, 768],  # _shape_param_1
    [4, 476, 768],  # _shape_param_2
    [1904, 768],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
