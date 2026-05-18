"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g36
Pattern hash: 305bac0b85dc
Shape hash: 49e09ea5
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
    def forward(self, mm: "f16[512, 768]", arg27_1: "f32[768]", arg195_1: "f16[512, 768]", arg196_1: "f32[1, 512, 1]", arg197_1: "f32[1, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        reshape_default: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type_default: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg27_1);  convert_element_type_default = arg27_1 = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        reshape_default_1: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(arg195_1, _shape_param_1);  arg195_1 = _shape_param_1 = None
        convert_element_type_default_1: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        mul_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.5)
        mul_tensor_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.7071067811865476)
        erf_default: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor);  mul_tensor_2 = None
        convert_element_type_default_2: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float32);  mul_tensor_4 = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, arg196_1);  convert_element_type_default_2 = arg196_1 = None
        mul_tensor_5: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, arg197_1);  sub_tensor = None
        mul_tensor_6: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_5);  mul_tensor = None
        sum_dim_int_list_1: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2], True);  mul_tensor_6 = None
        mul_tensor_7: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_5, sum_dim_int_list_1);  mul_tensor_5 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_7);  sub_tensor_1 = mul_tensor_7 = None
        div_tensor: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(arg197_1, 768);  arg197_1 = None
        mul_tensor_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        convert_element_type_default_3: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.float32);  mul_tensor_8 = None
        mul_tensor_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default_1)
        mul_tensor_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_10, -0.5);  mul_tensor_10 = None
        exp_default: "f32[1, 512, 768]" = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        mul_tensor_12: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_13: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, mul_tensor_12);  convert_element_type_default_1 = mul_tensor_12 = None
        add_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_9, mul_tensor_13);  mul_tensor_9 = mul_tensor_13 = None
        mul_tensor_14: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, add_tensor_1);  convert_element_type_default_3 = add_tensor_1 = None
        convert_element_type_default_4: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_14, torch.float16);  mul_tensor_14 = None
        reshape_default_2: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_4, _shape_param_2);  convert_element_type_default_4 = _shape_param_2 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    [1, 512, 768],  # _shape_param_0
    [1, 512, 768],  # _shape_param_1
    [512, 768],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
