"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g36
Pattern hash: d499127dcf33
Shape hash: f558c72a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_130: "f16[512, 768]", mul_327: "f32[1, 512, 768]", mm_132: "f16[512, 768]", mm_134: "f16[512, 768]", arg4_1: "f32[768]", arg50_1: "f32[1, 512, 768]", arg337_1: "f32[1, 512, 1]", arg49_1: "b8[1, 512, 768]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(mm_130, [1, 512, 768]);  mm_130 = None
        convert_element_type_default: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        add_tensor: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_327, convert_element_type_default);  mul_327 = convert_element_type_default = None
        reshape_default_1: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(mm_132, [1, 512, 768]);  mm_132 = None
        convert_element_type_default_1: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        add_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_1);  add_tensor = convert_element_type_default_1 = None
        reshape_default_2: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(mm_134, [1, 512, 768]);  mm_134 = None
        convert_element_type_default_2: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        add_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_2);  add_tensor_1 = convert_element_type_default_2 = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_2, arg4_1);  add_tensor_2 = arg4_1 = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg50_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg50_1, sum_dim_int_list_1);  arg50_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg337_1, sub_tensor_1);  arg337_1 = sub_tensor_1 = None
        convert_element_type_default_3: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float16);  mul_tensor_4 = None
        convert_element_type_default_4: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(arg49_1, torch.float16);  arg49_1 = None
        mul_tensor_5: "f16[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.1111111111111112);  convert_element_type_default_4 = None
        mul_tensor_6: "f16[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, mul_tensor_5);  convert_element_type_default_3 = mul_tensor_5 = None
        reshape_default_3: "f16[512, 768]" = torch.ops.aten.reshape.default(mul_tensor_6, [512, 768]);  mul_tensor_6 = None
        return reshape_default_3


def _default_make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 512, 768], dtype=torch.bool, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
