"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g36
Pattern hash: 6c53be373419
Shape hash: 5ad63736
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f16[512, 768]", arg26_1: "f32[768]", arg193_1: "f32[1, 512, 768]", arg205_1: "f32[1, 512, 1]", arg192_1: "b8[1, 512, 768]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(mm_2, [1, 512, 768]);  mm_2 = None
        convert_element_type_default: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg26_1);  convert_element_type_default = arg26_1 = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg193_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg193_1, sum_dim_int_list_1);  arg193_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg205_1, sub_tensor_1);  arg205_1 = sub_tensor_1 = None
        convert_element_type_default_1: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float16);  mul_tensor_4 = None
        convert_element_type_default_2: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(arg192_1, torch.float16);  arg192_1 = None
        mul_tensor_5: "f16[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.1111111111111112);  convert_element_type_default_2 = None
        mul_tensor_6: "f16[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, mul_tensor_5);  convert_element_type_default_1 = mul_tensor_5 = None
        reshape_default_1: "f16[512, 768]" = torch.ops.aten.reshape.default(mul_tensor_6, [512, 768]);  mul_tensor_6 = None
        return reshape_default_1


def _default_make_inputs():
    return [
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
