"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g35
Pattern hash: 54cee2efbfb6
Shape hash: 0bc32d96
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
    def forward(self, mm_173: "f16[4096, 512]", arg4_1: "f32[512]", arg61_1: "f32[4, 1024, 512]", arg62_1: "f32[4, 1024, 1]", add_95: "f32[4, 1024, 512]", arg60_1: "b8[4, 1024, 512]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_173, [4, 1024, 512]);  mm_173 = None
        convert_element_type_default: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg4_1);  convert_element_type_default = arg4_1 = None
        mul_tensor_1: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg61_1)
        mul_tensor_2: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg62_1);  mul_tensor = None
        sum_dim_int_list: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        add_tensor: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_95, mul_tensor_2);  add_95 = mul_tensor_2 = None
        pow_tensor_scalar: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg62_1, 3);  arg62_1 = None
        mul_scalar: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_3, [4, 1024, 512]);  mul_tensor_3 = None
        div_scalar: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg61_1, 1.0);  arg61_1 = None
        mul_scalar_1: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_1: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_4);  add_tensor = mul_tensor_4 = None
        convert_element_type_default_1: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        convert_element_type_default_2: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg60_1, torch.float16);  arg60_1 = None
        mul_tensor_5: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.1111111111111112);  convert_element_type_default_2 = None
        mul_tensor_6: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, mul_tensor_5);  convert_element_type_default_1 = mul_tensor_5 = None
        reshape_default_1: "f16[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_6, [4096, 512]);  mul_tensor_6 = None
        return reshape_default_1


def _default_make_inputs():
    return [
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1024, 512], dtype=torch.bool, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
