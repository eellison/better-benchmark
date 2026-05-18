"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g35
Pattern hash: 224060d2d472
Shape hash: c714df5e
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
    def forward(self, mm_177: "f16[4096, 512]", mm_179: "f16[4096, 512]", mm_181: "f16[4096, 512]", arg3_1: "f32[512]", arg54_1: "f32[4, 1024, 512]", arg55_1: "f32[4, 1024, 1]", add_97: "f32[4, 1024, 512]", arg53_1: "b8[4, 1024, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_177, _shape_param_0);  mm_177 = _shape_param_0 = None
        convert_element_type_default: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        reshape_default_1: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_179, _shape_param_1);  mm_179 = _shape_param_1 = None
        convert_element_type_default_1: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        add_tensor: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_1);  convert_element_type_default = convert_element_type_default_1 = None
        reshape_default_2: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_181, _shape_param_2);  mm_181 = _shape_param_2 = None
        convert_element_type_default_2: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        add_tensor_1: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_2);  add_tensor = convert_element_type_default_2 = None
        mul_tensor: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg3_1);  add_tensor_1 = arg3_1 = None
        mul_tensor_1: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg54_1)
        mul_tensor_2: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg55_1);  mul_tensor = None
        sum_dim_int_list: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        add_tensor_2: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_97, mul_tensor_2);  add_97 = mul_tensor_2 = None
        pow_tensor_scalar: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg55_1, 3);  arg55_1 = None
        mul_scalar: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_3, _shape_param_3);  mul_tensor_3 = _shape_param_3 = None
        div_scalar: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg54_1, 1.0);  arg54_1 = None
        mul_scalar_1: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_3: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_4);  add_tensor_2 = mul_tensor_4 = None
        convert_element_type_default_3: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None
        convert_element_type_default_4: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg53_1, torch.float16);  arg53_1 = None
        mul_tensor_5: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.1111111111111112);  convert_element_type_default_4 = None
        mul_tensor_6: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, mul_tensor_5);  convert_element_type_default_3 = mul_tensor_5 = None
        reshape_default_3: "f16[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_4);  mul_tensor_6 = _shape_param_4 = None
        return reshape_default_3


def _default_make_inputs():
    return [
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1024, 512], dtype=torch.bool, device='cuda'),
    [4, 1024, 512],  # _shape_param_0
    [4, 1024, 512],  # _shape_param_1
    [4, 1024, 512],  # _shape_param_2
    [4, 1024, 512],  # _shape_param_3
    [4096, 512],  # _shape_param_4
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
