"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g36
Pattern hash: c7a9ffed22e2
Shape hash: 822de3ee
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
    def forward(self, bmm_45: "f16[12, 512, 512]", arg42_1: "b8[1, 12, 512, 512]", arg39_1: "f16[12, 512, 512]", arg40_1: "f32[1, 12, 512, 1]", arg41_1: "f32[1, 12, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        reshape_default: "f16[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_45, _shape_param_0);  bmm_45 = _shape_param_0 = None
        convert_element_type_default: "f16[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.float16);  arg42_1 = None
        mul_tensor: "f16[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f16[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None
        convert_element_type_default_1: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float32);  mul_tensor_1 = None
        reshape_default_1: "f16[1, 12, 512, 512]" = torch.ops.aten.reshape.default(arg39_1, _shape_param_1);  arg39_1 = _shape_param_1 = None
        convert_element_type_default_2: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        mul_tensor_2: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        sub_tensor: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_2, arg40_1);  mul_tensor_2 = arg40_1 = None
        mul_tensor_3: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        exp_default: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        div_tensor: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, arg41_1);  exp_default = arg41_1 = None
        mul_tensor_4: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, div_tensor);  convert_element_type_default_1 = None
        sum_dim_int_list: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [-1], True)
        neg_default: "f32[1, 12, 512, 512]" = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default: "f32[1, 12, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_4);  neg_default = sum_dim_int_list = mul_tensor_4 = None
        convert_element_type_default_3: "f16[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_default, torch.float16);  fma_default = None
        mul_tensor_5: "f16[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 0.125);  convert_element_type_default_3 = None
        reshape_default_2: "f16[12, 512, 512]" = torch.ops.aten.reshape.default(mul_tensor_5, _shape_param_2);  mul_tensor_5 = _shape_param_2 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [1, 12, 512, 512], dtype=torch.bool, device='cuda'),
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([1, 12, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 12, 512, 1], dtype=torch.float32, device='cuda'),
    [1, 12, 512, 512],  # _shape_param_0
    [1, 12, 512, 512],  # _shape_param_1
    [12, 512, 512],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
