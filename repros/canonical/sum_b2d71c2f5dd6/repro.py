"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g174
Pattern hash: b2d71c2f5dd6
Shape hash: 4ba12473
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_45: "f16[48, 476, 476]", arg31_1: "f16[48, 476, 476]", arg27_1: "f32[4, 1, 1, 476]", arg32_1: "f32[4, 12, 476, 1]", arg33_1: "f32[4, 12, 476, 1]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 12, 476, 476]" = torch.ops.aten.reshape.default(bmm_45, [4, 12, 476, 476]);  bmm_45 = None
        convert_element_type_default: "f32[4, 12, 476, 476]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        reshape_default_1: "f16[4, 12, 476, 476]" = torch.ops.aten.reshape.default(arg31_1, [4, 12, 476, 476]);  arg31_1 = None
        div_tensor: "f16[4, 12, 476, 476]" = torch.ops.aten.div.Tensor(reshape_default_1, 8.0);  reshape_default_1 = None
        add_tensor: "f32[4, 12, 476, 476]" = torch.ops.aten.add.Tensor(div_tensor, arg27_1);  div_tensor = arg27_1 = None
        sub_tensor: "f32[4, 12, 476, 476]" = torch.ops.aten.sub.Tensor(add_tensor, arg32_1);  add_tensor = arg32_1 = None
        exp_default: "f32[4, 12, 476, 476]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor_1: "f32[4, 12, 476, 476]" = torch.ops.aten.div.Tensor(exp_default, arg33_1);  exp_default = arg33_1 = None
        mul_tensor: "f32[4, 12, 476, 476]" = torch.ops.aten.mul.Tensor(convert_element_type_default, div_tensor_1);  convert_element_type_default = None
        sum_dim_int_list: "f32[4, 12, 476, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[4, 12, 476, 476]" = torch.ops.aten.neg.default(div_tensor_1);  div_tensor_1 = None
        fma_default: "f32[4, 12, 476, 476]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_1: "f16[4, 12, 476, 476]" = torch.ops.prims.convert_element_type.default(fma_default, torch.float16);  fma_default = None
        div_tensor_2: "f16[4, 12, 476, 476]" = torch.ops.aten.div.Tensor(convert_element_type_default_1, 8.0);  convert_element_type_default_1 = None
        reshape_default_2: "f16[48, 476, 476]" = torch.ops.aten.reshape.default(div_tensor_2, [48, 476, 476]);  div_tensor_2 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([48, 476, 476], dtype=torch.float16, device='cuda'),
    torch.randn([48, 476, 476], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1, 1, 476], dtype=torch.float32, device='cuda'),
    torch.randn(22896, dtype=torch.float32, device='cuda').as_strided([4, 12, 476, 1], [5728, 476, 1, 1]),  # arg32_1
    torch.randn(22896, dtype=torch.float32, device='cuda').as_strided([4, 12, 476, 1], [5728, 476, 1, 1]),  # arg33_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
