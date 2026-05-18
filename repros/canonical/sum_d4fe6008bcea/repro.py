"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g36
Pattern hash: d4fe6008bcea
Shape hash: 5e875a8c
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
    def forward(self, arg350_1: "f32[]", arg202_1: "f32[]", arg28_1: "i64[1, 512]", arg199_1: "f16[512, 30522]", arg200_1: "f32[512, 1]", arg201_1: "f32[512, 1]"):
        # No stacktrace found for following nodes
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(arg350_1, arg202_1);  arg350_1 = arg202_1 = None
        reshape_default: "i64[512]" = torch.ops.aten.reshape.default(arg28_1, [-1]);  arg28_1 = None
        unsqueeze_default: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(reshape_default, 1);  reshape_default = None
        ne_scalar: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[512, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default);  unsqueeze_default = full_default = None
        iota_default: "i64[30522]" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_1: "i64[1, 30522]" = torch.ops.aten.reshape.default(iota_default, [1, 30522]);  iota_default = None
        expand_default: "i64[512, 30522]" = torch.ops.aten.expand.default(where_self, [512, 30522]);  where_self = None
        eq_tensor: "b8[512, 30522]" = torch.ops.aten.eq.Tensor(expand_default, reshape_default_1);  expand_default = reshape_default_1 = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_self_1: "f32[512, 30522]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "f32[512, 1]" = torch.ops.aten.where.self(ne_scalar, div_tensor, full_default_1);  ne_scalar = div_tensor = full_default_1 = None
        mul_tensor: "f32[512, 30522]" = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None
        convert_element_type_default: "f32[512, 30522]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_1: "f32[512, 30522]" = torch.ops.prims.convert_element_type.default(arg199_1, torch.float32);  arg199_1 = None
        sub_tensor: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_default_1, arg200_1);  convert_element_type_default_1 = arg200_1 = None
        sub_tensor_1: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(sub_tensor, arg201_1);  sub_tensor = arg201_1 = None
        convert_element_type_default_2: "f16[512, 30522]" = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.float16);  sub_tensor_1 = None
        convert_element_type_default_3: "f32[512, 30522]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float32);  convert_element_type_default_2 = None
        exp_default: "f32[512, 30522]" = torch.ops.aten.exp.default(convert_element_type_default_3);  convert_element_type_default_3 = None
        sum_dim_int_list: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_default, [1], True)
        mul_tensor_1: "f32[512, 30522]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_default, mul_tensor_1);  convert_element_type_default = mul_tensor_1 = None
        convert_element_type_default_4: "f16[512, 30522]" = torch.ops.prims.convert_element_type.default(sub_tensor_2, torch.float16);  sub_tensor_2 = None
        return convert_element_type_default_4


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn(15630330, dtype=torch.float16, device='cuda').as_strided([512, 30522], [30528, 1]),  # arg199_1
    torch.randn([512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
