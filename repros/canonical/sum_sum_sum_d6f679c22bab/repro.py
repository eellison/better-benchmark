"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g53
Pattern hash: d6f679c22bab
Shape hash: 2ce1f9ce
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg532_1: "f32[32, 512]", arg359_1: "f32[32, 1]", arg357_1: "f16[32, 512]", full: "f32[]", mm_99: "f16[32, 768]", arg27_1: "f32[768]", arg458_1: "f32[32, 768]", arg459_1: "f32[32, 1]"):
        # No stacktrace found for following nodes
        neg_default: "f32[32, 512]" = torch.ops.aten.neg.default(arg532_1)
        clamp_min_default: "f32[32, 1]" = torch.ops.aten.clamp_min.default(arg359_1, 1e-12)
        expand_default: "f32[32, 512]" = torch.ops.aten.expand.default(clamp_min_default, [32, 512]);  clamp_min_default = None
        div_tensor: "f32[32, 512]" = torch.ops.aten.div.Tensor(arg357_1, expand_default)
        div_tensor_1: "f32[32, 512]" = torch.ops.aten.div.Tensor(div_tensor, expand_default);  div_tensor = None
        mul_tensor: "f32[32, 512]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2: "f32[32, 512]" = torch.ops.aten.div.Tensor(arg532_1, expand_default);  arg532_1 = expand_default = None
        convert_element_type_default: "f16[32, 512]" = torch.ops.prims.convert_element_type.default(div_tensor_2, torch.float16);  div_tensor_2 = None
        sum_dim_int_list: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True);  mul_tensor = None
        ge_scalar: "b8[32, 1]" = torch.ops.aten.ge.Scalar(arg359_1, 1e-12)
        where_self: "f32[32, 1]" = torch.ops.aten.where.self(ge_scalar, sum_dim_int_list, full);  ge_scalar = sum_dim_int_list = None
        div_tensor_3: "f32[32, 512]" = torch.ops.aten.div.Tensor(arg357_1, arg359_1);  arg357_1 = None
        eq_scalar: "b8[32, 1]" = torch.ops.aten.eq.Scalar(arg359_1, 0);  arg359_1 = None
        where_self_1: "f32[32, 512]" = torch.ops.aten.where.self(eq_scalar, full, div_tensor_3);  eq_scalar = full = div_tensor_3 = None
        mul_tensor_1: "f32[32, 512]" = torch.ops.aten.mul.Tensor(where_self, where_self_1);  where_self = where_self_1 = None
        convert_element_type_default_1: "f16[32, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None
        add_tensor: "f16[32, 512]" = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_1);  convert_element_type_default = convert_element_type_default_1 = None
        convert_element_type_default_2: "f32[32, 768]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        mul_tensor_2: "f32[32, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, arg27_1);  convert_element_type_default_2 = arg27_1 = None
        mul_tensor_3: "f32[32, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 768)
        sum_dim_int_list_1: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [1], True)
        mul_tensor_4: "f32[32, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg458_1);  mul_tensor_2 = None
        sum_dim_int_list_2: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [1], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[32, 768]" = torch.ops.aten.mul.Tensor(arg458_1, sum_dim_int_list_2);  arg458_1 = sum_dim_int_list_2 = None
        sub_tensor: "f32[32, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list_1);  mul_tensor_3 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[32, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_5);  sub_tensor = mul_tensor_5 = None
        mul_tensor_6: "f32[32, 768]" = torch.ops.aten.mul.Tensor(arg459_1, sub_tensor_1);  arg459_1 = sub_tensor_1 = None
        full_default: "f32[32, 50, 768]" = torch.ops.aten.full.default([32, 50, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[32, 50, 768]" = torch.ops.aten.select_scatter.default(full_default, mul_tensor_6, 1, 0);  full_default = mul_tensor_6 = None
        convert_element_type_default_3: "f16[32, 50, 768]" = torch.ops.prims.convert_element_type.default(select_scatter_default, torch.float16);  select_scatter_default = None
        reshape_default: "f16[1600, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_3, [1600, 768]);  convert_element_type_default_3 = None
        return (add_tensor, reshape_default)


def _default_make_inputs():
    return [
    torch.randn([32, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512], dtype=torch.float16, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
