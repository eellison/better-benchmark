"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g35
Pattern hash: eceb9dacdab4
Shape hash: efd56dfc
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_45: "f16[32, 1024, 1024]", arg132_1: "b8[4, 8, 1024, 1024]", arg127_1: "f16[32, 1024, 1024]", arg129_1: "f32[1024, 1024, 8]", arg123_1: "f32[4, 1, 1024, 1024]", arg130_1: "f32[4, 8, 1024, 1]", arg131_1: "f32[4, 8, 1024, 1]", bmm_69: "f16[32, 1024, 1024]", arg45_1: "b8[4, 8, 1024, 1024]", arg36_1: "b8[1, 1, 1024, 1]", full_1: "f32[]", arg40_1: "f16[32, 1024, 1024]", arg42_1: "f32[1024, 1024, 8]", arg43_1: "f32[4, 8, 1024, 1]", arg44_1: "f32[4, 8, 1024, 1]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_45, [4, 8, 1024, 1024]);  bmm_45 = None
        convert_element_type_default: "f16[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.float16);  arg132_1 = None
        mul_tensor: "f16[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f16[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None
        convert_element_type_default_1: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float32);  mul_tensor_1 = None
        reshape_default_1: "f16[4, 8, 1024, 1024]" = torch.ops.aten.reshape.default(arg127_1, [4, 8, 1024, 1024]);  arg127_1 = None
        permute_default: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(arg129_1, [2, 0, 1]);  arg129_1 = None
        unsqueeze_default: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_default, 0);  permute_default = None
        add_tensor: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default, arg123_1);  unsqueeze_default = arg123_1 = None
        add_tensor_1: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(reshape_default_1, add_tensor);  reshape_default_1 = add_tensor = None
        convert_element_type_default_2: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        sub_tensor: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, arg130_1);  convert_element_type_default_2 = arg130_1 = None
        exp_default: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default, arg131_1);  exp_default = arg131_1 = None
        mul_tensor_2: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, div_tensor);  convert_element_type_default_1 = None
        sum_dim_int_list: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[4, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default: "f32[4, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        convert_element_type_default_3: "f16[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_default, torch.float16);  fma_default = None
        reshape_default_2: "f16[32, 1024, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default_3, [32, 1024, 1024]);  convert_element_type_default_3 = None
        reshape_default_3: "f16[4, 8, 1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_2, [4, 8, 1024, 1024]);  reshape_default_2 = None
        reshape_default_4: "f16[32, 1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_3, [32, 1024, 1024]);  reshape_default_3 = None
        reshape_default_5: "f16[4, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_69, [4, 8, 1024, 1024]);  bmm_69 = None
        convert_element_type_default_4: "f16[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg45_1, torch.float16);  arg45_1 = None
        mul_tensor_3: "f16[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.1111111111111112);  convert_element_type_default_4 = None
        mul_tensor_4: "f16[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(reshape_default_5, mul_tensor_3);  reshape_default_5 = mul_tensor_3 = None
        convert_element_type_default_5: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float32);  mul_tensor_4 = None
        expand_default: "b8[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(arg36_1, [4, -1, 1024, 1024]);  arg36_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[4, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_1, full_default);  expand_default = full_1 = full_default = None
        reshape_default_6: "f16[4, 8, 1024, 1024]" = torch.ops.aten.reshape.default(arg40_1, [4, 8, 1024, 1024]);  arg40_1 = None
        permute_default_1: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(arg42_1, [2, 0, 1]);  arg42_1 = None
        unsqueeze_default_1: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_default_1, 0);  permute_default_1 = None
        add_tensor_2: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default_1, where_self);  unsqueeze_default_1 = where_self = None
        add_tensor_3: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(reshape_default_6, add_tensor_2);  reshape_default_6 = add_tensor_2 = None
        convert_element_type_default_6: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32);  add_tensor_3 = None
        sub_tensor_1: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default_6, arg43_1);  convert_element_type_default_6 = arg43_1 = None
        exp_default_1: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        div_tensor_1: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default_1, arg44_1);  exp_default_1 = arg44_1 = None
        mul_tensor_5: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, div_tensor_1);  convert_element_type_default_5 = None
        sum_dim_int_list_1: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [-1], True)
        neg_default_1: "f32[4, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_tensor_1);  div_tensor_1 = None
        fma_default_1: "f32[4, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_default_1, sum_dim_int_list_1, mul_tensor_5);  neg_default_1 = sum_dim_int_list_1 = mul_tensor_5 = None
        convert_element_type_default_7: "f16[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_default_1, torch.float16);  fma_default_1 = None
        reshape_default_7: "f16[32, 1024, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default_7, [32, 1024, 1024]);  convert_element_type_default_7 = None
        reshape_default_8: "f16[4, 8, 1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_7, [4, 8, 1024, 1024]);  reshape_default_7 = None
        reshape_default_9: "f16[32, 1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_8, [32, 1024, 1024]);  reshape_default_8 = None
        return (reshape_default_4, reshape_default_9)


def _default_make_inputs():
    return [
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [4, 8, 1024, 1024], dtype=torch.bool, device='cuda'),
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([1024, 1024, 8], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [4, 8, 1024, 1024], dtype=torch.bool, device='cuda'),
    torch.randint(0, 2, [1, 1, 1024, 1], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([1024, 1024, 8], dtype=torch.float32, device='cuda'),
    torch.randn([4, 8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 8, 1024, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
