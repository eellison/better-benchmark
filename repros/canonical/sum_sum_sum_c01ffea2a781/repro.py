"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train_003
Pattern hash: c01ffea2a781
Shape hash: b6cfacd8
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([50260, 768], f32), T([16384, 768], f32), T([768], f32), T([32, 512, 768], f32), T([1, 512, 768], f32), T([32, 512, 768], b8), T([32, 512, 1], f32), T([32, 512, 1], f32), T([32, 512, 768], f32), T([1, 512], i64, gen=Index(1024)), T([32, 512], i64, gen=Index(50257)), S([32, 512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[50260, 768]", mm_48: "f32[16384, 768]", arg2_1: "f32[768]", arg39_1: "f32[32, 512, 768]", arg41_1: "f32[1, 512, 768]", arg42_1: "b8[32, 512, 768]", arg43_1: "f32[32, 512, 1]", arg44_1: "f32[32, 512, 1]", add_35: "f32[32, 512, 768]", arg40_1: "i64[1, 512]", arg0_1: "i64[32, 512]", _shape_param_0):
        # No stacktrace found for following nodes
        slice_tensor: "f32[50257, 768]" = torch.ops.aten.slice.Tensor(mm, 0, 0, -3);  mm = None
        view_default: "f32[32, 512, 768]" = torch.ops.aten.view.default(mm_48, _shape_param_0);  mm_48 = _shape_param_0 = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_default, arg2_1);  arg2_1 = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg39_1, arg41_1);  arg39_1 = arg41_1 = None
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg42_1, add_tensor);  add_tensor = None
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.0);  mul_tensor_2 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_3, arg43_1);  mul_tensor_3 = arg43_1 = None
        mul_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, arg44_1);  sub_tensor = None
        mul_tensor_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_6);  sub_tensor_1 = mul_tensor_6 = None
        div_tensor: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(arg44_1, 768);  arg44_1 = None
        mul_tensor_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor_4);  mul_tensor_4 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_35, mul_tensor_7);  add_35 = mul_tensor_7 = None
        convert_element_type_default: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.float32);  arg42_1 = None
        mul_tensor_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_tensor_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_9);  add_tensor_1 = mul_tensor_9 = None
        sum_dim_int_list_4: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0], True)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg40_1, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default_1: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_1, [arg40_1], where_self, True);  full_default_1 = arg40_1 = where_self = None
        eq_scalar_1: "b8[32, 512]" = torch.ops.aten.eq.Scalar(arg0_1, -1)
        unsqueeze_default_1: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, mul_tensor_10);  unsqueeze_default_1 = full_default = mul_tensor_10 = None
        full_default_2: "f32[50257, 768]" = torch.ops.aten.full.default([50257, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[50257, 768]" = torch.ops.aten.index_put.default(full_default_2, [arg0_1], where_self_1, True);  full_default_2 = arg0_1 = where_self_1 = None
        add_tensor_2: "f32[50257, 768]" = torch.ops.aten.add.Tensor(slice_tensor, index_put_default_1);  slice_tensor = index_put_default_1 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, add_tensor_2)

def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)

def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
