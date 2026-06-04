"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_base_train_001
Pattern hash: 3e8dba104ec0
Shape hash: 199c262d
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([768], f32), T([8, 1024, 768], b8), T([8, 1024, 768], f32), T([8, 1024, 1], f32), T([8, 1024, 768], f32), T([8, 1024], i64, gen=Index(32128)), T([], f32), T([32128, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([768], f32), T([8, 1024, 768], b8), T([8, 1024, 768], f32), T([8, 1024, 1], f32), T([8, 1024, 768], f32), T([8, 1024], i64, gen=Index(32128)), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([768]), S([8, 1024, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_237: "f32[8192, 768]", mm_239: "f32[8192, 768]", mm_241: "f32[8192, 768]", arg100_1: "f32[768]", arg425_1: "b8[8, 1024, 768]", arg424_1: "f32[8, 1024, 768]", arg426_1: "f32[8, 1024, 1]", add_127: "f32[8, 1024, 768]", arg423_1: "i64[8, 1024]", full_1: "f32[]", mm: "f32[32128, 768]", mm_381: "f32[8192, 768]", mm_383: "f32[8192, 768]", mm_385: "f32[8192, 768]", arg2_1: "f32[768]", arg259_1: "b8[8, 1024, 768]", arg257_1: "f32[8, 1024, 768]", arg260_1: "f32[8, 1024, 1]", add_212: "f32[8, 1024, 768]", arg0_1: "i64[8, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 768]" = torch.ops.aten.view.default(mm_237, _shape_param_0);  mm_237 = _shape_param_0 = None
        view_default_1: "f32[8, 1024, 768]" = torch.ops.aten.view.default(mm_239, _shape_param_1);  mm_239 = _shape_param_1 = None
        add_tensor: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[8, 1024, 768]" = torch.ops.aten.view.default(mm_241, _shape_param_2);  mm_241 = _shape_param_2 = None
        add_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg100_1);  arg100_1 = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg425_1, arg424_1);  arg424_1 = None
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.1111111111111112);  mul_tensor_1 = None
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg426_1)
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_3);  add_tensor_1 = mul_tensor_3 = None
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True);  mul_tensor_4 = None
        view_default_3: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None
        mul_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2)
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg426_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        add_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_127, mul_tensor_6);  add_127 = mul_tensor_6 = None
        pow_tensor_scalar: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg426_1, 3);  arg426_1 = None
        mul_scalar: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_7: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_tensor_7, _shape_param_4);  mul_tensor_7 = _shape_param_4 = None
        div_scalar: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        pow_tensor_scalar_1: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_2, 1.0);  mul_tensor_2 = None
        mul_scalar_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_8: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_8);  add_tensor_2 = mul_tensor_8 = None
        convert_element_type_default: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg425_1, torch.float32);  arg425_1 = None
        mul_tensor_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_3, mul_tensor_9);  add_tensor_3 = mul_tensor_9 = None
        eq_scalar: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(arg423_1, -1)
        unsqueeze_default: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_1, mul_tensor_10);  unsqueeze_default = mul_tensor_10 = None
        full_default: "f32[32128, 768]" = torch.ops.aten.full.default([32128, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32128, 768]" = torch.ops.aten.index_put.default(full_default, [arg423_1], where_self, True);  arg423_1 = where_self = None
        add_tensor_4: "f32[32128, 768]" = torch.ops.aten.add.Tensor(mm, index_put_default);  mm = index_put_default = None
        view_default_4: "f32[8, 1024, 768]" = torch.ops.aten.view.default(mm_381, _shape_param_5);  mm_381 = _shape_param_5 = None
        view_default_5: "f32[8, 1024, 768]" = torch.ops.aten.view.default(mm_383, _shape_param_6);  mm_383 = _shape_param_6 = None
        add_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_default_4, view_default_5);  view_default_4 = view_default_5 = None
        view_default_6: "f32[8, 1024, 768]" = torch.ops.aten.view.default(mm_385, _shape_param_7);  mm_385 = _shape_param_7 = None
        add_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_5, view_default_6);  add_tensor_5 = view_default_6 = None
        mul_tensor_11: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_6, arg2_1);  arg2_1 = None
        mul_tensor_12: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg259_1, arg257_1);  arg257_1 = None
        mul_tensor_13: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 1.1111111111111112);  mul_tensor_12 = None
        mul_tensor_14: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_13, arg260_1)
        mul_tensor_15: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_6, mul_tensor_14);  add_tensor_6 = mul_tensor_14 = None
        sum_dim_int_list_2: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        view_default_7: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_2, _shape_param_8);  sum_dim_int_list_2 = _shape_param_8 = None
        mul_tensor_16: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_11, mul_tensor_13)
        mul_tensor_17: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_11, arg260_1);  mul_tensor_11 = None
        sum_dim_int_list_3: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [2], True);  mul_tensor_16 = None
        add_tensor_7: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_212, mul_tensor_17);  add_212 = mul_tensor_17 = None
        pow_tensor_scalar_2: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg260_1, 3);  arg260_1 = None
        mul_scalar_2: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_3, -0.5);  sum_dim_int_list_3 = None
        mul_tensor_18: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar_2, pow_tensor_scalar_2);  mul_scalar_2 = pow_tensor_scalar_2 = None
        expand_default_1: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_tensor_18, _shape_param_9);  mul_tensor_18 = _shape_param_9 = None
        div_scalar_1: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_default_1, 768);  expand_default_1 = None
        pow_tensor_scalar_3: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_13, 1.0);  mul_tensor_13 = None
        mul_scalar_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_3, 2.0);  pow_tensor_scalar_3 = None
        mul_tensor_19: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_3);  div_scalar_1 = mul_scalar_3 = None
        add_tensor_8: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_7, mul_tensor_19);  add_tensor_7 = mul_tensor_19 = None
        convert_element_type_default_1: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg259_1, torch.float32);  arg259_1 = None
        mul_tensor_20: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_21: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_8, mul_tensor_20);  add_tensor_8 = mul_tensor_20 = None
        eq_scalar_1: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(arg0_1, -1)
        unsqueeze_default_1: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_1, mul_tensor_21);  unsqueeze_default_1 = full_1 = mul_tensor_21 = None
        index_put_default_1: "f32[32128, 768]" = torch.ops.aten.index_put.default(full_default, [arg0_1], where_self_1, True);  full_default = arg0_1 = where_self_1 = None
        add_tensor_9: "f32[32128, 768]" = torch.ops.aten.add.Tensor(add_tensor_4, index_put_default_1);  add_tensor_4 = index_put_default_1 = None
        return (view_default_3, view_default_7, add_tensor_9)

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
