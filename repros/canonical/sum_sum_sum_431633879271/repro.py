"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train_001
Pattern hash: 431633879271
Shape hash: c3a784b1
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 320, 56, 56], f32), T([128, 320, 1, 1], f32), T([128, 1, 1, 1], f32), T([128, 320, 56, 56], f32), T([320], f32), T([], f32), S([320]), S([320]), S([128, 320, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, arg89_1: "f32[128, 320, 56, 56]", arg90_1: "f32[128, 320, 1, 1]", arg91_1: "f32[128, 1, 1, 1]", getitem_126: "f32[128, 320, 56, 56]", arg6_1: "f32[320]", full_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(arg89_1, 0.5)
        mul_tensor_1: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(arg89_1, 0.7071067811865476)
        erf_default: "f32[128, 320, 56, 56]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 320, 56, 56]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = None
        div_tensor: "f32[128, 320, 1, 1]" = torch.ops.aten.div.Tensor(arg90_1, arg91_1)
        mul_tensor_3: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_2, div_tensor)
        mul_scalar: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Scalar(mul_tensor_3, 1);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_126, mul_scalar);  mul_scalar = None
        view_default: "f32[1, 320, 1, 1]" = torch.ops.aten.view.default(arg6_1, [1, -1, 1, 1]);  arg6_1 = None
        mul_scalar_1: "f32[1, 320, 1, 1]" = torch.ops.aten.mul.Scalar(view_default, 1);  view_default = None
        mul_tensor_5: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_126, mul_scalar_1);  mul_scalar_1 = None
        sum_dim_int_list: "f32[1, 320, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_126, [0, 2, 3], True)
        sum_dim_int_list_1: "f32[1, 320, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3], True);  mul_tensor_4 = None
        mul_tensor_6: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_5, mul_tensor_2)
        mul_tensor_7: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_5, div_tensor);  mul_tensor_5 = None
        sum_dim_int_list_2: "f32[128, 320, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2, 3], True);  mul_tensor_6 = None
        add_tensor_1: "f32[128, 320, 56, 56]" = torch.ops.aten.add.Tensor(getitem_126, mul_tensor_7);  getitem_126 = mul_tensor_7 = None
        view_default_1: "f32[320]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_0);  sum_dim_int_list_1 = _shape_param_0 = None
        view_default_2: "f32[320]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        div_tensor_1: "f32[128, 320, 1, 1]" = torch.ops.aten.div.Tensor(div_tensor, arg91_1);  div_tensor = None
        neg_default: "f32[128, 320, 1, 1]" = torch.ops.aten.neg.default(sum_dim_int_list_2)
        mul_tensor_8: "f32[128, 320, 1, 1]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2: "f32[128, 320, 1, 1]" = torch.ops.aten.div.Tensor(sum_dim_int_list_2, arg91_1);  sum_dim_int_list_2 = arg91_1 = None
        sum_dim_int_list_3: "f32[128, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [1], True);  mul_tensor_8 = None
        expand_default: "f32[128, 320, 1, 1]" = torch.ops.aten.expand.default(sum_dim_int_list_3, _shape_param_2);  sum_dim_int_list_3 = _shape_param_2 = None
        div_scalar: "f32[128, 320, 1, 1]" = torch.ops.aten.div.Scalar(expand_default, 320);  expand_default = None
        add_tensor_2: "f32[128, 320, 1, 1]" = torch.ops.aten.add.Tensor(div_tensor_2, div_scalar);  div_tensor_2 = div_scalar = None
        div_tensor_3: "f32[128, 320, 56, 56]" = torch.ops.aten.div.Tensor(mul_tensor_2, arg90_1);  mul_tensor_2 = None
        eq_scalar: "b8[128, 320, 1, 1]" = torch.ops.aten.eq.Scalar(arg90_1, 0);  arg90_1 = None
        where_self: "f32[128, 320, 56, 56]" = torch.ops.aten.where.self(eq_scalar, full_1, div_tensor_3);  eq_scalar = full_1 = div_tensor_3 = None
        clone_default: "f32[128, 320, 56, 56]" = torch.ops.aten.clone.default(where_self, memory_format = torch.contiguous_format);  where_self = None
        mul_tensor_9: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(add_tensor_2, clone_default);  add_tensor_2 = clone_default = None
        add_tensor_3: "f32[128, 320, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_9);  add_tensor_1 = mul_tensor_9 = None
        mul_tensor_10: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_11: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(arg89_1, arg89_1)
        mul_tensor_12: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_11, -0.5);  mul_tensor_11 = None
        exp_default: "f32[128, 320, 56, 56]" = torch.ops.aten.exp.default(mul_tensor_12);  mul_tensor_12 = None
        mul_tensor_13: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_14: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(arg89_1, mul_tensor_13);  arg89_1 = mul_tensor_13 = None
        add_tensor_4: "f32[128, 320, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_14);  mul_tensor_10 = mul_tensor_14 = None
        mul_tensor_15: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(add_tensor_3, add_tensor_4);  add_tensor_3 = add_tensor_4 = None
        sum_dim_int_list_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 2, 3]);  mul_tensor_15 = None
        return (view_default_1, view_default_2, sum_dim_int_list_4)

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
