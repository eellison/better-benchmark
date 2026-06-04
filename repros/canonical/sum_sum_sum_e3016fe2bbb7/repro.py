"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForQuestionAnswering_train_001
Pattern hash: e3016fe2bbb7
Shape hash: ae31d711
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([], f32), T([1], b8), T([1], i64), T([1, 128], f32), T([1, 1], f32), T([1, 1], f32), T([1, 128], f32), T([1], b8), T([1], i64), T([1, 128], f32), T([1, 1], f32), T([1, 1], f32), T([1, 128], f32), S([1, 128]), S([1, 128]), S([1, 128]), S([128, 2]), S([2]))"

class Repro(torch.nn.Module):
    def forward(self, arg634_1: "f32[]", arg605_1: "b8[1]", arg200_1: "i64[1]", arg599_1: "f32[1, 128]", arg603_1: "f32[1, 1]", arg604_1: "f32[1, 1]", arg636_1: "f32[1, 128]", arg602_1: "b8[1]", arg199_1: "i64[1]", arg598_1: "f32[1, 128]", arg600_1: "f32[1, 1]", arg601_1: "f32[1, 1]", arg635_1: "f32[1, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(arg634_1, 2);  arg634_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(arg605_1);  arg605_1 = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        div_tensor_1: "f32[]" = torch.ops.aten.div.Tensor(div_tensor, convert_element_type_default);  convert_element_type_default = None
        clamp_min_default: "i64[1]" = torch.ops.aten.clamp_min.default(arg200_1, 0);  arg200_1 = None
        clamp_max_default: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 128);  clamp_min_default = None
        unsqueeze_default: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_default, 1);  clamp_max_default = None
        ne_scalar: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, 128)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[1, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default);  unsqueeze_default = None
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 128]" = torch.ops.aten.view.default(iota_default, _shape_param_0);  iota_default = _shape_param_0 = None
        expand_default: "i64[1, 128]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        eq_tensor: "b8[1, 128]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_self_1: "f32[1, 128]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "f32[1, 1]" = torch.ops.aten.where.self(ne_scalar, div_tensor_1, full_default_1);  ne_scalar = div_tensor_1 = None
        mul_tensor: "f32[1, 128]" = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None
        sub_tensor: "f32[1, 128]" = torch.ops.aten.sub.Tensor(arg599_1, arg603_1);  arg599_1 = arg603_1 = None
        sub_tensor_1: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, arg604_1);  sub_tensor = arg604_1 = None
        exp_default: "f32[1, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2: "f32[1, 128]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        add_tensor: "f32[1, 128]" = torch.ops.aten.add.Tensor(arg636_1, sub_tensor_2);  arg636_1 = sub_tensor_2 = None
        sum_default_1: "i64[]" = torch.ops.aten.sum.default(arg602_1);  arg602_1 = None
        convert_element_type_default_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_1, torch.float32);  sum_default_1 = None
        div_tensor_2: "f32[]" = torch.ops.aten.div.Tensor(div_tensor, convert_element_type_default_1);  div_tensor = convert_element_type_default_1 = None
        clamp_min_default_1: "i64[1]" = torch.ops.aten.clamp_min.default(arg199_1, 0);  arg199_1 = None
        clamp_max_default_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_default_1, 128);  clamp_min_default_1 = None
        unsqueeze_default_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_default_1, 1);  clamp_max_default_1 = None
        ne_scalar_1: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default_1, 128)
        where_self_3: "i64[1, 1]" = torch.ops.aten.where.self(ne_scalar_1, unsqueeze_default_1, full_default);  unsqueeze_default_1 = full_default = None
        expand_default_1: "i64[1, 128]" = torch.ops.aten.expand.default(where_self_3, _shape_param_2);  where_self_3 = _shape_param_2 = None
        eq_tensor_1: "b8[1, 128]" = torch.ops.aten.eq.Tensor(expand_default_1, view_default);  expand_default_1 = view_default = None
        where_self_4: "f32[1, 128]" = torch.ops.aten.where.self(eq_tensor_1, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor_1 = scalar_tensor_default_1 = scalar_tensor_default = None
        where_self_5: "f32[1, 1]" = torch.ops.aten.where.self(ne_scalar_1, div_tensor_2, full_default_1);  ne_scalar_1 = div_tensor_2 = full_default_1 = None
        mul_tensor_2: "f32[1, 128]" = torch.ops.aten.mul.Tensor(where_self_4, where_self_5);  where_self_4 = where_self_5 = None
        sub_tensor_3: "f32[1, 128]" = torch.ops.aten.sub.Tensor(arg598_1, arg600_1);  arg598_1 = arg600_1 = None
        sub_tensor_4: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_3, arg601_1);  sub_tensor_3 = arg601_1 = None
        exp_default_1: "f32[1, 128]" = torch.ops.aten.exp.default(sub_tensor_4);  sub_tensor_4 = None
        sum_dim_int_list_1: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [1], True)
        mul_tensor_3: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp_default_1, sum_dim_int_list_1);  exp_default_1 = sum_dim_int_list_1 = None
        sub_tensor_5: "f32[1, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        add_tensor_1: "f32[1, 128]" = torch.ops.aten.add.Tensor(arg635_1, sub_tensor_5);  arg635_1 = sub_tensor_5 = None
        unsqueeze_default_2: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(add_tensor, 2);  add_tensor = None
        unsqueeze_default_3: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 2);  add_tensor_1 = None
        cat_default: "f32[1, 128, 2]" = torch.ops.aten.cat.default([unsqueeze_default_3, unsqueeze_default_2], 2);  unsqueeze_default_3 = unsqueeze_default_2 = None
        view_default_1: "f32[128, 2]" = torch.ops.aten.view.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None
        permute_default: "f32[2, 128]" = torch.ops.aten.permute.default(view_default_1, [1, 0])
        sum_dim_int_list_2: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True);  view_default_1 = None
        view_default_2: "f32[2]" = torch.ops.aten.view.default(sum_dim_int_list_2, _shape_param_4);  sum_dim_int_list_2 = _shape_param_4 = None
        return (permute_default, view_default_2)

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
