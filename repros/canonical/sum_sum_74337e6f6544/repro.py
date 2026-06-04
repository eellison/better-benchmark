"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_004
Pattern hash: 74337e6f6544
Shape hash: fdcd8095
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([], f32), T([], f32), T([8, 1024], i64), T([8, 1024, 50265], f32), T([8192, 1], f32), T([8192, 1], f32), T([8, 1024, 50265], f32), S([1, 50265]), S([8192, 50265]), S([-1, 50265]), S([8, 1024, 50265]), S([8192, 50265]), S([50265]))"

class Repro(torch.nn.Module):
    def forward(self, arg13_1: "f32[]", arg12_1: "f32[]", arg3_1: "i64[8, 1024]", arg9_1: "f32[8, 1024, 50265]", arg10_1: "f32[8192, 1]", arg11_1: "f32[8192, 1]", arg14_1: "f32[8, 1024, 50265]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(arg13_1, arg12_1);  arg13_1 = arg12_1 = None
        view_default: "i64[8192]" = torch.ops.aten.view.default(arg3_1, [-1]);  arg3_1 = None
        unsqueeze_default: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(view_default, 1);  view_default = None
        ne_scalar: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[8192, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default);  unsqueeze_default = full_default = None
        iota_default: "i64[50265]" = torch.ops.prims.iota.default(50265, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default_1: "i64[1, 50265]" = torch.ops.aten.view.default(iota_default, _shape_param_0);  iota_default = _shape_param_0 = None
        expand_default: "i64[8192, 50265]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        eq_tensor: "b8[8192, 50265]" = torch.ops.aten.eq.Tensor(expand_default, view_default_1);  expand_default = view_default_1 = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_self_1: "f32[8192, 50265]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "f32[8192, 1]" = torch.ops.aten.where.self(ne_scalar, div_tensor, full_default_1);  ne_scalar = div_tensor = full_default_1 = None
        mul_tensor: "f32[8192, 50265]" = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None
        view_default_2: "f32[8192, 50265]" = torch.ops.aten.view.default(arg9_1, _shape_param_2);  arg9_1 = _shape_param_2 = None
        sub_tensor: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(view_default_2, arg10_1);  view_default_2 = arg10_1 = None
        sub_tensor_1: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(sub_tensor, arg11_1);  sub_tensor = arg11_1 = None
        exp_default: "f32[8192, 50265]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[8192, 50265]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        view_default_3: "f32[8, 1024, 50265]" = torch.ops.aten.view.default(sub_tensor_2, _shape_param_3);  sub_tensor_2 = _shape_param_3 = None
        add_tensor: "f32[8, 1024, 50265]" = torch.ops.aten.add.Tensor(arg14_1, view_default_3);  arg14_1 = view_default_3 = None
        view_default_4: "f32[8192, 50265]" = torch.ops.aten.view.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None
        constant_pad_nd_default: "f32[8192, 50268]" = torch.ops.aten.constant_pad_nd.default(view_default_4, [0, 3, 0, 0])
        permute_default: "f32[50265, 8192]" = torch.ops.aten.permute.default(view_default_4, [1, 0])
        constant_pad_nd_default_1: "f32[50268, 8192]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 3]);  permute_default = None
        sum_dim_int_list_1: "f32[1, 50265]" = torch.ops.aten.sum.dim_IntList(view_default_4, [0], True);  view_default_4 = None
        view_default_5: "f32[50265]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_5);  sum_dim_int_list_1 = _shape_param_5 = None
        return (constant_pad_nd_default, constant_pad_nd_default_1, view_default_5)

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
