"""
Standalone repro captured via capture_hook.
Label: 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_train_007
Pattern hash: 027aee0fe13f
Shape hash: 17feed2c
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([16, 1, 128, 128], b8), T([], f32), T([], f32), T([512, 128, 128], f32), S([16, 32, 128, 128]), S([16, 32, 128, 128]), S([512, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, arg21_1: "b8[16, 1, 128, 128]", full_1: "f32[]", full: "f32[]", bmm: "f32[512, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        where_self: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(arg21_1, full_1, full);  arg21_1 = full_1 = full = None
        view_default: "f32[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        add_tensor: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_default, where_self);  view_default = where_self = None
        amax_default: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
        exp_default: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        eq_scalar: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
        logical_not_default: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_default_1, full_default, div_tensor);  logical_not_default_1 = full_default = div_tensor = None
        expand_default: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_1, _shape_param_1);  where_self_1 = _shape_param_1 = None
        view_default_1: "f32[512, 128, 128]" = torch.ops.aten.view.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        return view_default_1

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
