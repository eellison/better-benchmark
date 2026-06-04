"""
Standalone repro captured via capture_hook.
Label: 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_infer_000
Pattern hash: 0815d0a11243
Shape hash: 998e0068
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 128, 128], f32), S([16, 32, 128, 128]), S([16, -1, 128, 128]), S([16, 32, 128, 128]), S([512, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_4: "f32[512, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_0);  bmm_4 = _shape_param_0 = None
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[128]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        ge_scalar: "b8[1, 1, 128, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_default_2, 0);  unsqueeze_default_2 = None
        expand_default: "b8[16, 1, 128, 128]" = torch.ops.aten.expand.default(ge_scalar, _shape_param_1);  ge_scalar = _shape_param_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default, full_default_1);  expand_default = full_default = full_default_1 = None
        add_tensor_1: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_default, where_self);  view_default = where_self = None
        eq_scalar: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_1, -inf)
        logical_not_default: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_2: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_tensor_1, [-1], True)
        sub_tensor: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_1, amax_default);  add_tensor_1 = amax_default = None
        exp_default: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self_1: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor);  logical_not_default_1 = full_default_2 = div_tensor = None
        expand_default_1: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_1, _shape_param_2);  where_self_1 = _shape_param_2 = None
        view_default_1: "f32[512, 128, 128]" = torch.ops.aten.view.default(expand_default_1, _shape_param_3);  expand_default_1 = _shape_param_3 = None
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
