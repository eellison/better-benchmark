"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_000
Pattern hash: 8b19996123a8
Shape hash: 14f8f924
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 512, 512], f32), T([8, 1, 512, 512], f32), T([8, 64, 512, 512], f32), S([8, 64, 512, 512]), S([8, 64, 512, 512]), S([512, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_22: "f32[512, 512, 512]", where: "f32[8, 1, 512, 512]", full_2: "f32[8, 64, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_0);  bmm_22 = _shape_param_0 = None
        add_tensor: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(view_default, where);  view_default = where = None
        amax_default: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
        exp_default: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        eq_scalar: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
        logical_not_default: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        where_self: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_default_1, full_2, div_tensor);  logical_not_default_1 = full_2 = div_tensor = None
        expand_default: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        view_default_1: "f32[512, 512, 512]" = torch.ops.aten.view.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
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
