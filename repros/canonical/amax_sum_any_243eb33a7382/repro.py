"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train_000
Pattern hash: 243eb33a7382
Shape hash: 4542832a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([3072, 128, 128], f32), T([256, 1, 128, 128], f32), T([256, 12, 128, 128], f32), T([13], i64), S([256, 12, 128, 128]), S([256, 12, 128, 128]), S([3072, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_10: "f32[3072, 128, 128]", where: "f32[256, 1, 128, 128]", full_2: "f32[256, 12, 128, 128]", inductor_seeds: "i64[13]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_0);  bmm_10 = _shape_param_0 = None
        add_tensor: "f32[256, 12, 128, 128]" = torch.ops.aten.add.Tensor(view_default, where);  view_default = where = None
        amax_default: "f32[256, 12, 128, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[256, 12, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
        exp_default: "f32[256, 12, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[256, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        eq_scalar: "b8[256, 12, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
        logical_not_default: "b8[256, 12, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[256, 12, 128, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[256, 12, 128, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        where_self: "f32[256, 12, 128, 128]" = torch.ops.aten.where.self(logical_not_default_1, full_2, div_tensor);  logical_not_default_1 = full_2 = div_tensor = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 11);  inductor_seeds = None
        inductor_random_default: "f32[256, 12, 128, 128]" = torch.ops.prims.inductor_random.default([256, 12, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[256, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_scalar, where_self);  gt_scalar = where_self = None
        mul_tensor_1: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        expand_default: "f32[256, 12, 128, 128]" = torch.ops.aten.expand.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        view_default_1: "f32[3072, 128, 128]" = torch.ops.aten.view.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
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
