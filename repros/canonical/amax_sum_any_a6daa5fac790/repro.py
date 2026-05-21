"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_train_008
Pattern hash: a6daa5fac790
Shape hash: c2fc68ae
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
_shapes_config = "(T([4, 1, 512, 512], b8), T([], f32), T([], f32), T([48, 512, 512], f32), T([5], i64), S([4, 12, 512, 512]), S([4, 12, 512, 512]), S([48, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, arg19_1: "b8[4, 1, 512, 512]", full_1: "f32[]", full: "f32[]", bmm: "f32[48, 512, 512]", inductor_seeds: "i64[5]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        where_self: "f32[4, 1, 512, 512]" = torch.ops.aten.where.self(arg19_1, full_1, full);  arg19_1 = full_1 = full = None
        view_default: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        add_tensor: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_default, where_self);  view_default = where_self = None
        amax_default: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
        exp_default: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        eq_scalar: "b8[4, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
        logical_not_default: "b8[4, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[4, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[4, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default: "f32[4, 12, 512, 512]" = torch.ops.aten.full.default([4, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[4, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_default_1, full_default, div_tensor);  logical_not_default_1 = full_default = div_tensor = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1);  inductor_seeds = None
        inductor_random_default: "f32[4, 12, 512, 512]" = torch.ops.prims.inductor_random.default([4, 12, 512, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[4, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, where_self_1);  gt_scalar = where_self_1 = None
        mul_tensor_1: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        expand_default: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        view_default_1: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        permute_default: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1]);  view_default_1 = None
        return permute_default



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
