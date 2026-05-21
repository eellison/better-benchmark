"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train_003
Pattern hash: 230cc68d5501
Shape hash: 6b5d9c1d
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
_shapes_config = "(T([6144, 64, 128], f32), S([8, 12, 64, 64, 128]), S([8, 12, 64, 64, 128]), S([6144, 64, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm: "f32[6144, 64, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 12, 64, 64, 128]" = torch.ops.prims.inductor_random.default([8, 12, 64, 64, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 12, 64, 64, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.05);  inductor_random_default = None
        view_default: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        amax_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.amax.default(view_default, [-1], True)
        abs_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax_default)
        eq_scalar: "b8[8, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_default, inf);  abs_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq_scalar, full_default, amax_default);  eq_scalar = full_default = amax_default = None
        sub_tensor: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(view_default, where_self)
        exp_default: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True);  exp_default = None
        log_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        add_tensor: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log_default, where_self);  log_default = where_self = None
        sub_tensor_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(view_default, add_tensor);  view_default = add_tensor = None
        exp_default_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        mul_tensor: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.mul.Tensor(gt_scalar, exp_default_1);  gt_scalar = exp_default_1 = None
        mul_tensor_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.0526315789473684);  mul_tensor = None
        expand_default: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.expand.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        view_default_1: "f32[6144, 64, 128]" = torch.ops.aten.view.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
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
