"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train_000
Pattern hash: 0561785713ab
Shape hash: 0a05ef0e
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
_shapes_config = "(T([1536, 128, 128], f32), T([128, 1, 128, 128], b8), T([], f32), T([61], i64), S([128, 12, 128, 128]), S([128, 12, 128, 128]), S([1536, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_22: "f32[1536, 128, 128]", eq: "b8[128, 1, 128, 128]", full: "f32[]", inductor_seeds: "i64[61]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[128, 12, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_0);  bmm_22 = _shape_param_0 = None
        div_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_default, 8.0);  view_default = None
        where_self: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full, div_tensor);  eq = full = div_tensor = None
        amax_default: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_self, [-1], True)
        sub_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_self, amax_default);  where_self = amax_default = None
        exp_default: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 56);  inductor_seeds = None
        inductor_random_default: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor_1);  gt_scalar = div_tensor_1 = None
        mul_tensor_1: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        expand_default: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        view_default_1: "f32[1536, 128, 128]" = torch.ops.aten.view.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        permute_default: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1]);  view_default_1 = None
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
