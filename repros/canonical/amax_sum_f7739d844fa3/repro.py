"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train_000
Pattern hash: f7739d844fa3
Shape hash: 100a5bbc
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
_shapes_config = "(T([192, 512, 512], f32), T([73], i64), S([-1, 24, 512, 512]), S([-1, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm: "f32[192, 512, 512]", inductor_seeds: "i64[73]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        full_default: "b8[8, 1, 512, 512]" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default, full_default_1, view_default);  full_default = full_default_1 = view_default = None
        amax_default: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_self, [-1], True)
        sub_tensor: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_self, amax_default);  where_self = amax_default = None
        exp_default: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1);  inductor_seeds = None
        inductor_random_default: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor);  gt_scalar = div_tensor = None
        mul_tensor_1: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        view_default_1: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
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
