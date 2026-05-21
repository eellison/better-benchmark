"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bert_large_infer_000
Pattern hash: a476543fe91a
Shape hash: 0261f4eb
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
_shapes_config = "(T([16, 512, 512], f32), T([1, 1, 512, 512], b8, stride=(512, 512, 1, 0)), S([1, 16, 512, 512]), S([1, 16, 512, 512]), S([16, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_46: "f32[16, 512, 512]", expand_4: "b8[1, 1, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_46, _shape_param_0);  bmm_46 = _shape_param_0 = None
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_4, full_default, full_default_1);  expand_4 = full_default = full_default_1 = None
        add_tensor: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_default, where_self);  view_default = where_self = None
        eq_scalar: "b8[1, 16, 512, 512]" = torch.ops.aten.eq.Scalar(add_tensor, -inf)
        logical_not_default: "b8[1, 16, 512, 512]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[1, 16, 512, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[1, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_2: "f32[1, 16, 512, 512]" = torch.ops.aten.full.default([1, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self_1: "f32[1, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor);  logical_not_default_1 = full_default_2 = div_tensor = None
        expand_default: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(where_self_1, _shape_param_1);  where_self_1 = _shape_param_1 = None
        view_default_1: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
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
