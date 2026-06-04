"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_infer_000
Pattern hash: e562199fff3c
Shape hash: da25c4f0
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([192, 512, 512], f32), S([-1, 24, 512, 512]), S([-1, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_46: "f32[192, 512, 512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        full_default: "b8[8, 1, 512, 512]" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_46, _shape_param_0);  bmm_46 = _shape_param_0 = None
        where_self: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default, full_default_1, view_default);  full_default = full_default_1 = view_default = None
        amax_default: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_self, [-1], True)
        sub_tensor: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_self, amax_default);  where_self = amax_default = None
        exp_default: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        view_default_1: "f32[192, 512, 512]" = torch.ops.aten.view.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
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
