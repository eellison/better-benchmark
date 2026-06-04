"""
Standalone repro captured via capture_hook.
Label: timm_timm_visformer_small_train_train_001
Pattern hash: 152a2d8c6ba7
Shape hash: 35f55d21
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([768, 196, 196], f32), T([128, 6, 196, 196], f32), S([128, 6, 196, 196]), S([768, 196, 196]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_29: "f32[768, 196, 196]", arg147_1: "f32[128, 6, 196, 196]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 6, 196, 196]" = torch.ops.aten.view.default(bmm_29, _shape_param_0);  bmm_29 = _shape_param_0 = None
        mul_tensor: "f32[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_default, arg147_1);  view_default = None
        sum_dim_int_list: "f32[128, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[128, 6, 196, 196]" = torch.ops.aten.neg.default(arg147_1);  arg147_1 = None
        fma_default: "f32[128, 6, 196, 196]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        mul_tensor_1: "f32[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(fma_default, 0.125);  fma_default = None
        view_default_1: "f32[768, 196, 196]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
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
