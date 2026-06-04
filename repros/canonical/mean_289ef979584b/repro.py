"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer_000
Pattern hash: 289ef979584b
Shape hash: a109de2f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 2304, 7, 7], f32), S([128, 2304]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_80: "f32[128, 2304, 7, 7]", _shape_param_0):
        # No stacktrace found for following nodes
        neg_default: "f32[128, 2304, 7, 7]" = torch.ops.aten.neg.default(convolution_80)
        exp_default: "f32[128, 2304, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 2304, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 2304, 7, 7]" = torch.ops.aten.div.Tensor(convolution_80, add_tensor);  convolution_80 = add_tensor = None
        mean_dim: "f32[128, 2304, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [-1, -2], True);  div_tensor = None
        as_strided_default: "f32[128, 2304, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 2304, 1, 1], [2304, 1, 2304, 2304]);  mean_dim = None
        view_default: "f32[128, 2304]" = torch.ops.aten.view.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None
        return view_default

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
