"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer_000
Pattern hash: c83a7895e053
Shape hash: 4859dd9a
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 384, 7, 7], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_76: "f32[128, 384, 7, 7]"):
        # No stacktrace found for following nodes
        neg_default: "f32[128, 384, 7, 7]" = torch.ops.aten.neg.default(convolution_76)
        exp_default: "f32[128, 384, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 384, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 384, 7, 7]" = torch.ops.aten.div.Tensor(convolution_76, add_tensor);  convolution_76 = add_tensor = None
        return div_tensor

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
