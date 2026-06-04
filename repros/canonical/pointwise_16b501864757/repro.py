"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer_000
Pattern hash: 16b501864757
Shape hash: f791b756
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 128, 56, 56], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_3: "f32[128, 128, 56, 56]"):
        # No stacktrace found for following nodes
        neg_default: "f32[128, 128, 56, 56]" = torch.ops.aten.neg.default(convolution_3)
        exp_default: "f32[128, 128, 56, 56]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 128, 56, 56]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 128, 56, 56]" = torch.ops.aten.div.Tensor(convolution_3, add_tensor);  convolution_3 = add_tensor = None
        mul_tensor: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(div_tensor, 1.0);  div_tensor = None
        return mul_tensor

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
