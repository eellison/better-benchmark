"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_infer_000
Pattern hash: 1c71ddeae558
Shape hash: 9ed9f1a2
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 128, 48, 48], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_3: "f32[128, 128, 48, 48]"):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, 0.5)
        mul_tensor_1: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476);  convolution_3 = None
        erf_default: "f32[128, 128, 48, 48]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        mul_tensor_3: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.7015043497085571);  mul_tensor_2 = None
        mul_tensor_4: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.0);  mul_tensor_3 = None
        return mul_tensor_4

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
