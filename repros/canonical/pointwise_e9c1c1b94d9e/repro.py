"""
Standalone repro captured via capture_hook.
Label: timm_timm_visformer_small_infer_infer_000
Pattern hash: e9c1c1b94d9e
Shape hash: 5657547b
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 3072, 7, 7], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_55: "f32[128, 3072, 7, 7]"):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_55, 0.5)
        mul_tensor_1: "f32[128, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_55, 0.7071067811865476);  convolution_55 = None
        erf_default: "f32[128, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        return mul_tensor_2

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
