"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_infer_000
Pattern hash: 8e1dc74da682
Shape hash: d27d17e3
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 640, 7, 7], f32), T([128, 640, 7, 7], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_42: "f32[128, 640, 7, 7]", convolution_39: "f32[128, 640, 7, 7]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[128, 640, 7, 7]" = torch.ops.aten.add.Tensor(convolution_42, convolution_39);  convolution_42 = convolution_39 = None
        return add_tensor

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
