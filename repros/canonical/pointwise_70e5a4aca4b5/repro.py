"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train_000
Pattern hash: 70e5a4aca4b5
Shape hash: 794de873
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 256, 13, 13], f32), T([512, 256, 13, 13], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_20: "f32[512, 256, 13, 13]", convolution_21: "f32[512, 256, 13, 13]"):
        # No stacktrace found for following nodes
        relu_default: "f32[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
        relu_default_1: "f32[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
        cat_default: "f32[512, 512, 13, 13]" = torch.ops.aten.cat.default([relu_default, relu_default_1], 1)
        le_scalar: "b8[512, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        le_scalar_1: "b8[512, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (cat_default, le_scalar, le_scalar_1)

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
