"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train_000
Pattern hash: 3b7ac433d31c
Shape hash: 6096be54
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 1000, 13, 13], f32), S([512, 1000]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_25: "f32[512, 1000, 13, 13]", _shape_param_0):
        # No stacktrace found for following nodes
        relu_default: "f32[512, 1000, 13, 13]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
        mean_dim: "f32[512, 1000, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True)
        view_default: "f32[512, 1000]" = torch.ops.aten.view.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        le_scalar: "b8[512, 1000, 13, 13]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (view_default, le_scalar)

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
