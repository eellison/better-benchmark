"""
Standalone repro captured via capture_hook.
Label: torchbench_dlrm_train_000
Pattern hash: f9aca3f32296
Shape hash: 3997c37d
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f32[2048, 1]"):
        # No stacktrace found for following nodes
        relu_default: "f32[2048, 1]" = torch.ops.aten.relu.default(addmm_5);  addmm_5 = None
        le_scalar: "b8[2048, 1]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return le_scalar

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
