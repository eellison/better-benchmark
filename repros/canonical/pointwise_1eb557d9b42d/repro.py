"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: 1eb557d9b42d
Shape hash: 643b7074
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([30522, 512], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg1125_1: "f32[30522, 512]"):
        # No stacktrace found for following nodes
        constant_pad_nd_default: "f32[30524, 512]" = torch.ops.aten.constant_pad_nd.default(arg1125_1, [0, 0, 0, 2]);  arg1125_1 = None
        return constant_pad_nd_default

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
