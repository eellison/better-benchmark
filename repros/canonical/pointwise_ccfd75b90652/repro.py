"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_004
Pattern hash: ccfd75b90652
Shape hash: 09f23c3a
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([50265, 768], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[50265, 768]"):
        # No stacktrace found for following nodes
        permute_default: "f32[768, 50265]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        permute_default_1: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        constant_pad_nd_default: "f32[50268, 768]" = torch.ops.aten.constant_pad_nd.default(permute_default_1, [0, 0, 0, 3]);  permute_default_1 = None
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
