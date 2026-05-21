"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_004
Pattern hash: 2a058cdc340f
Shape hash: b95e40c0
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 50265], f32))"

class Repro(torch.nn.Module):
    def forward(self, view_4: "f32[8192, 50265]"):
        # No stacktrace found for following nodes
        permute_default: "f32[50265, 8192]" = torch.ops.aten.permute.default(view_4, [1, 0]);  view_4 = None
        constant_pad_nd_default: "f32[50268, 8192]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 3]);  permute_default = None
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
