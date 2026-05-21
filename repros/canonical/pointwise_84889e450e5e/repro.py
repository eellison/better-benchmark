"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer_003
Pattern hash: 84889e450e5e
Shape hash: 09f23c3a
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
_shapes_config = "(T([50265, 768], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg5_1: "f32[50265, 768]"):
        # No stacktrace found for following nodes
        permute_default: "f32[768, 50265]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        constant_pad_nd_default: "f32[768, 50268]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 3, 0, 0]);  permute_default = None
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
