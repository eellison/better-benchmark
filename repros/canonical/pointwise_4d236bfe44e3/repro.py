"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_infer_000
Pattern hash: 4d236bfe44e3
Shape hash: 9e714a29
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
_shapes_config = "(T([768], f32), T([768], f32), T([768], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg205_1: "f32[768]", arg206_1: "f32[768]", arg207_1: "f32[768]"):
        # No stacktrace found for following nodes
        cat_default: "f32[2304]" = torch.ops.aten.cat.default([arg205_1, arg206_1, arg207_1]);  arg205_1 = arg206_1 = arg207_1 = None
        return cat_default



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
