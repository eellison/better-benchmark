"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_infer
Pattern hash: f30674e2e5ee
Shape hash: b6410e3b
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 3, 224, 224], f32, stride=(150528, 1, 672, 3)))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[128, 3, 224, 224]"):
        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[128, 3, 225, 225]" = torch.ops.aten.constant_pad_nd.default(arg1_1, [0, 1, 0, 1], 0.0);  arg1_1 = None
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
