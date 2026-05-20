"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_infer
Pattern hash: 3abc926270f6
Shape hash: 7fab38d3
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 240, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_58: "f32[512, 240, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        relu_default: "f32[512, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_58);  convolution_58 = None
        return relu_default


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
