"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_infer
Pattern hash: 182f6f9450b9
Shape hash: 0168f91a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 1152, 1, 1], f32), T([128, 1152, 7, 7], f32, stride=(56448, 1, 8064, 1152)))"

class Repro(torch.nn.Module):
    def forward(self, convolution_78: "f32[128, 1152, 1, 1]", div_46: "f32[128, 1152, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_default: "f32[128, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_78);  convolution_78 = None
        mul_tensor: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(div_46, sigmoid_default);  div_46 = sigmoid_default = None
        return mul_tensor


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
