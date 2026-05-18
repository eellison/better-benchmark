"""
Standalone repro captured via capture_hook.
Label: timm_efficientnet_b0
Pattern hash: 818690bf8db1
Shape hash: e0478824
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_77: "f32[8, 48, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_default: "f32[8, 48, 1, 1]" = torch.ops.aten.neg.default(convolution_77)
        exp_default: "f32[8, 48, 1, 1]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(convolution_77, add_tensor);  convolution_77 = add_tensor = None
        return div_tensor


def _default_make_inputs():
    return [
    torch.randn([8, 48, 1, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
