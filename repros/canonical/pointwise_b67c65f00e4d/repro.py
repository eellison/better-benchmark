"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: b67c65f00e4d
Shape hash: 119642d5
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, relu_7: "f32[512, 24, 1, 1]", full_default: "f32[]", getitem_242: "f32[512, 24, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        le_scalar: "b8[512, 24, 1, 1]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_self: "f32[512, 24, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_242);  le_scalar = full_default = getitem_242 = None
        return where_self


def _default_make_inputs():
    return [
    torch.randn([512, 24, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([512, 24, 1, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
