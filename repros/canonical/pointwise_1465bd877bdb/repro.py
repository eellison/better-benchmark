"""
Standalone repro captured via capture_hook.
Label: vgg16_training
Pattern hash: 1465bd877bdb
Shape hash: 3a464cb8
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
    def forward(self, relu_2: "f32[4, 128, 112, 112]", full_default: "f32[]", getitem_37: "f32[4, 128, 112, 112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        le_scalar: "b8[4, 128, 112, 112]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_self: "f32[4, 128, 112, 112]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_37);  le_scalar = full_default = getitem_37 = None
        return where_self


def _default_make_inputs():
    return [
    torch.randn(6422528, dtype=torch.float32, device='cuda').as_strided([4, 128, 112, 112], [1605632, 1, 14336, 128]),  # relu_2
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(6422528, dtype=torch.float32, device='cuda').as_strided([4, 128, 112, 112], [1605632, 1, 14336, 128]),  # getitem_37
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
