"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s6_g10
Pattern hash: 1c629e1a11a7
Shape hash: 648bba09
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_134: "bf16[256, 4, 16, 60]", getitem_135: "bf16[256, 4, 16, 60]", getitem_136: "bf16[256, 4, 16, 60]"):
        # No stacktrace found for following nodes
        constant_pad_nd_default: "bf16[256, 4, 16, 64]" = torch.ops.aten.constant_pad_nd.default(getitem_134, [0, 4], 0.0);  getitem_134 = None
        constant_pad_nd_default_1: "bf16[256, 4, 16, 64]" = torch.ops.aten.constant_pad_nd.default(getitem_135, [0, 4], 0.0);  getitem_135 = None
        constant_pad_nd_default_2: "bf16[256, 4, 16, 64]" = torch.ops.aten.constant_pad_nd.default(getitem_136, [0, 4], 0.0);  getitem_136 = None
        return (constant_pad_nd_default, constant_pad_nd_default_1, constant_pad_nd_default_2)


def _default_make_inputs():
    return [
    torch.randn(2948640, dtype=torch.bfloat16, device='cuda').as_strided([256, 4, 16, 60], [11520, 60, 720, 1]),  # getitem_134
    torch.randn(2948640, dtype=torch.bfloat16, device='cuda').as_strided([256, 4, 16, 60], [11520, 60, 720, 1]),  # getitem_135
    torch.randn(2948640, dtype=torch.bfloat16, device='cuda').as_strided([256, 4, 16, 60], [11520, 60, 720, 1]),  # getitem_136
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
