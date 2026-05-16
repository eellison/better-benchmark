"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g167
Pattern hash: 14bdbf866838
Shape hash: ecac49c6
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 474, 768]"):
        # No stacktrace found for following nodes
        full_default: "f32[1, 4, 474, 768]" = torch.ops.aten.full.default([1, 4, 474, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[1, 4, 474, 768]" = torch.ops.aten.select_scatter.default(full_default, arg0_1, 0, 0);  full_default = arg0_1 = None
        return select_scatter_default


def _default_make_inputs():
    return [
    torch.randn([4, 474, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
