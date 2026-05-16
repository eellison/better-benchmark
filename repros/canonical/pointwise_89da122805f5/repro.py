"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s8_g21
Pattern hash: 89da122805f5
Shape hash: 55f85091
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f16[256, 2]"):
        # No stacktrace found for following nodes
        split_tensor = torch.ops.aten.split.Tensor(addmm_2, 1, 1);  addmm_2 = None
        getitem: "f16[256, 1]" = split_tensor[0]
        getitem_1: "f16[256, 1]" = split_tensor[1];  split_tensor = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([256, 2], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
