"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import glob
import os
import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, _grouped_mm_6: "bf16[16384, 1536]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        split_tensor = torch.ops.aten.split.Tensor(_grouped_mm_6, 768, -1);  _grouped_mm_6 = None
        getitem: "bf16[16384, 768]" = split_tensor[0]
        getitem_1: "bf16[16384, 768]" = split_tensor[1];  split_tensor = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([16384, 1536], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
