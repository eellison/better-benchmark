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
    def forward(self, getitem_55: "i64[2048, 8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:386 in grouped_mm_experts_forward, code: expert_ids = top_k_index.reshape(-1)  # (S,)
        reshape_default: "i64[16384]" = torch.ops.aten.reshape.default(getitem_55, [-1]);  getitem_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:389 in grouped_mm_experts_forward, code: expert_ids_g, perm = torch.sort(expert_ids)
        sort_default = torch.ops.aten.sort.default(reshape_default);  reshape_default = None
        getitem: "i64[16384]" = sort_default[0]
        getitem_56: "i64[16384]" = sort_default[1];  sort_default = None
        return (getitem, getitem_56)


def _default_make_inputs():
    return [
    torch.randint(0, 100, [2048, 8], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
