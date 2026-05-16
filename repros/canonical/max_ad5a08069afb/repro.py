"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=max, ranges=[], reduction_ranges=[]
#   origins: ['aten.max.default']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[8, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:292 in forward, code: max_position_id = position_ids.max().item()
        max_default: "i64[]" = torch.ops.aten.max.default(arg0_1);  arg0_1 = None
        return max_default


def _default_make_inputs():
    return [
    torch.randint(0, 2, [8, 4096], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
