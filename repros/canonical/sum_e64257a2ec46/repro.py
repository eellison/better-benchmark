"""
Standalone repro captured via capture_hook.
Label: tritonbench_sum_dim1_32768x1024
Pattern hash: e64257a2ec46
Shape hash: 5ed612bf
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[32768, 1024]"):
        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:234 in sum_reduction, code: return x.sum(dim=-1)
        sum_dim_int_list: "f32[32768]" = torch.ops.aten.sum.dim_IntList(arg0_1, [-1]);  arg0_1 = None
        return sum_dim_int_list


def _default_make_inputs():
    return [
    torch.randn([32768, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
