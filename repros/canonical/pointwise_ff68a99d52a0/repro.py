"""
Standalone repro captured via capture_hook.
Label: tritonbench_vector_exp_1048576
Pattern hash: ff68a99d52a0
Shape hash: 0474e40a
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
    def forward(self, arg0_1: "f32[1048576]"):
        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:260 in vector_exp, code: return torch.exp(x)
        exp_default: "f32[1048576]" = torch.ops.aten.exp.default(arg0_1);  arg0_1 = None
        return exp_default


def _default_make_inputs():
    return [
    torch.randn([1048576], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
