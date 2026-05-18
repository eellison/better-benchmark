"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g62
Pattern hash: 86266e03b17c
Shape hash: e2e87efd
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
    def forward(self, arg0_1: "i64[2, 199981]", arg1_1: "i64[2, 10000]"):
        # No stacktrace found for following nodes
        cat_default: "i64[2, 209981]" = torch.ops.aten.cat.default([arg0_1, arg1_1], 1);  arg0_1 = arg1_1 = None
        return cat_default


def _default_make_inputs():
    return [
    torch.randint(0, 2, [2, 199981], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [2, 10000], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
