"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g86
Pattern hash: 7b803b65a438
Shape hash: ecd54916
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
    def forward(self, arg0_1: "f32[1, 4, 474, 768]"):
        # No stacktrace found for following nodes
        select_int: "f32[4, 474, 768]" = torch.ops.aten.select.int(arg0_1, 0, 0);  arg0_1 = None
        clone_default: "f32[4, 474, 768]" = torch.ops.aten.clone.default(select_int);  select_int = None
        return clone_default


def _default_make_inputs():
    return [
    torch.randn([1, 4, 474, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
