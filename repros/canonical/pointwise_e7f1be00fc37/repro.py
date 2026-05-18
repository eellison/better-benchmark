"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s8_g27
Pattern hash: e7f1be00fc37
Shape hash: 5fe738e0
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
    def forward(self, arg3_1: "f16[256, 1024]", mm: "f16[256, 1024]"):
        # No stacktrace found for following nodes
        le_scalar: "b8[256, 1024]" = torch.ops.aten.le.Scalar(arg3_1, 0);  arg3_1 = None
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[256, 1024]" = torch.ops.aten.where.self(le_scalar, full_default, mm);  le_scalar = full_default = mm = None
        return where_self


def _default_make_inputs():
    return [
    torch.randn([256, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([256, 1024], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
