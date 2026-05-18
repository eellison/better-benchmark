"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s8_g27
Pattern hash: dd054684d981
Shape hash: 2b3a5dba
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
    def forward(self, cat: "f16[256, 2]", where: "f16[256, 1024]", arg2_1: "f16[256, 1024]", full: "f16[]", mm_2: "f16[256, 1024]"):
        # No stacktrace found for following nodes
        permute_default: "f16[2, 256]" = torch.ops.aten.permute.default(cat, [1, 0]);  cat = None
        permute_default_1: "f16[1024, 256]" = torch.ops.aten.permute.default(where, [1, 0]);  where = None
        le_scalar: "b8[256, 1024]" = torch.ops.aten.le.Scalar(arg2_1, 0);  arg2_1 = None
        where_self: "f16[256, 1024]" = torch.ops.aten.where.self(le_scalar, full, mm_2);  le_scalar = full = mm_2 = None
        permute_default_2: "f16[1024, 256]" = torch.ops.aten.permute.default(where_self, [1, 0]);  where_self = None
        return (permute_default, permute_default_1, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([256, 2], dtype=torch.float16, device='cuda'),
    torch.randn([256, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([256, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([], dtype=torch.float16, device='cuda'),
    torch.randn([256, 1024], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
