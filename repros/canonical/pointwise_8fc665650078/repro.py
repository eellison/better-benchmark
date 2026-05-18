"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g53
Pattern hash: 8fc665650078
Shape hash: fa4dd950
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
    def forward(self, arg263_1: "f16[8, 1000]"):
        # No stacktrace found for following nodes
        div_tensor: "f16[8, 1000]" = torch.ops.aten.div.Tensor(arg263_1, 2);  arg263_1 = None
        return div_tensor


def _default_make_inputs():
    return [
    torch.randn([8, 1000], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
