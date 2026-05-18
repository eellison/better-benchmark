"""
Standalone repro captured via capture_hook.
Label: genai_swiglu_decode
Pattern hash: 89da122805f5
Shape hash: 5d73f110
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[32, 1, 8192]"):
        # No stacktrace found for following nodes
        split_tensor = torch.ops.aten.split.Tensor(arg0_1, 4096, -1);  arg0_1 = None
        getitem: "bf16[32, 1, 4096]" = split_tensor[0]
        getitem_1: "bf16[32, 1, 4096]" = split_tensor[1];  split_tensor = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([32, 1, 8192], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
