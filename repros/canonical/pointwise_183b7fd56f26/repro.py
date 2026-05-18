"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_11: "f32[2048, 16384]", mm_12: "f32[2048, 16384]"):
        # File: /tmp/scratch_space/better_benchmark/extract_reductions.py:738 in forward, code: x = x + self.down_proj(torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h))
        reshape_default: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(mm_11, [4, 512, 16384]);  mm_11 = None
        neg_default: "f32[4, 512, 16384]" = torch.ops.aten.neg.default(reshape_default)
        exp_default: "f32[4, 512, 16384]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[4, 512, 16384]" = torch.ops.aten.div.Tensor(reshape_default, add_tensor);  reshape_default = add_tensor = None
        reshape_default_1: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(mm_12, [4, 512, 16384]);  mm_12 = None
        mul_tensor: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(div_tensor, reshape_default_1);  div_tensor = reshape_default_1 = None
        reshape_default_2: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_tensor, [2048, 16384]);  mul_tensor = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([2048, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 16384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
