"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, div: "f32[32, 16, 128, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:228 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_default: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div, [32, 16, 128, 128]);  div = None
        reshape_default: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_default, [512, 128, 128]);  expand_default = None
        permute_default: "f32[512, 128, 128]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        return permute_default


def _default_make_inputs():
    return [
    torch.randn([32, 16, 128, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
