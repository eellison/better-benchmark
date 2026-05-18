"""
Standalone repro captured via capture_hook.
Label: hf_Reformer_training
Pattern hash: a585c2e89135
Shape hash: d4e37485
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
    def forward(self, tangents_1: "f32[8, 4096, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1780 in forward, code: hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        slice_tensor: "f32[8, 4096, 256]" = torch.ops.aten.slice.Tensor(tangents_1, 2, 0, 256)
        slice_tensor_1: "f32[8, 4096, 256]" = torch.ops.aten.slice.Tensor(tangents_1, 2, 256, 512);  tangents_1 = None
        add_tensor: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        return add_tensor


def _default_make_inputs():
    return [
    torch.randn([8, 4096, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
