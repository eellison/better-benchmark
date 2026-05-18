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
    def forward(self, arg0_1: "f32[8, 12, 4096, 64]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:750 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_default: "f32[8, 12, 4096, 64, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 4);  arg0_1 = None
        unsqueeze_default_1: "f32[8, 12, 4096, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 5);  unsqueeze_default = None
        permute_default: "f32[8, 12, 1, 4096, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_1, [0, 1, 4, 2, 5, 3]);  unsqueeze_default_1 = None
        permute_default_1: "f32[12, 8, 4096, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default, [1, 0, 3, 5, 2, 4]);  permute_default = None
        reshape_default: "f32[12, 32768, 64]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_0);  permute_default_1 = _shape_param_0 = None
        return reshape_default


def _default_make_inputs():
    return [
    torch.randn([8, 12, 4096, 64], dtype=torch.float32, device='cuda'),
    [12, 32768, 64],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
