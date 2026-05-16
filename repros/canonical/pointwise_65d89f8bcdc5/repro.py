"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch
from torch import device
import torch._inductor.inductor_prims

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:748 in _hash_vectors, code: random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[12, 64, 1, 64]" = torch.ops.prims.inductor_random.default([12, 64, 1, 64], inductor_lookup_seed_default, 'randn');  inductor_lookup_seed_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:750 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_default: "f32[12, 64, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(inductor_random_default, 4);  inductor_random_default = None
        unsqueeze_default_1: "f32[12, 64, 1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 5);  unsqueeze_default = None
        reshape_default: "f32[12, 64, 64]" = torch.ops.aten.reshape.default(unsqueeze_default_1, [12, 64, 64]);  unsqueeze_default_1 = None
        return reshape_default


def _default_make_inputs():
    return [

    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
