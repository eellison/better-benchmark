"""
Standalone repro captured via capture_hook.
Label: hf_Reformer_inference
Pattern hash: cc3d08b79f6e
Shape hash: aa83b929
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 12, 4096, 64]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_default: "f32[8, 12, 4096, 64, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 4);  arg0_1 = None
        unsqueeze_default_1: "f32[8, 12, 4096, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 5);  unsqueeze_default = None
        permute_default: "f32[8, 12, 1, 4096, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_1, [0, 1, 4, 2, 5, 3]);  unsqueeze_default_1 = None
        permute_default_1: "f32[12, 8, 4096, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default, [1, 0, 3, 5, 2, 4]);  permute_default = None
        reshape_default: "f32[12, 32768, 64]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_0);  permute_default_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:719 in _hash_vectors, code: random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[12, 64, 1, 64]" = torch.ops.prims.inductor_random.default([12, 64, 1, 64], inductor_lookup_seed_default, 'randn');  inductor_lookup_seed_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_default_2: "f32[12, 64, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(inductor_random_default, 4);  inductor_random_default = None
        unsqueeze_default_3: "f32[12, 64, 1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 5);  unsqueeze_default_2 = None
        reshape_default_1: "f32[12, 64, 64]" = torch.ops.aten.reshape.default(unsqueeze_default_3, _shape_param_1);  unsqueeze_default_3 = _shape_param_1 = None
        return (reshape_default, reshape_default_1)


def _default_make_inputs():
    return [
    torch.randn(25165824, dtype=torch.float32, device='cuda').as_strided([8, 12, 4096, 64], [3145728, 64, 768, 1]),  # arg0_1
    [12, 32768, 64],  # _shape_param_0
    [12, 64, 64],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
