"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer
Pattern hash: 2c5e8cda4fb5
Shape hash: 4fa461c7
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 768], f16), S([1, 4096, 768]), S([1, 4096, 12, 64]), S([12, 4096, 64]), S([12, 64, 64]))"

class Repro(torch.nn.Module):
    def forward(self, mm_4: "f16[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[3]" = torch.ops.prims.inductor_seeds.default(3, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:511 in forward, code: query_key_vectors = self.query_key(hidden_states)
        reshape_default: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_4, _shape_param_0);  mm_4 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        reshape_default_1: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_default: "f16[1, 12, 4096, 64, 1]" = torch.ops.aten.unsqueeze.default(permute_default, 4);  permute_default = None
        unsqueeze_default_1: "f16[1, 12, 4096, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 5);  unsqueeze_default = None
        permute_default_1: "f16[1, 12, 1, 4096, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_1, [0, 1, 4, 2, 5, 3]);  unsqueeze_default_1 = None
        permute_default_2: "f16[12, 4096, 64, 1, 1, 1]" = torch.ops.aten.permute.default(permute_default_1, [1, 3, 5, 0, 2, 4]);  permute_default_1 = None
        reshape_default_2: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_2);  permute_default_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:719 in _hash_vectors, code: random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[12, 64, 1, 64]" = torch.ops.prims.inductor_random.default([12, 64, 1, 64], inductor_lookup_seed_default, 'randn');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[12, 64, 1, 64]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        unsqueeze_default_2: "f16[12, 64, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, 4);  convert_element_type_default = None
        unsqueeze_default_3: "f16[12, 64, 1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 5);  unsqueeze_default_2 = None
        permute_default_3: "f16[1, 12, 1, 1, 64, 64]" = torch.ops.aten.permute.default(unsqueeze_default_3, [4, 0, 2, 5, 3, 1]);  unsqueeze_default_3 = None
        permute_default_4: "f16[12, 64, 1, 1, 64, 1]" = torch.ops.aten.permute.default(permute_default_3, [1, 5, 0, 2, 4, 3]);  permute_default_3 = None
        reshape_default_3: "f16[12, 64, 64]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_3);  permute_default_4 = _shape_param_3 = None
        return (reshape_default_2, reshape_default_3)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
