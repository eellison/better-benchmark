"""
Standalone repro captured via capture_hook.
Label: hf_Reformer_inference
Pattern hash: 3a85ade71999
Shape hash: 79ca7511
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
    def forward(self, bmm: "f32[12, 32768, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        reshape_default: "f32[12, 8, 4096, 1, 1, 64]" = torch.ops.aten.reshape.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute_default: "f32[8, 12, 1, 4096, 64, 1]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 4, 2, 5, 3]);  reshape_default = None
        reshape_default_1: "f32[8, 12, 1, 4096, 64]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:724 in _hash_vectors, code: rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
        neg_default: "f32[8, 12, 1, 4096, 64]" = torch.ops.aten.neg.default(reshape_default_1)
        cat_default: "f32[8, 12, 1, 4096, 128]" = torch.ops.aten.cat.default([reshape_default_1, neg_default], -1);  reshape_default_1 = neg_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:725 in _hash_vectors, code: buckets = torch.argmax(rotated_vectors, dim=-1)
        argmax_default: "i64[8, 12, 1, 4096]" = torch.ops.aten.argmax.default(cat_default, -1);  cat_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:753 in _hash_vectors, code: offsets = torch.arange(num_hashes, device=vectors.device)
        iota_default: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:754 in _hash_vectors, code: offsets = (offsets * num_buckets).view((1, 1, -1, 1))
        mul_tensor: "i64[1]" = torch.ops.aten.mul.Tensor(iota_default, 128);  iota_default = None
        reshape_default_2: "i64[1, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor, [1, 1, -1, 1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:757 in _hash_vectors, code: offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        expand_default: "i64[8, 12, 1, 1]" = torch.ops.aten.expand.default(reshape_default_2, _shape_param_2);  reshape_default_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:758 in _hash_vectors, code: offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)
        add_tensor: "i64[8, 12, 1, 4096]" = torch.ops.aten.add.Tensor(argmax_default, expand_default);  argmax_default = expand_default = None
        reshape_default_3: "i64[8, 12, 4096]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_3);  add_tensor = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:155 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        mul_tensor_1: "i64[8, 12, 4096]" = torch.ops.aten.mul.Tensor(reshape_default_3, 4096);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:153 in _stable_argsort, code: scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
        iota_default_1: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_4: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_default_1, [1, 1, -1]);  iota_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:154 in _stable_argsort, code: scale_offset = scale_offset.expand(vector.shape)
        expand_default_1: "i64[8, 12, 4096]" = torch.ops.aten.expand.default(reshape_default_4, _shape_param_4);  reshape_default_4 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:155 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        remainder_scalar: "i64[8, 12, 4096]" = torch.ops.aten.remainder.Scalar(expand_default_1, 4096);  expand_default_1 = None
        add_tensor_1: "i64[8, 12, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, remainder_scalar);  mul_tensor_1 = remainder_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:156 in _stable_argsort, code: return torch.argsort(scaled_vector, dim=dim)
        sort_default = torch.ops.aten.sort.default(add_tensor_1);  add_tensor_1 = None
        getitem: "i64[8, 12, 4096]" = sort_default[1];  sort_default = None
        return getitem


def _default_make_inputs():
    return [
    torch.randn([12, 32768, 64], dtype=torch.float32, device='cuda'),
    [12, 8, 4096, 1, 1, 64],  # _shape_param_0
    [8, 12, 1, 4096, 64],  # _shape_param_1
    [8, 12, 1, 1],  # _shape_param_2
    [8, 12, 4096],  # _shape_param_3
    [8, 12, 4096],  # _shape_param_4
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
