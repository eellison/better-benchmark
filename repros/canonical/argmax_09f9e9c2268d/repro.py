"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=argmax, ranges=['8', '12', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.argmax.default']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm: "f32[12, 32768, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:750 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        reshape_default: "f32[12, 8, 4096, 1, 1, 64]" = torch.ops.aten.reshape.default(bmm, [12, 8, 4096, 1, 1, 64]);  bmm = None
        permute_default: "f32[8, 12, 1, 4096, 64, 1]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 4, 2, 5, 3]);  reshape_default = None
        reshape_default_1: "f32[8, 12, 1, 4096, 64]" = torch.ops.aten.reshape.default(permute_default, [8, 12, 1, 4096, 64]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:753 in _hash_vectors, code: rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
        neg_default: "f32[8, 12, 1, 4096, 64]" = torch.ops.aten.neg.default(reshape_default_1)
        cat_default: "f32[8, 12, 1, 4096, 128]" = torch.ops.aten.cat.default([reshape_default_1, neg_default], -1);  reshape_default_1 = neg_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:754 in _hash_vectors, code: buckets = torch.argmax(rotated_vectors, dim=-1)
        argmax_default: "i64[8, 12, 1, 4096]" = torch.ops.aten.argmax.default(cat_default, -1);  cat_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:782 in _hash_vectors, code: offsets = torch.arange(num_hashes, device=vectors.device)
        iota_default: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:783 in _hash_vectors, code: offsets = (offsets * num_buckets).view((1, 1, -1, 1))
        mul_tensor: "i64[1]" = torch.ops.aten.mul.Tensor(iota_default, 128);  iota_default = None
        reshape_default_2: "i64[1, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor, [1, 1, -1, 1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:786 in _hash_vectors, code: offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        expand_default: "i64[8, 12, 1, 1]" = torch.ops.aten.expand.default(reshape_default_2, [8, 12, 1, 1]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:787 in _hash_vectors, code: offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)
        add_tensor: "i64[8, 12, 1, 4096]" = torch.ops.aten.add.Tensor(argmax_default, expand_default);  argmax_default = expand_default = None
        reshape_default_3: "i64[8, 12, 4096]" = torch.ops.aten.reshape.default(add_tensor, [8, 12, 4096]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:181 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        mul_tensor_1: "i64[8, 12, 4096]" = torch.ops.aten.mul.Tensor(reshape_default_3, 4096)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:179 in _stable_argsort, code: scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
        iota_default_1: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_4: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_default_1, [1, 1, -1]);  iota_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:180 in _stable_argsort, code: scale_offset = scale_offset.expand(vector.shape)
        expand_default_1: "i64[8, 12, 4096]" = torch.ops.aten.expand.default(reshape_default_4, [8, 12, 4096]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:181 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        remainder_scalar: "i64[8, 12, 4096]" = torch.ops.aten.remainder.Scalar(expand_default_1, 4096);  expand_default_1 = None
        add_tensor_1: "i64[8, 12, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, remainder_scalar);  mul_tensor_1 = remainder_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:182 in _stable_argsort, code: return torch.argsort(scaled_vector, dim=dim)
        sort_default = torch.ops.aten.sort.default(add_tensor_1);  add_tensor_1 = None
        getitem: "i64[8, 12, 4096]" = sort_default[1];  sort_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:703 in torch_dynamo_resume_in_forward_at_572, code: self.num_attention_heads, self.attention_head_size, self.hidden_size
        reshape_default_5: "i64[8, 12, 1, 4096]" = torch.ops.aten.reshape.default(reshape_default_3, [8, 12, 1, -1]);  reshape_default_3 = None
        return (reshape_default_5, getitem)


def _default_make_inputs():
    return [
    torch.randn([12, 32768, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
