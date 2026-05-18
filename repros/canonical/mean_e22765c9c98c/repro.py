"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=mean, ranges=['8', '12', '64', '64', '1'], reduction_ranges=[]
#   origins: ['aten.mean.dim']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_1: "i64[8, 12, 4096]", arg0_1: "f32[8, 12, 4096, 64]", arg1_1: "f32[8, 12, 4096, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:591 in torch_dynamo_resume_in_forward_at_572, code: sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length
        remainder_scalar: "i64[8, 12, 4096]" = torch.ops.aten.remainder.Scalar(getitem_1, 4096);  getitem_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1097 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_default: "f32[8, 12, 4096, 64]" = torch.ops.aten.repeat.default(arg0_1, [1, 1, 1, 1]);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1096 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_default: "i64[8, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_scalar, -1)
        expand_default: "i64[8, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default, [-1, -1, -1, 64]);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1098 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather_default: "f32[8, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_default, 2, expand_default);  repeat_default = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:424 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        reshape_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather_default, [8, 12, -1, 64, 64]);  gather_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:848 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand_default_1: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.expand.default(reshape_default, [8, 12, 64, 64, 64])
        reshape_default_1: "f32[6144, 64, 64]" = torch.ops.aten.reshape.default(expand_default_1, [6144, 64, 64]);  expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1088 in _len_norm, code: variance = torch.mean(x**2, -1, keepdim=True)
        pow_tensor_scalar: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default, 2)
        mean_dim: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1089 in _len_norm, code: norm_x = x * torch.rsqrt(variance + epsilon)
        add_tensor: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.mul.Tensor(reshape_default, rsqrt_default);  reshape_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:626 in torch_dynamo_resume_in_forward_at_572, code: sqrt_num = np.sqrt(self.attention_head_size)
        full_default: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1081 in _len_and_dim_norm, code: vectors = vectors / sqrt_num
        div_tensor: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.div.Tensor(mul_tensor, full_default);  mul_tensor = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_tensor: "f32[8, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(div_tensor, 2, -1, 9223372036854775807)
        slice_tensor_1: "f32[8, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(div_tensor, 2, 0, -1)
        cat_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:399 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_default_1: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_default, div_tensor], 3);  cat_default = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:848 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_default: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_default_1, [0, 1, 2, 4, 3]);  cat_default_1 = None
        expand_default_2: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_default, [8, 12, 64, 64, 128]);  permute_default = None
        reshape_default_2: "f32[6144, 64, 128]" = torch.ops.aten.reshape.default(expand_default_2, [6144, 64, 128]);  expand_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1097 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_default_1: "f32[8, 12, 4096, 64]" = torch.ops.aten.repeat.default(arg1_1, [1, 1, 1, 1]);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1096 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_default_1: "i64[8, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_scalar, -1);  remainder_scalar = None
        expand_default_3: "i64[8, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default_1, [-1, -1, -1, 64]);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1098 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather_default_1: "f32[8, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_default_1, 2, expand_default_3);  repeat_default_1 = expand_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:424 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        reshape_default_3: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather_default_1, [8, 12, -1, 64, 64]);  gather_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_tensor_2: "f32[8, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_3, 2, -1, 9223372036854775807)
        slice_tensor_3: "f32[8, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_3, 2, 0, -1)
        cat_default_2: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_tensor_2, slice_tensor_3], 2);  slice_tensor_2 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:399 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_default_3: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_default_2, reshape_default_3], 3);  cat_default_2 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:930 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_default_4: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_default_3, [8, 12, 64, 128, 64]);  cat_default_3 = None
        reshape_default_4: "f32[6144, 128, 64]" = torch.ops.aten.reshape.default(expand_default_4, [6144, 128, 64]);  expand_default_4 = None
        return (reshape_default_1, reshape_default_2, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [8, 12, 4096], dtype=torch.int64, device='cuda'),
    torch.randn([8, 12, 4096, 64], dtype=torch.float32, device='cuda'),
    torch.randn([8, 12, 4096, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
