"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer
Pattern hash: eb2799a2706a
Shape hash: 81a6dedf
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
_shapes_config = "(T([12, 4096, 64], f16), T([1, 12, 4096, 64], f16, stride=(3145728, 64, 768, 1)), S([12, 4096, 1, 1, 1, 64]), S([1, 12, 1, 4096, 64]), S([1, 12, 1, 1]), S([1, 12, 4096]), S([1, 12, 4096]), S([-1, -1, -1, 64]), S([1, 12, -1, 64, 64]), S([1, 12, 64, 64, 64]), S([768, 64, 64]), S([1, 12, 64, 64, 128]), S([768, 64, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_12: "f16[12, 4096, 64]", permute_63: "f16[1, 12, 4096, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:721 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        reshape_default: "f16[12, 4096, 1, 1, 1, 64]" = torch.ops.aten.reshape.default(bmm_12, _shape_param_0);  bmm_12 = _shape_param_0 = None
        permute_default: "f16[1, 12, 1, 4096, 64, 1]" = torch.ops.aten.permute.default(reshape_default, [3, 0, 4, 1, 5, 2]);  reshape_default = None
        reshape_default_1: "f16[1, 12, 1, 4096, 64]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:724 in _hash_vectors, code: rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
        neg_default: "f16[1, 12, 1, 4096, 64]" = torch.ops.aten.neg.default(reshape_default_1)
        cat_default: "f16[1, 12, 1, 4096, 128]" = torch.ops.aten.cat.default([reshape_default_1, neg_default], -1);  reshape_default_1 = neg_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:725 in _hash_vectors, code: buckets = torch.argmax(rotated_vectors, dim=-1)
        argmax_default: "i64[1, 12, 1, 4096]" = torch.ops.aten.argmax.default(cat_default, -1);  cat_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:753 in _hash_vectors, code: offsets = torch.arange(num_hashes, device=vectors.device)
        iota_default: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:754 in _hash_vectors, code: offsets = (offsets * num_buckets).view((1, 1, -1, 1))
        mul_tensor: "i64[1]" = torch.ops.aten.mul.Tensor(iota_default, 128);  iota_default = None
        reshape_default_2: "i64[1, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor, [1, 1, -1, 1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:757 in _hash_vectors, code: offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        expand_default: "i64[1, 12, 1, 1]" = torch.ops.aten.expand.default(reshape_default_2, _shape_param_2);  reshape_default_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:758 in _hash_vectors, code: offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)
        add_tensor: "i64[1, 12, 1, 4096]" = torch.ops.aten.add.Tensor(argmax_default, expand_default);  argmax_default = expand_default = None
        reshape_default_3: "i64[1, 12, 4096]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_3);  add_tensor = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:155 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        mul_tensor_1: "i64[1, 12, 4096]" = torch.ops.aten.mul.Tensor(reshape_default_3, 4096);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:153 in _stable_argsort, code: scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
        iota_default_1: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_4: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_default_1, [1, 1, -1]);  iota_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:154 in _stable_argsort, code: scale_offset = scale_offset.expand(vector.shape)
        expand_default_1: "i64[1, 12, 4096]" = torch.ops.aten.expand.default(reshape_default_4, _shape_param_4);  reshape_default_4 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:155 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        remainder_scalar: "i64[1, 12, 4096]" = torch.ops.aten.remainder.Scalar(expand_default_1, 4096);  expand_default_1 = None
        add_tensor_1: "i64[1, 12, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, remainder_scalar);  mul_tensor_1 = remainder_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:156 in _stable_argsort, code: return torch.argsort(scaled_vector, dim=dim)
        sort_default = torch.ops.aten.sort.default(add_tensor_1);  add_tensor_1 = None
        getitem: "i64[1, 12, 4096]" = sort_default[1];  sort_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:563 in forward, code: sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length
        remainder_scalar_1: "i64[1, 12, 4096]" = torch.ops.aten.remainder.Scalar(getitem, 4096);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1063 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_63, [1, 1, 1, 1]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1062 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_default: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_scalar_1, -1);  remainder_scalar_1 = None
        expand_default_2: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_5);  unsqueeze_default = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1064 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_default, 2, expand_default_2);  repeat_default = expand_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        reshape_default_5: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather_default, _shape_param_6);  gather_default = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand_default_3: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.expand.default(reshape_default_5, _shape_param_7);  _shape_param_7 = None
        reshape_default_6: "f16[768, 64, 64]" = torch.ops.aten.reshape.default(expand_default_3, _shape_param_8);  expand_default_3 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1054 in _len_norm, code: variance = torch.mean(x**2, -1, keepdim=True)
        pow_tensor_scalar: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_5, 2)
        mean_dim: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1055 in _len_norm, code: norm_x = x * torch.rsqrt(variance + epsilon)
        add_tensor_2: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_2: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.mul.Tensor(reshape_default_5, rsqrt_default);  reshape_default_5 = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:598 in forward, code: sqrt_num = np.sqrt(self.attention_head_size)
        full_default: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1047 in _len_and_dim_norm, code: vectors = vectors / sqrt_num
        div_tensor: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.div.Tensor(mul_tensor_2, full_default);  mul_tensor_2 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_tensor: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(div_tensor, 2, -1, 9223372036854775807)
        slice_tensor_1: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(div_tensor, 2, 0, -1)
        cat_default_1: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_default_2: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_default_1, div_tensor], 3);  cat_default_1 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_default_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_default_2, [0, 1, 2, 4, 3]);  cat_default_2 = None
        expand_default_4: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_9);  permute_default_1 = _shape_param_9 = None
        reshape_default_7: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_default_4, _shape_param_10);  expand_default_4 = _shape_param_10 = None
        return (reshape_default_6, reshape_default_7)



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
