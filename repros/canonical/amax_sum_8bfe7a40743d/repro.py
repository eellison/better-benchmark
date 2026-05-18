"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=amax, ranges=['8', '12', '64', '64', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['8', '12', '64', '64', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
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
    def forward(self, remainder_1: "i64[8, 12, 4096]", bmm_1: "f32[6144, 64, 128]", arg2_1: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:426 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape)
        reshape_default: "i64[8, 12, 64, 64]" = torch.ops.aten.reshape.default(remainder_1, [8, 12, -1, 64]);  remainder_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:905 in _attend, code: self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
        unsqueeze_default: "i64[8, 12, 64, 64, 1]" = torch.ops.aten.unsqueeze.default(reshape_default, -1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_tensor: "i64[8, 12, 1, 64]" = torch.ops.aten.slice.Tensor(reshape_default, 2, -1, 9223372036854775807)
        slice_tensor_1: "i64[8, 12, 63, 64]" = torch.ops.aten.slice.Tensor(reshape_default, 2, 0, -1)
        cat_default: "i64[8, 12, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:399 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_default_1: "i64[8, 12, 64, 128]" = torch.ops.aten.cat.default([cat_default, reshape_default], 3);  cat_default = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:905 in _attend, code: self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
        unsqueeze_default_1: "i64[8, 12, 64, 1, 128]" = torch.ops.aten.unsqueeze.default(cat_default_1, -2);  cat_default_1 = None
        ne_tensor: "b8[8, 12, 64, 64, 128]" = torch.ops.aten.ne.Tensor(unsqueeze_default, unsqueeze_default_1);  unsqueeze_default = unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:848 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        reshape_default_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.reshape.default(bmm_1, [8, 12, 64, 64, 128]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:910 in _attend, code: query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)
        where_self: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.where.self(ne_tensor, reshape_default_1, arg2_1);  ne_tensor = reshape_default_1 = arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:915 in _attend, code: logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        amax_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.amax.default(where_self, [-1], True)
        abs_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax_default)
        eq_scalar: "b8[8, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_default, inf);  abs_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq_scalar, full_default, amax_default);  eq_scalar = full_default = amax_default = None
        sub_tensor: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(where_self, where_self_1)
        exp_default: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True);  exp_default = None
        log_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        add_tensor: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log_default, where_self_1);  log_default = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:917 in _attend, code: attention_probs = torch.exp(query_key_dots - logits)
        sub_tensor_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(where_self, add_tensor);  where_self = add_tensor = None
        exp_default_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:930 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_default: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_default_1, [8, 12, 64, 64, 128]);  exp_default_1 = None
        reshape_default_2: "f32[6144, 64, 128]" = torch.ops.aten.reshape.default(expand_default, [6144, 64, 128]);  expand_default = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randint(0, 2, [8, 12, 4096], dtype=torch.int64, device='cuda'),
    torch.randn([6144, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
