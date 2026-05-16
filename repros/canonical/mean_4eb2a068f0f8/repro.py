"""
Standalone repro captured via capture_hook.
Label: hf_Reformer_inference
Pattern hash: 4eb2a068f0f8
Shape hash: bbf19dbe
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_1: "i64[8, 12, 4096]", arg0_1: "f32[8, 12, 4096, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:563 in torch_dynamo_resume_in_forward_at_544, code: sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length
        remainder_scalar: "i64[8, 12, 4096]" = torch.ops.aten.remainder.Scalar(getitem_1, 4096);  getitem_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1063 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_default: "f32[8, 12, 4096, 64]" = torch.ops.aten.repeat.default(arg0_1, [1, 1, 1, 1]);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1062 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_default: "i64[8, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_scalar, -1);  remainder_scalar = None
        expand_default: "i64[8, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1064 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather_default: "f32[8, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_default, 2, expand_default);  repeat_default = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        reshape_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather_default, _shape_param_1);  gather_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand_default_1: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.expand.default(reshape_default, _shape_param_2);  _shape_param_2 = None
        reshape_default_1: "f32[6144, 64, 64]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_3);  expand_default_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1054 in _len_norm, code: variance = torch.mean(x**2, -1, keepdim=True)
        pow_tensor_scalar: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default, 2)
        mean_dim: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1055 in _len_norm, code: norm_x = x * torch.rsqrt(variance + epsilon)
        add_tensor: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.mul.Tensor(reshape_default, rsqrt_default);  reshape_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:598 in torch_dynamo_resume_in_forward_at_544, code: sqrt_num = np.sqrt(self.attention_head_size)
        full_default: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1047 in _len_and_dim_norm, code: vectors = vectors / sqrt_num
        div_tensor: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.div.Tensor(mul_tensor, full_default);  mul_tensor = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_tensor: "f32[8, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(div_tensor, 2, -1, 9223372036854775807)
        slice_tensor_1: "f32[8, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(div_tensor, 2, 0, -1)
        cat_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_default_1: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_default, div_tensor], 3);  cat_default = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_default: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_default_1, [0, 1, 2, 4, 3]);  cat_default_1 = None
        expand_default_2: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_default, _shape_param_4);  permute_default = _shape_param_4 = None
        reshape_default_2: "f32[6144, 64, 128]" = torch.ops.aten.reshape.default(expand_default_2, _shape_param_5);  expand_default_2 = _shape_param_5 = None
        return (reshape_default_1, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [8, 12, 4096], dtype=torch.int64, device='cuda'),
    torch.randn(25165824, dtype=torch.float32, device='cuda').as_strided([8, 12, 4096, 64], [3145728, 64, 768, 1]),  # arg0_1
    [-1, -1, -1, 64],  # _shape_param_0
    [8, 12, -1, 64, 64],  # _shape_param_1
    [8, 12, 64, 64, 64],  # _shape_param_2
    [6144, 64, 64],  # _shape_param_3
    [8, 12, 64, 64, 128],  # _shape_param_4
    [6144, 64, 128],  # _shape_param_5
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
