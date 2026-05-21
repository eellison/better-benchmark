"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['4', '8', '1', '512', '64'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4', '8', '1', '512', '64'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import glob
import os
import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 128256], bf16), T([4, 32, 512, 64], bf16), T([4, 32, 512, 64], bf16), T([1, 1, 512, 64], bf16), T([4, 8, 512, 64], bf16), T([1, 1, 512, 64], bf16), T([4, 32, 512, 64], bf16, stride=(1048576, 64, 2048, 1)), T([4, 32, 512, 64], bf16), T([512, 2048], bf16), T([512, 2048], bf16), T([2048, 2048], bf16))"

class Repro(torch.nn.Module):
    def forward(self, view_329: "bf16[2048, 128256]", getitem_191: "bf16[4, 32, 512, 64]", getitem_190: "bf16[4, 32, 512, 64]", unsqueeze_15: "bf16[1, 1, 512, 64]", full_default_38: "bf16[4, 8, 512, 64]", unsqueeze_14: "bf16[1, 1, 512, 64]", getitem_189: "bf16[4, 32, 512, 64]", full_default_40: "bf16[4, 32, 512, 64]", primals_7: "bf16[512, 2048]", primals_6: "bf16[512, 2048]", primals_5: "bf16[2048, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:487 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_default: "bf16[128256, 2048]" = torch.ops.aten.permute.default(view_329, [1, 0]);  view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        reshape_default: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_191, [4, 8, 4, 512, 64]);  getitem_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_dim_int_list: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(reshape_default, [2], True);  reshape_default = None
        squeeze_dim: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 2);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        reshape_default_1: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.reshape.default(getitem_190, [4, 8, 4, 512, 64]);  getitem_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_dim_int_list_1: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [2], True);  reshape_default_1 = None
        squeeze_dim_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 2);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_dim_1, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_tensor: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor, 3, 0, 32)
        slice_tensor_1: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor, 3, 32, 64);  mul_tensor = None
        neg_default: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_default: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, neg_default, 3, 32, 9223372036854775807);  neg_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_default_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_38, slice_tensor_1, 3, 0, 32);  full_default_38 = slice_tensor_1 = None
        add_tensor: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1);  slice_scatter_default = slice_scatter_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(squeeze_dim_1, unsqueeze_14);  squeeze_dim_1 = None
        add_tensor_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_1);  add_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_2: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_189, unsqueeze_15);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_tensor_2: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor_2, 3, 0, 32)
        slice_tensor_3: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor_2, 3, 32, 64);  mul_tensor_2 = None
        neg_default_1: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_default_2: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, neg_default_1, 3, 32, 9223372036854775807);  neg_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_default_3: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_40, slice_tensor_3, 3, 0, 32);  full_default_40 = slice_tensor_3 = None
        add_tensor_2: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default_2, slice_scatter_default_3);  slice_scatter_default_2 = slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_3: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_189, unsqueeze_14);  getitem_189 = unsqueeze_14 = None
        add_tensor_3: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3);  add_tensor_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_1: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(squeeze_dim, [0, 2, 1, 3]);  squeeze_dim = None
        clone_default: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_default, [4, 512, 512]);  clone_default = None
        reshape_default_3: "bf16[2048, 512]" = torch.ops.aten.reshape.default(reshape_default_2, [2048, 512]);  reshape_default_2 = None
        permute_default_2: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_default_3: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_4: "bf16[4, 512, 8, 64]" = torch.ops.aten.permute.default(add_tensor_1, [0, 2, 1, 3]);  add_tensor_1 = None
        clone_default_1: "bf16[4, 512, 8, 64]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        reshape_default_4: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(clone_default_1, [4, 512, 512]);  clone_default_1 = None
        reshape_default_5: "bf16[2048, 512]" = torch.ops.aten.reshape.default(reshape_default_4, [2048, 512]);  reshape_default_4 = None
        permute_default_5: "bf16[2048, 512]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_6: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_7: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_tensor_3, [0, 2, 1, 3]);  add_tensor_3 = None
        clone_default_2: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_6: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_default_2, [4, 512, 2048]);  clone_default_2 = None
        reshape_default_7: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_6, [2048, 2048]);  reshape_default_6 = None
        permute_default_8: "bf16[2048, 2048]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_default_9: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (permute_default, reshape_default_3, permute_default_3, reshape_default_5, permute_default_6, reshape_default_7, permute_default_9)


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
