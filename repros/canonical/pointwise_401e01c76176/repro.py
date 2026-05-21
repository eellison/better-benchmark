"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

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
_shapes_config = "(T([2048, 2048], bf16), T([1, 512, 64], bf16), T([1, 512, 64], bf16), T([2048, 512], bf16), T([2048, 512], bf16), T([4, 1, 512, 512], b8))"

class Repro(torch.nn.Module):
    def forward(self, mm_105: "bf16[2048, 2048]", convert_element_type_1: "bf16[1, 512, 64]", convert_element_type_2: "bf16[1, 512, 64]", mm_106: "bf16[2048, 512]", mm_107: "bf16[2048, 512]", expand_1: "b8[4, 1, 512, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_105, [4, 512, 2048]);  mm_105 = None
        reshape_default_1: "bf16[4, 512, 32, 64]" = torch.ops.aten.reshape.default(reshape_default, [4, 512, -1, 64]);  reshape_default = None
        permute_default: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_1: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 32);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_1: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_1: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = None
        add_tensor: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_106, [4, 512, 512]);  mm_106 = None
        reshape_default_3: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_2, [4, 512, -1, 64]);  reshape_default_2 = None
        permute_default_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_2: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_default_1, unsqueeze_default);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_2: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_3: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 32);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_3: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_1);  cat_default_1 = unsqueeze_default_1 = None
        add_tensor_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_2: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 2);  add_tensor_1 = None
        expand_default: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_default_2, [4, 8, 4, 512, 64]);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_4: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_default, [4, 32, 512, 64]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:264 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_5: "bf16[4, 512, 512]" = torch.ops.aten.reshape.default(mm_107, [4, 512, 512]);  mm_107 = None
        reshape_default_6: "bf16[4, 512, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_5, [4, 512, -1, 64]);  reshape_default_5 = None
        permute_default_2: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_3: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_default_2, 2);  permute_default_2 = None
        expand_default_1: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_default_3, [4, 8, 4, 512, 64]);  unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default_1: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_7: "bf16[4, 32, 512, 64]" = torch.ops.aten.reshape.default(clone_default_1, [4, 32, 512, 64]);  clone_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default, full_default_1);  expand_1 = full_default = full_default_1 = None
        return (add_tensor, reshape_default_4, reshape_default_7, where_self)


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
