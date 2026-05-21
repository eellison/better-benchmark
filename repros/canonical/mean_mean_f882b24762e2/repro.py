"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=mean, ranges=['4', '512', '16', '1'], reduction_ranges=[]
#   origins: ['aten.mean.dim']
#   type=mean, ranges=['4', '512', '8', '1'], reduction_ranges=[]
#   origins: ['aten.mean.dim']
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
_shapes_config = "(T([2048, 2048], bf16), T([128], bf16), T([64], f32), T([1, 512], i64, gen=Index(100)), T([2048, 1024], bf16), T([128], bf16), T([2048, 1024], bf16), T([4, 512], i64, gen=Index(100)))"

class Repro(torch.nn.Module):
    def forward(self, mm: "bf16[2048, 2048]", arg5_1: "bf16[128]", arg2_1: "f32[64]", unsqueeze: "i64[1, 512]", mm_1: "bf16[2048, 1024]", arg7_1: "bf16[128]", mm_2: "bf16[2048, 1024]", cumsum: "i64[4, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm, [4, 512, 2048]);  mm = None
        reshape_default_1: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(reshape_default, [4, 512, -1, 128]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg5_1, convert_element_type_default_1);  arg5_1 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_tensor_1, [0, 2, 1, 3]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:138 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_default: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_default_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        expand_default: "f32[1, 64, 1]" = torch.ops.aten.expand.default(unsqueeze_default_1, [1, -1, 1]);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:143 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_1: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand_default, [1, 64, 1]);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:139 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_default_2: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        convert_element_type_default_2: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_2, torch.float32);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:143 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_2: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_default_2, [1, 1, 512]);  convert_element_type_default_2 = None
        mul_tensor_2: "f32[1, 64, 512]" = torch.ops.aten.mul.Tensor(expand_default_1, expand_default_2);  expand_default_1 = expand_default_2 = None
        permute_default_1: "f32[1, 512, 64]" = torch.ops.aten.permute.default(mul_tensor_2, [0, 2, 1]);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:144 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_default_3: "f32[1, 512, 1, 64]" = torch.ops.aten.unsqueeze.default(permute_default_1, 2);  permute_default_1 = None
        expand_default_3: "f32[1, 512, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_default_3, [1, 512, 2, 64]);  unsqueeze_default_3 = None
        clone_default: "f32[1, 512, 2, 64]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        reshape_default_2: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(clone_default, [1, 512, 128]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:145 in forward, code: cos = emb.cos() * self.attention_scaling
        cos_default: "f32[1, 512, 128]" = torch.ops.aten.cos.default(reshape_default_2)
        mul_tensor_3: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(cos_default, 1.0);  cos_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:148 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_3: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default_4: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, 1);  convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_4: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_1: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 64);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:146 in forward, code: sin = emb.sin() * self.attention_scaling
        sin_default: "f32[1, 512, 128]" = torch.ops.aten.sin.default(reshape_default_2);  reshape_default_2 = None
        mul_tensor_5: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sin_default, 1.0);  sin_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:148 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_4: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_5: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_4, 1);  convert_element_type_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_6: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_5);  cat_default = None
        add_tensor_1: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default_3: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_1, [4, 512, 1024]);  mm_1 = None
        reshape_default_4: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(reshape_default_3, [4, 512, -1, 128]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_5: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(reshape_default_4, torch.float32);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar_1: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_5, 2)
        mean_dim_1: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True);  pow_tensor_scalar_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_2: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06);  mean_dim_1 = None
        rsqrt_default_1: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_7: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, rsqrt_default_1);  convert_element_type_default_5 = rsqrt_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_6: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        mul_tensor_8: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg7_1, convert_element_type_default_6);  arg7_1 = convert_element_type_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_2: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_tensor_8, [0, 2, 1, 3]);  mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_9: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default_2, unsqueeze_default_4);  unsqueeze_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_2: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_3: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 0, 64);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_10: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_5);  cat_default_1 = unsqueeze_default_5 = None
        add_tensor_3: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_9, mul_tensor_10);  mul_tensor_9 = mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_6: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_tensor_3, 2);  add_tensor_3 = None
        expand_default_4: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_default_6, [4, 8, 2, 512, 128]);  unsqueeze_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default_1: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_default_4, memory_format = torch.contiguous_format);  expand_default_4 = None
        reshape_default_5: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_default_1, [4, 16, 512, 128]);  clone_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_6: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_2, [4, 512, 1024]);  mm_2 = None
        reshape_default_7: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(reshape_default_6, [4, 512, -1, 128]);  reshape_default_6 = None
        permute_default_3: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(reshape_default_7, [0, 2, 1, 3]);  reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_7: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_default_3, 2);  permute_default_3 = None
        expand_default_5: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_default_7, [4, 8, 2, 512, 128]);  unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default_2: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_default_5, memory_format = torch.contiguous_format);  expand_default_5 = None
        reshape_default_8: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_default_2, [4, 16, 512, 128]);  clone_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_4: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_default_8: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_4, 0);  add_tensor_4 = None
        unsqueeze_default_9: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 1);  unsqueeze_default_8 = None
        unsqueeze_default_10: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_default_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_5: "i64[512]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_default_11: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_5, 0);  add_tensor_5 = None
        unsqueeze_default_12: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 1);  unsqueeze_default_11 = None
        unsqueeze_default_13: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 3);  unsqueeze_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_default_10, unsqueeze_default_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_default_2: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_default_14: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, 1);  iota_default_2 = None
        unsqueeze_default_15: "i64[4, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 2);  unsqueeze_default_14 = None
        unsqueeze_default_16: "i64[4, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 3);  unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index_tensor: "i64[4, 1, 512, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_16, unsqueeze_default_13]);  unsqueeze_default_13 = None
        index_tensor_1: "i64[4, 1, 1, 512]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_16, unsqueeze_default_10]);  cumsum = unsqueeze_default_16 = unsqueeze_default_10 = None
        eq_tensor: "b8[4, 1, 512, 512]" = torch.ops.aten.eq.Tensor(index_tensor, index_tensor_1);  index_tensor = index_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_tensor_1: "b8[4, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_default_6: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_tensor_1, [4, -1, 512, 512]);  bitwise_and_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default_6, full_default_1, full_default_2);  expand_default_6 = full_default_1 = full_default_2 = None
        return (add_tensor_1, reshape_default_5, reshape_default_8, where_self)


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
