"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['4', '8', '1', '512', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4', '8', '1', '512', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4', '512', '8', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4', '512', '16', '1'], reduction_ranges=[]
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

class Repro(torch.nn.Module):
    def forward(self, getitem_254: "bf16[4, 16, 512, 128]", getitem_253: "bf16[4, 16, 512, 128]", primals_3: "f32[64]", getitem_252: "bf16[4, 16, 512, 128]", primals_306: "bf16[1024, 1024]", primals_305: "bf16[128]", mm_190: "bf16[2048, 1024]", rsqrt_110: "f32[4, 512, 8, 1]", primals_304: "bf16[1024, 1024]", primals_303: "bf16[128]", mm_189: "bf16[2048, 2048]", rsqrt_109: "f32[4, 512, 16, 1]", primals_302: "bf16[2048, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        reshape_default: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.reshape.default(getitem_254, [4, 8, 2, 512, 128]);  getitem_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_dim_int_list: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(reshape_default, [2], True);  reshape_default = None
        squeeze_dim: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 2);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        reshape_default_1: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.reshape.default(getitem_253, [4, 8, 2, 512, 128]);  getitem_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_dim_int_list_1: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [2], True);  reshape_default_1 = None
        squeeze_dim_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 2);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:399 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:400 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_default_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:138 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_default_2: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
        unsqueeze_default_3: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 2);  unsqueeze_default_2 = None
        expand_default: "f32[1, 64, 1]" = torch.ops.aten.expand.default(unsqueeze_default_3, [1, -1, 1]);  unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:139 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        convert_element_type_default: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float32);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:143 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_1: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand_default, [1, 64, 1]);  expand_default = None
        expand_default_2: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_default, [1, 1, 512]);  convert_element_type_default = None
        mul_tensor: "f32[1, 64, 512]" = torch.ops.aten.mul.Tensor(expand_default_1, expand_default_2);  expand_default_1 = expand_default_2 = None
        permute_default: "f32[1, 512, 64]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:144 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_default_4: "f32[1, 512, 1, 64]" = torch.ops.aten.unsqueeze.default(permute_default, 2);  permute_default = None
        expand_default_3: "f32[1, 512, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_default_4, [1, 512, 2, 64]);  unsqueeze_default_4 = None
        clone_default: "f32[1, 512, 2, 64]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        reshape_default_2: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(clone_default, [1, 512, 128]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:146 in forward, code: sin = emb.sin() * self.attention_scaling
        sin_default: "f32[1, 512, 128]" = torch.ops.aten.sin.default(reshape_default_2)
        mul_tensor_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sin_default, 1.0);  sin_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:148 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_1: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_5: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 1);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_2: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_dim_1, unsqueeze_default_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_tensor: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_tensor_2, 3, 0, 64)
        slice_tensor_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_tensor_2, 3, 64, 128);  mul_tensor_2 = None
        neg_default: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default: "bf16[4, 8, 512, 128]" = torch.ops.aten.full.default([4, 8, 512, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default, neg_default, 3, 64, 9223372036854775807);  neg_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_default_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_1, 3, 0, 64);  full_default = slice_tensor_1 = None
        add_tensor_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1);  slice_scatter_default = slice_scatter_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:145 in forward, code: cos = emb.cos() * self.attention_scaling
        cos_default: "f32[1, 512, 128]" = torch.ops.aten.cos.default(reshape_default_2);  reshape_default_2 = None
        mul_tensor_3: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(cos_default, 1.0);  cos_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:148 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_2: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default_6: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 1);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_4: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_dim_1, unsqueeze_default_6);  squeeze_dim_1 = None
        add_tensor_2: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_4);  add_tensor_1 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_5: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_252, unsqueeze_default_5);  unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_tensor_2: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(mul_tensor_5, 3, 0, 64)
        slice_tensor_3: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(mul_tensor_5, 3, 64, 128);  mul_tensor_5 = None
        neg_default_1: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default_1: "bf16[4, 16, 512, 128]" = torch.ops.aten.full.default([4, 16, 512, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_2: "bf16[4, 16, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_1, neg_default_1, 3, 64, 9223372036854775807);  neg_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_default_3: "bf16[4, 16, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_tensor_3, 3, 0, 64);  full_default_1 = slice_tensor_3 = None
        add_tensor_3: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_default_2, slice_scatter_default_3);  slice_scatter_default_2 = slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_6: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_252, unsqueeze_default_6);  getitem_252 = unsqueeze_default_6 = None
        add_tensor_4: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_6);  add_tensor_3 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_1: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(squeeze_dim, [0, 2, 1, 3]);  squeeze_dim = None
        clone_default_1: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_default_1, [4, 512, 1024]);  clone_default_1 = None
        reshape_default_4: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(reshape_default_3, [2048, 1024]);  reshape_default_3 = None
        permute_default_2: "bf16[1024, 1024]" = torch.ops.aten.permute.default(primals_306, [1, 0]);  primals_306 = None
        permute_default_3: "bf16[1024, 1024]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_4: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(add_tensor_2, [0, 2, 1, 3]);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor_7: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_default_4, primals_305);  permute_default_4 = primals_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default_5: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_190, [4, 512, 1024]);  mm_190 = None
        reshape_default_6: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(reshape_default_5, [4, 512, -1, 128]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_3: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(reshape_default_6, torch.float32);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_4: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.float32);  mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_8: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, convert_element_type_default_3)
        mul_tensor_9: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, rsqrt_110);  convert_element_type_default_4 = None
        sum_dim_int_list_2: "f32[4, 512, 8, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [3], True);  mul_tensor_8 = None
        pow_tensor_scalar: "f32[4, 512, 8, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_110, 3);  rsqrt_110 = None
        mul_scalar: "f32[4, 512, 8, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_2, -0.5);  sum_dim_int_list_2 = None
        mul_tensor_10: "f32[4, 512, 8, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default_4: "f32[4, 512, 8, 128]" = torch.ops.aten.expand.default(mul_tensor_10, [4, 512, 8, 128]);  mul_tensor_10 = None
        div_scalar: "f32[4, 512, 8, 128]" = torch.ops.aten.div.Scalar(expand_default_4, 128);  expand_default_4 = None
        pow_tensor_scalar_1: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_3, 1.0);  convert_element_type_default_3 = None
        mul_scalar_1: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_11: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_5: "f32[4, 512, 8, 128]" = torch.ops.aten.add.Tensor(mul_tensor_9, mul_tensor_11);  mul_tensor_9 = mul_tensor_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_5: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16);  add_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        clone_default_2: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(convert_element_type_default_5, memory_format = torch.contiguous_format);  convert_element_type_default_5 = None
        reshape_default_7: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_default_2, [4, 512, 1024]);  clone_default_2 = None
        reshape_default_8: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(reshape_default_7, [2048, 1024]);  reshape_default_7 = None
        permute_default_5: "bf16[1024, 1024]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
        permute_default_6: "bf16[1024, 1024]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_7: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(add_tensor_4, [0, 2, 1, 3]);  add_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor_12: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_default_7, primals_303);  permute_default_7 = primals_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default_9: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_189, [4, 512, 2048]);  mm_189 = None
        reshape_default_10: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(reshape_default_9, [4, 512, -1, 128]);  reshape_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_6: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(reshape_default_10, torch.float32);  reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_7: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.float32);  mul_tensor_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_13: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, convert_element_type_default_6)
        mul_tensor_14: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, rsqrt_109);  convert_element_type_default_7 = None
        sum_dim_int_list_3: "f32[4, 512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [3], True);  mul_tensor_13 = None
        pow_tensor_scalar_2: "f32[4, 512, 16, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_109, 3);  rsqrt_109 = None
        mul_scalar_2: "f32[4, 512, 16, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_3, -0.5);  sum_dim_int_list_3 = None
        mul_tensor_15: "f32[4, 512, 16, 1]" = torch.ops.aten.mul.Tensor(mul_scalar_2, pow_tensor_scalar_2);  mul_scalar_2 = pow_tensor_scalar_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default_5: "f32[4, 512, 16, 128]" = torch.ops.aten.expand.default(mul_tensor_15, [4, 512, 16, 128]);  mul_tensor_15 = None
        div_scalar_1: "f32[4, 512, 16, 128]" = torch.ops.aten.div.Scalar(expand_default_5, 128);  expand_default_5 = None
        pow_tensor_scalar_3: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_6, 1.0);  convert_element_type_default_6 = None
        mul_scalar_3: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_3, 2.0);  pow_tensor_scalar_3 = None
        mul_tensor_16: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_3);  div_scalar_1 = mul_scalar_3 = None
        add_tensor_6: "f32[4, 512, 16, 128]" = torch.ops.aten.add.Tensor(mul_tensor_14, mul_tensor_16);  mul_tensor_14 = mul_tensor_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_8: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.bfloat16);  add_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        clone_default_3: "bf16[4, 512, 16, 128]" = torch.ops.aten.clone.default(convert_element_type_default_8, memory_format = torch.contiguous_format);  convert_element_type_default_8 = None
        reshape_default_11: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(clone_default_3, [4, 512, 2048]);  clone_default_3 = None
        reshape_default_12: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(reshape_default_11, [2048, 2048]);  reshape_default_11 = None
        permute_default_8: "bf16[1024, 2048]" = torch.ops.aten.permute.default(primals_302, [1, 0]);  primals_302 = None
        permute_default_9: "bf16[2048, 1024]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (reshape_default_4, permute_default_3, reshape_default_8, permute_default_6, reshape_default_12, permute_default_9)


def _default_make_inputs():
    return [
    torch.randn([4, 16, 512, 128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 16, 512, 128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn(4194304, dtype=torch.bfloat16, device='cuda').as_strided([4, 16, 512, 128], [1048576, 128, 2048, 1]),  # getitem_252
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 8, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
