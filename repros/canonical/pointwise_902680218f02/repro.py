"""
Standalone repro captured via capture_hook.
Label: hf_pythia_410m_train
Pattern hash: 902680218f02
Shape hash: 93443a52
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_3: "f16[2048, 4096]", _shape_param_0, addmm_94: "f16[2048, 4096]", _shape_param_1, _shape_param_2, primals_288: "f16[4096, 1024]", tangents_48: "f16[4, 16, 512, 64]", getitem_387: "f16[4, 16, 512, 64]", tangents_47: "f16[4, 16, 512, 64]", getitem_388: "f16[4, 16, 512, 64]", getitem_386: "f16[4, 16, 512, 64]", primals_3: "f32[8]", _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, primals_282: "f16[3072, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        reshape_default: "f16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_3, _shape_param_0);  mm_3 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_default: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        reshape_default_1: "f16[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_94, _shape_param_1);  addmm_94 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_default_1: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        mul_tensor: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.7071067811865476)
        erf_default: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_tensor);  mul_tensor = None
        add_tensor: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_1: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_2: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default_1)
        mul_tensor_3: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_2, -0.5);  mul_tensor_2 = None
        exp_default: "f32[4, 512, 4096]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_5: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, mul_tensor_4);  convert_element_type_default_1 = mul_tensor_4 = None
        add_tensor_1: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_1 = mul_tensor_5 = None
        mul_tensor_6: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, add_tensor_1);  convert_element_type_default = add_tensor_1 = None
        convert_element_type_default_2: "f16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.float16);  mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        reshape_default_2: "f16[2048, 4096]" = torch.ops.aten.reshape.default(convert_element_type_default_2, _shape_param_2);  convert_element_type_default_2 = _shape_param_2 = None
        permute_default: "f16[1024, 4096]" = torch.ops.aten.permute.default(primals_288, [1, 0]);  primals_288 = None
        permute_default_1: "f16[4096, 1024]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        add_tensor_2: "f16[4, 16, 512, 64]" = torch.ops.aten.add.Tensor(tangents_48, getitem_387);  tangents_48 = getitem_387 = None
        add_tensor_3: "f16[4, 16, 512, 64]" = torch.ops.aten.add.Tensor(tangents_47, getitem_388);  tangents_47 = getitem_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:158 in apply_rotary_pos_emb, code: k_embed = torch.cat([k_embed, k_pass], dim=-1)
        slice_tensor: "f16[4, 16, 512, 16]" = torch.ops.aten.slice.Tensor(add_tensor_2, 3, 0, 16)
        slice_tensor_1: "f16[4, 16, 512, 48]" = torch.ops.aten.slice.Tensor(add_tensor_2, 3, 16, 64);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:157 in apply_rotary_pos_emb, code: q_embed = torch.cat([q_embed, q_pass], dim=-1)
        slice_tensor_2: "f16[4, 16, 512, 16]" = torch.ops.aten.slice.Tensor(getitem_386, 3, 0, 16)
        slice_tensor_3: "f16[4, 16, 512, 48]" = torch.ops.aten.slice.Tensor(getitem_386, 3, 16, 64);  getitem_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:352 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_4: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:353 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_4, 0);  add_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_default_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:106 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_default_2: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
        unsqueeze_default_3: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 2);  unsqueeze_default_2 = None
        expand_default: "f32[1, 8, 1]" = torch.ops.aten.expand.default(unsqueeze_default_3, [1, -1, 1]);  unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:107 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        convert_element_type_default_3: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float32);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:111 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_1: "f32[1, 8, 1]" = torch.ops.aten.expand.default(expand_default, _shape_param_3);  expand_default = _shape_param_3 = None
        expand_default_2: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_default_3, _shape_param_4);  convert_element_type_default_3 = _shape_param_4 = None
        mul_tensor_7: "f32[1, 8, 512]" = torch.ops.aten.mul.Tensor(expand_default_1, expand_default_2);  expand_default_1 = expand_default_2 = None
        permute_default_2: "f32[1, 512, 8]" = torch.ops.aten.permute.default(mul_tensor_7, [0, 2, 1]);  mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:112 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_default_4: "f32[1, 512, 1, 8]" = torch.ops.aten.unsqueeze.default(permute_default_2, 2);  permute_default_2 = None
        expand_default_3: "f32[1, 512, 2, 8]" = torch.ops.aten.expand.default(unsqueeze_default_4, _shape_param_5);  unsqueeze_default_4 = _shape_param_5 = None
        clone_default: "f32[1, 512, 2, 8]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        reshape_default_3: "f32[1, 512, 16]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:114 in forward, code: sin = emb.sin() * self.attention_scaling
        sin_default: "f32[1, 512, 16]" = torch.ops.aten.sin.default(reshape_default_3)
        mul_tensor_8: "f32[1, 512, 16]" = torch.ops.aten.mul.Tensor(sin_default, 1.0);  sin_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:116 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_4: "f16[1, 512, 16]" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.float16);  mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:145 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_5: "f16[1, 1, 512, 16]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_4, 1);  convert_element_type_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:154 in apply_rotary_pos_emb, code: k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        mul_tensor_9: "f16[4, 16, 512, 16]" = torch.ops.aten.mul.Tensor(slice_tensor, unsqueeze_default_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:123 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_tensor_4: "f16[4, 16, 512, 8]" = torch.ops.aten.slice.Tensor(mul_tensor_9, 3, 0, 8)
        slice_tensor_5: "f16[4, 16, 512, 8]" = torch.ops.aten.slice.Tensor(mul_tensor_9, 3, 8, 16);  mul_tensor_9 = None
        neg_default: "f16[4, 16, 512, 8]" = torch.ops.aten.neg.default(slice_tensor_4);  slice_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:122 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default: "f16[4, 16, 512, 16]" = torch.ops.aten.full.default([4, 16, 512, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f16[4, 16, 512, 16]" = torch.ops.aten.slice_scatter.default(full_default, neg_default, 3, 8, 9223372036854775807);  neg_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:121 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_default_1: "f16[4, 16, 512, 16]" = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_5, 3, 0, 8);  slice_tensor_5 = None
        add_tensor_5: "f16[4, 16, 512, 16]" = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1);  slice_scatter_default = slice_scatter_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:113 in forward, code: cos = emb.cos() * self.attention_scaling
        cos_default: "f32[1, 512, 16]" = torch.ops.aten.cos.default(reshape_default_3);  reshape_default_3 = None
        mul_tensor_10: "f32[1, 512, 16]" = torch.ops.aten.mul.Tensor(cos_default, 1.0);  cos_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:116 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_5: "f16[1, 512, 16]" = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.float16);  mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:144 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default_6: "f16[1, 1, 512, 16]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_5, 1);  convert_element_type_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:154 in apply_rotary_pos_emb, code: k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        mul_tensor_11: "f16[4, 16, 512, 16]" = torch.ops.aten.mul.Tensor(slice_tensor, unsqueeze_default_6);  slice_tensor = None
        add_tensor_6: "f16[4, 16, 512, 16]" = torch.ops.aten.add.Tensor(add_tensor_5, mul_tensor_11);  add_tensor_5 = mul_tensor_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:153 in apply_rotary_pos_emb, code: q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        mul_tensor_12: "f16[4, 16, 512, 16]" = torch.ops.aten.mul.Tensor(slice_tensor_2, unsqueeze_default_5);  unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:123 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_tensor_6: "f16[4, 16, 512, 8]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 3, 0, 8)
        slice_tensor_7: "f16[4, 16, 512, 8]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 3, 8, 16);  mul_tensor_12 = None
        neg_default_1: "f16[4, 16, 512, 8]" = torch.ops.aten.neg.default(slice_tensor_6);  slice_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:122 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_default_2: "f16[4, 16, 512, 16]" = torch.ops.aten.slice_scatter.default(full_default, neg_default_1, 3, 8, 9223372036854775807);  neg_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:121 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_default_3: "f16[4, 16, 512, 16]" = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_7, 3, 0, 8);  full_default = slice_tensor_7 = None
        add_tensor_7: "f16[4, 16, 512, 16]" = torch.ops.aten.add.Tensor(slice_scatter_default_2, slice_scatter_default_3);  slice_scatter_default_2 = slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:153 in apply_rotary_pos_emb, code: q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        mul_tensor_13: "f16[4, 16, 512, 16]" = torch.ops.aten.mul.Tensor(slice_tensor_2, unsqueeze_default_6);  slice_tensor_2 = unsqueeze_default_6 = None
        add_tensor_8: "f16[4, 16, 512, 16]" = torch.ops.aten.add.Tensor(add_tensor_7, mul_tensor_13);  add_tensor_7 = mul_tensor_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:150 in apply_rotary_pos_emb, code: k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        full_default_1: "f16[4, 16, 512, 64]" = torch.ops.aten.full.default([4, 16, 512, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_4: "f16[4, 16, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_tensor_1, 3, 16, 9223372036854775807);  slice_tensor_1 = None
        slice_scatter_default_5: "f16[4, 16, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_1, add_tensor_6, 3, 0, 16);  add_tensor_6 = None
        add_tensor_9: "f16[4, 16, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default_4, slice_scatter_default_5);  slice_scatter_default_4 = slice_scatter_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:149 in apply_rotary_pos_emb, code: q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        slice_scatter_default_6: "f16[4, 16, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_tensor_3, 3, 16, 9223372036854775807);  slice_tensor_3 = None
        slice_scatter_default_7: "f16[4, 16, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_1, add_tensor_8, 3, 0, 16);  full_default_1 = add_tensor_8 = None
        add_tensor_10: "f16[4, 16, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default_6, slice_scatter_default_7);  slice_scatter_default_6 = slice_scatter_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:215 in forward, code: query_states, key_states, value_states = qkv.chunk(3, dim=-1)
        cat_default: "f16[4, 16, 512, 192]" = torch.ops.aten.cat.default([add_tensor_10, add_tensor_9, add_tensor_3], 3);  add_tensor_10 = add_tensor_9 = add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_3: "f16[4, 512, 16, 192]" = torch.ops.aten.permute.default(cat_default, [0, 2, 1, 3]);  cat_default = None
        clone_default_1: "f16[4, 512, 16, 192]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        reshape_default_4: "f16[4, 512, 3072]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None
        reshape_default_5: "f16[2048, 3072]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_8);  reshape_default_4 = _shape_param_8 = None
        permute_default_4: "f16[1024, 3072]" = torch.ops.aten.permute.default(primals_282, [1, 0]);  primals_282 = None
        permute_default_5: "f16[3072, 1024]" = torch.ops.aten.permute.default(permute_default_4, [1, 0]);  permute_default_4 = None
        return (reshape_default_2, permute_default_1, reshape_default_5, permute_default_5)


def _default_make_inputs():
    return [
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4, 512, 4096],  # _shape_param_0
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4, 512, 4096],  # _shape_param_1
    [2048, 4096],  # _shape_param_2
    torch.randn([4096, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16, 512, 64], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16, 512, 64], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16, 512, 64], dtype=torch.float16, device='cuda'),
    torch.randn(2097152, dtype=torch.float16, device='cuda').as_strided([4, 16, 512, 64], [524288, 64, 1024, 1]),  # getitem_388
    torch.randn([4, 16, 512, 64], dtype=torch.float16, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    [1, 8, 1],  # _shape_param_3
    [1, 1, 512],  # _shape_param_4
    [1, 512, 2, 8],  # _shape_param_5
    [1, 512, 16],  # _shape_param_6
    [4, 512, 3072],  # _shape_param_7
    [2048, 3072],  # _shape_param_8
    torch.randn([3072, 1024], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
