"""
Standalone repro captured via capture_hook.
Label: torchbench_moondream_infer
Pattern hash: 2b75aa851a35
Shape hash: 20a51b25
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
_shapes_config = "(T([512, 2048], f16), T([16], f16), T([1, 512], i64), T([512, 2048], f16), T([512, 2048], f16), T([1, 512], i64, gen=Index(1)), T([1, 512, 2048], f16), T([8192, 2048], f16), S([1, 512, 2048]), S([1, 512, -1, 64]), S([1, 16, 1]), S([1, 1, 512]), S([1, 512, 2, 16]), S([1, 512, 32]), S([1, 512, 2048]), S([1, 512, -1, 64]), S([1, 512, 2048]), S([1, 512, -1, 64]), S([1, -1, 512, 512]), S([512, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, addmm: "f16[512, 2048]", arg2_1: "f16[16]", unsqueeze: "i64[1, 512]", addmm_1: "f16[512, 2048]", addmm_2: "f16[512, 2048]", cumsum: "i64[1, 512]", convert_element_type_5: "f16[1, 512, 2048]", arg13_1: "f16[8192, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        reshape_default_1: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_tensor: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:87 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_default: "f16[1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_default_1: "f16[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        convert_element_type_default: "f32[1, 16, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float32);  unsqueeze_default_1 = None
        expand_default: "f32[1, 16, 1]" = torch.ops.aten.expand.default(convert_element_type_default, [1, -1, 1]);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:92 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_1: "f32[1, 16, 1]" = torch.ops.aten.expand.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:88 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_default_2: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        convert_element_type_default_1: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_2, torch.float32);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:92 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_2: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_default_1, _shape_param_3);  convert_element_type_default_1 = _shape_param_3 = None
        mul_tensor: "f32[1, 16, 512]" = torch.ops.aten.mul.Tensor(expand_default_1, expand_default_2);  expand_default_1 = expand_default_2 = None
        permute_default_1: "f32[1, 512, 16]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:93 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_default_3: "f32[1, 512, 1, 16]" = torch.ops.aten.unsqueeze.default(permute_default_1, 2);  permute_default_1 = None
        expand_default_3: "f32[1, 512, 2, 16]" = torch.ops.aten.expand.default(unsqueeze_default_3, _shape_param_4);  unsqueeze_default_3 = _shape_param_4 = None
        clone_default: "f32[1, 512, 2, 16]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        reshape_default_2: "f32[1, 512, 32]" = torch.ops.aten.reshape.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:94 in forward, code: cos = emb.cos() * self.attention_scaling
        cos_default: "f32[1, 512, 32]" = torch.ops.aten.cos.default(reshape_default_2)
        mul_tensor_1: "f32[1, 512, 32]" = torch.ops.aten.mul.Tensor(cos_default, 1.0);  cos_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:97 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_2: "f16[1, 512, 32]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default_4: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 1);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_2: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_tensor, unsqueeze_default_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_1: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_tensor_1);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_2: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 16);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_default, slice_tensor_2], -1);  neg_default = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:95 in forward, code: sin = emb.sin() * self.attention_scaling
        sin_default: "f32[1, 512, 32]" = torch.ops.aten.sin.default(reshape_default_2);  reshape_default_2 = None
        mul_tensor_3: "f32[1, 512, 32]" = torch.ops.aten.mul.Tensor(sin_default, 1.0);  sin_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:97 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_3: "f16[1, 512, 32]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.float16);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_5: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, 1);  convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_4: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_5);  cat_default = None
        add_tensor: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_tensor_3: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 32, 9223372036854775807);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_default_1: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_tensor, slice_tensor_3], -1);  add_tensor = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_3: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_6);  addmm_1 = _shape_param_6 = None
        reshape_default_4: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_7);  reshape_default_3 = _shape_param_7 = None
        permute_default_2: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_tensor_4: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_5: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_tensor_4, unsqueeze_default_4);  unsqueeze_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_5: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_tensor_5);  slice_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_6: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 0, 16);  slice_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default_2: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_6], -1);  neg_default_1 = slice_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_6: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_default_2, unsqueeze_default_5);  cat_default_2 = unsqueeze_default_5 = None
        add_tensor_1: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_tensor_7: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 32, 9223372036854775807);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_default_3: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_tensor_1, slice_tensor_7], -1);  add_tensor_1 = slice_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_5: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_8);  addmm_2 = _shape_param_8 = None
        reshape_default_6: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(reshape_default_5, _shape_param_9);  reshape_default_5 = _shape_param_9 = None
        permute_default_3: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_2: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_default_6: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_2, 0);  add_tensor_2 = None
        unsqueeze_default_7: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 1);  unsqueeze_default_6 = None
        unsqueeze_default_8: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_default_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_3: "i64[512]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_default_9: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_3, 0);  add_tensor_3 = None
        unsqueeze_default_10: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 1);  unsqueeze_default_9 = None
        unsqueeze_default_11: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_default_8, unsqueeze_default_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_default_2: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_default_12: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, 1);  iota_default_2 = None
        unsqueeze_default_13: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index_tensor: "i64[1, 1, 512, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_14, unsqueeze_default_11]);  unsqueeze_default_11 = None
        index_tensor_1: "i64[1, 1, 1, 512]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_14, unsqueeze_default_8]);  cumsum = unsqueeze_default_14 = unsqueeze_default_8 = None
        eq_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.eq.Tensor(index_tensor, index_tensor_1);  index_tensor = index_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_tensor_1: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_default_4: "b8[1, 1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_tensor_1, _shape_param_10);  bitwise_and_tensor_1 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default_4, full_default_1, full_default_2);  expand_default_4 = full_default_1 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        reshape_default_7: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_5, _shape_param_11);  convert_element_type_5 = _shape_param_11 = None
        permute_default_4: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        return (cat_default_1, cat_default_3, permute_default_3, where_self, reshape_default_7, permute_default_4)



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
