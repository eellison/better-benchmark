"""
Standalone repro captured via capture_hook.
Label: hf_pythia_410m_train
Pattern hash: 21f01203c8be
Shape hash: a5f0f69e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, _shape_param_0, primals_3: "f32[8]", _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, getitem_2: "f16[4, 16, 512, 64]", getitem_3: "f16[4, 16, 512, 64]", getitem_4: "f16[4, 16, 512, 64]", mul_3: "f32[4, 512, 1024]", primals_10: "f16[1024]", primals_11: "f16[1024]", _shape_param_5, primals_12: "f16[4096, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:352 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:353 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_default_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_default_3: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_default_3, unsqueeze_default_2);  unsqueeze_default_3 = unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_default: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(le_tensor, _shape_param_0);  le_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:106 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_default_4: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
        unsqueeze_default_5: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        expand_default_1: "f32[1, 8, 1]" = torch.ops.aten.expand.default(unsqueeze_default_5, [1, -1, 1]);  unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:107 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        convert_element_type_default: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float32);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:111 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_2: "f32[1, 8, 1]" = torch.ops.aten.expand.default(expand_default_1, _shape_param_1);  expand_default_1 = _shape_param_1 = None
        expand_default_3: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_default, _shape_param_2);  convert_element_type_default = _shape_param_2 = None
        mul_tensor: "f32[1, 8, 512]" = torch.ops.aten.mul.Tensor(expand_default_2, expand_default_3);  expand_default_2 = expand_default_3 = None
        permute_default: "f32[1, 512, 8]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:112 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_default_6: "f32[1, 512, 1, 8]" = torch.ops.aten.unsqueeze.default(permute_default, 2);  permute_default = None
        expand_default_4: "f32[1, 512, 2, 8]" = torch.ops.aten.expand.default(unsqueeze_default_6, _shape_param_3);  unsqueeze_default_6 = _shape_param_3 = None
        clone_default: "f32[1, 512, 2, 8]" = torch.ops.aten.clone.default(expand_default_4, memory_format = torch.contiguous_format);  expand_default_4 = None
        reshape_default: "f32[1, 512, 16]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:113 in forward, code: cos = emb.cos() * self.attention_scaling
        cos_default: "f32[1, 512, 16]" = torch.ops.aten.cos.default(reshape_default)
        mul_tensor_1: "f32[1, 512, 16]" = torch.ops.aten.mul.Tensor(cos_default, 1.0);  cos_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:114 in forward, code: sin = emb.sin() * self.attention_scaling
        sin_default: "f32[1, 512, 16]" = torch.ops.aten.sin.default(reshape_default);  reshape_default = None
        mul_tensor_2: "f32[1, 512, 16]" = torch.ops.aten.mul.Tensor(sin_default, 1.0);  sin_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:116 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_1: "f16[1, 512, 16]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None
        convert_element_type_default_2: "f16[1, 512, 16]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float16);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:144 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default_7: "f16[1, 1, 512, 16]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 1);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:145 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_8: "f16[1, 1, 512, 16]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 1);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:149 in apply_rotary_pos_emb, code: q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        slice_tensor: "f16[4, 16, 512, 16]" = torch.ops.aten.slice.Tensor(getitem_2, 3, 0, 16)
        slice_tensor_1: "f16[4, 16, 512, 48]" = torch.ops.aten.slice.Tensor(getitem_2, 3, 16, 9223372036854775807);  getitem_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:150 in apply_rotary_pos_emb, code: k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        slice_tensor_2: "f16[4, 16, 512, 16]" = torch.ops.aten.slice.Tensor(getitem_3, 3, 0, 16)
        slice_tensor_3: "f16[4, 16, 512, 48]" = torch.ops.aten.slice.Tensor(getitem_3, 3, 16, 9223372036854775807);  getitem_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:153 in apply_rotary_pos_emb, code: q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        mul_tensor_3: "f16[4, 16, 512, 16]" = torch.ops.aten.mul.Tensor(slice_tensor, unsqueeze_default_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:121 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_4: "f16[4, 16, 512, 8]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:122 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_5: "f16[4, 16, 512, 8]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 8, 9223372036854775807);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:123 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "f16[4, 16, 512, 8]" = torch.ops.aten.neg.default(slice_tensor_5);  slice_tensor_5 = None
        cat_default: "f16[4, 16, 512, 16]" = torch.ops.aten.cat.default([neg_default, slice_tensor_4], -1);  neg_default = slice_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:153 in apply_rotary_pos_emb, code: q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        mul_tensor_4: "f16[4, 16, 512, 16]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_8);  cat_default = None
        add_tensor_1: "f16[4, 16, 512, 16]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:154 in apply_rotary_pos_emb, code: k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        mul_tensor_5: "f16[4, 16, 512, 16]" = torch.ops.aten.mul.Tensor(slice_tensor_2, unsqueeze_default_7);  unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:121 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_6: "f16[4, 16, 512, 8]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 0, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:122 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_7: "f16[4, 16, 512, 8]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 8, 9223372036854775807);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:123 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "f16[4, 16, 512, 8]" = torch.ops.aten.neg.default(slice_tensor_7);  slice_tensor_7 = None
        cat_default_1: "f16[4, 16, 512, 16]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_6], -1);  neg_default_1 = slice_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:154 in apply_rotary_pos_emb, code: k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        mul_tensor_6: "f16[4, 16, 512, 16]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_8);  cat_default_1 = unsqueeze_default_8 = None
        add_tensor_2: "f16[4, 16, 512, 16]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:157 in apply_rotary_pos_emb, code: q_embed = torch.cat([q_embed, q_pass], dim=-1)
        cat_default_2: "f16[4, 16, 512, 64]" = torch.ops.aten.cat.default([add_tensor_1, slice_tensor_1], -1);  add_tensor_1 = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:158 in apply_rotary_pos_emb, code: k_embed = torch.cat([k_embed, k_pass], dim=-1)
        cat_default_3: "f16[4, 16, 512, 64]" = torch.ops.aten.cat.default([add_tensor_2, slice_tensor_3], -1);  add_tensor_2 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:144 in update, code: self.values = torch.cat([self.values, value_states], dim=-2)
        clone_default_1: "f16[4, 16, 512, 64]" = torch.ops.aten.clone.default(getitem_4);  getitem_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default);  expand_default = full_default_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_7: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_3, primals_10);  mul_3 = primals_10 = None
        add_tensor_3: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_7, primals_11);  mul_tensor_7 = primals_11 = None
        convert_element_type_default_3: "f16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        reshape_default_1: "f16[2048, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default_3, _shape_param_5);  convert_element_type_default_3 = _shape_param_5 = None
        permute_default_1: "f16[1024, 4096]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        return (cat_default_2, cat_default_3, clone_default_1, where_self, reshape_default_1, permute_default_1)


def _default_make_inputs():
    return [
    [4, -1, 512, 512],  # _shape_param_0
    torch.randn([8], dtype=torch.float32, device='cuda'),
    [1, 8, 1],  # _shape_param_1
    [1, 1, 512],  # _shape_param_2
    [1, 512, 2, 8],  # _shape_param_3
    [1, 512, 16],  # _shape_param_4
    torch.randn(6291328, dtype=torch.float16, device='cuda').as_strided([4, 16, 512, 64], [1572864, 192, 3072, 1]),  # getitem_2
    torch.randn(6291328, dtype=torch.float16, device='cuda').as_strided([4, 16, 512, 64], [1572864, 192, 3072, 1]),  # getitem_3
    torch.randn(6291328, dtype=torch.float16, device='cuda').as_strided([4, 16, 512, 64], [1572864, 192, 3072, 1]),  # getitem_4
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    [2048, 1024],  # _shape_param_5
    torch.randn([4096, 1024], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
