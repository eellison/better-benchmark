"""
Standalone repro captured via capture_hook.
Label: hf_qwen2_0.5b_train
Pattern hash: 8e563be61831
Shape hash: ce539f14
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
    def forward(self, getitem_218: "bf16[4, 14, 512, 64]", _shape_param_0, tangents_47: "bf16[4, 2, 512, 64]", getitem_217: "bf16[4, 14, 512, 64]", _shape_param_1, tangents_48: "bf16[4, 2, 512, 64]", primals_3: "f32[32]", _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, getitem_216: "bf16[4, 14, 512, 64]", _shape_param_6, _shape_param_7, primals_285: "bf16[128, 896]", _shape_param_8, _shape_param_9, primals_283: "bf16[128, 896]", _shape_param_10, _shape_param_11, primals_281: "bf16[896, 896]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        reshape_default: "bf16[4, 2, 7, 512, 64]" = torch.ops.aten.reshape.default(getitem_218, _shape_param_0);  getitem_218 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_dim_int_list: "bf16[4, 2, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(reshape_default, [2], True);  reshape_default = None
        squeeze_dim: "bf16[4, 2, 512, 64]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 2);  sum_dim_int_list = None
        add_tensor: "bf16[4, 2, 512, 64]" = torch.ops.aten.add.Tensor(tangents_47, squeeze_dim);  tangents_47 = squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        reshape_default_1: "bf16[4, 2, 7, 512, 64]" = torch.ops.aten.reshape.default(getitem_217, _shape_param_1);  getitem_217 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_dim_int_list_1: "bf16[4, 2, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [2], True);  reshape_default_1 = None
        squeeze_dim_1: "bf16[4, 2, 512, 64]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 2);  sum_dim_int_list_1 = None
        add_tensor_1: "bf16[4, 2, 512, 64]" = torch.ops.aten.add.Tensor(tangents_48, squeeze_dim_1);  tangents_48 = squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:374 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_2: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:375 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_2, 0);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_default_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:103 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_default_2: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
        unsqueeze_default_3: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 2);  unsqueeze_default_2 = None
        expand_default: "f32[1, 32, 1]" = torch.ops.aten.expand.default(unsqueeze_default_3, [1, -1, 1]);  unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:104 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        convert_element_type_default: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float32);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:108 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_1: "f32[1, 32, 1]" = torch.ops.aten.expand.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        expand_default_2: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_default, _shape_param_3);  convert_element_type_default = _shape_param_3 = None
        mul_tensor: "f32[1, 32, 512]" = torch.ops.aten.mul.Tensor(expand_default_1, expand_default_2);  expand_default_1 = expand_default_2 = None
        permute_default: "f32[1, 512, 32]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:109 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_default_4: "f32[1, 512, 1, 32]" = torch.ops.aten.unsqueeze.default(permute_default, 2);  permute_default = None
        expand_default_3: "f32[1, 512, 2, 32]" = torch.ops.aten.expand.default(unsqueeze_default_4, _shape_param_4);  unsqueeze_default_4 = _shape_param_4 = None
        clone_default: "f32[1, 512, 2, 32]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        reshape_default_2: "f32[1, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:111 in forward, code: sin = emb.sin() * self.attention_scaling
        sin_default: "f32[1, 512, 64]" = torch.ops.aten.sin.default(reshape_default_2)
        mul_tensor_1: "f32[1, 512, 64]" = torch.ops.aten.mul.Tensor(sin_default, 1.0);  sin_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:113 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_1: "bf16[1, 512, 64]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:143 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_5: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 1);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:145 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_2: "bf16[4, 2, 512, 64]" = torch.ops.aten.mul.Tensor(add_tensor_1, unsqueeze_default_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:120 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_tensor: "bf16[4, 2, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor_2, 3, 0, 32)
        slice_tensor_1: "bf16[4, 2, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor_2, 3, 32, 64);  mul_tensor_2 = None
        neg_default: "bf16[4, 2, 512, 32]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:119 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default: "bf16[4, 2, 512, 64]" = torch.ops.aten.full.default([4, 2, 512, 64], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "bf16[4, 2, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default, neg_default, 3, 32, 9223372036854775807);  neg_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:118 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_default_1: "bf16[4, 2, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_1, 3, 0, 32);  full_default = slice_tensor_1 = None
        add_tensor_3: "bf16[4, 2, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1);  slice_scatter_default = slice_scatter_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:110 in forward, code: cos = emb.cos() * self.attention_scaling
        cos_default: "f32[1, 512, 64]" = torch.ops.aten.cos.default(reshape_default_2);  reshape_default_2 = None
        mul_tensor_3: "f32[1, 512, 64]" = torch.ops.aten.mul.Tensor(cos_default, 1.0);  cos_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:113 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_2: "bf16[1, 512, 64]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:142 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default_6: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 1);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:145 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_4: "bf16[4, 2, 512, 64]" = torch.ops.aten.mul.Tensor(add_tensor_1, unsqueeze_default_6);  add_tensor_1 = None
        add_tensor_4: "bf16[4, 2, 512, 64]" = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_4);  add_tensor_3 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:144 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_5: "bf16[4, 14, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_216, unsqueeze_default_5);  unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:120 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_tensor_2: "bf16[4, 14, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor_5, 3, 0, 32)
        slice_tensor_3: "bf16[4, 14, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor_5, 3, 32, 64);  mul_tensor_5 = None
        neg_default_1: "bf16[4, 14, 512, 32]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:119 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default_1: "bf16[4, 14, 512, 64]" = torch.ops.aten.full.default([4, 14, 512, 64], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_2: "bf16[4, 14, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_1, neg_default_1, 3, 32, 9223372036854775807);  neg_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:118 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_default_3: "bf16[4, 14, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_tensor_3, 3, 0, 32);  full_default_1 = slice_tensor_3 = None
        add_tensor_5: "bf16[4, 14, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default_2, slice_scatter_default_3);  slice_scatter_default_2 = slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:144 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_6: "bf16[4, 14, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_216, unsqueeze_default_6);  getitem_216 = unsqueeze_default_6 = None
        add_tensor_6: "bf16[4, 14, 512, 64]" = torch.ops.aten.add.Tensor(add_tensor_5, mul_tensor_6);  add_tensor_5 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_1: "bf16[4, 512, 2, 64]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1, 3]);  add_tensor = None
        clone_default_1: "bf16[4, 512, 2, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "bf16[4, 512, 128]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        reshape_default_4: "bf16[2048, 128]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_7);  reshape_default_3 = _shape_param_7 = None
        permute_default_2: "bf16[896, 128]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
        permute_default_3: "bf16[128, 896]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_4: "bf16[4, 512, 2, 64]" = torch.ops.aten.permute.default(add_tensor_4, [0, 2, 1, 3]);  add_tensor_4 = None
        clone_default_2: "bf16[4, 512, 2, 64]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        reshape_default_5: "bf16[4, 512, 128]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_8);  clone_default_2 = _shape_param_8 = None
        reshape_default_6: "bf16[2048, 128]" = torch.ops.aten.reshape.default(reshape_default_5, _shape_param_9);  reshape_default_5 = _shape_param_9 = None
        permute_default_5: "bf16[896, 128]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
        permute_default_6: "bf16[128, 896]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_7: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(add_tensor_6, [0, 2, 1, 3]);  add_tensor_6 = None
        clone_default_3: "bf16[4, 512, 14, 64]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_7: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_10);  clone_default_3 = _shape_param_10 = None
        reshape_default_8: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_11);  reshape_default_7 = _shape_param_11 = None
        permute_default_8: "bf16[896, 896]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
        permute_default_9: "bf16[896, 896]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (reshape_default_4, permute_default_3, reshape_default_6, permute_default_6, reshape_default_8, permute_default_9)


def _default_make_inputs():
    return [
    torch.randn([4, 14, 512, 64], dtype=torch.bfloat16, device='cuda'),
    [4, 2, 7, 512, 64],  # _shape_param_0
    torch.randn([4, 2, 512, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 14, 512, 64], dtype=torch.bfloat16, device='cuda'),
    [4, 2, 7, 512, 64],  # _shape_param_1
    torch.randn([4, 2, 512, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    [1, 32, 1],  # _shape_param_2
    [1, 1, 512],  # _shape_param_3
    [1, 512, 2, 32],  # _shape_param_4
    [1, 512, 64],  # _shape_param_5
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_216
    [4, 512, 128],  # _shape_param_6
    [2048, 128],  # _shape_param_7
    torch.randn([128, 896], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 128],  # _shape_param_8
    [2048, 128],  # _shape_param_9
    torch.randn([128, 896], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 896],  # _shape_param_10
    [2048, 896],  # _shape_param_11
    torch.randn([896, 896], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
