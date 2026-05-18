"""
Standalone repro captured via capture_hook.
Label: hf_qwen2_0.5b_train
Pattern hash: 812bda4cdb34
Shape hash: 03c990c6
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
    def forward(self, getitem_284: "bf16[4, 14, 512, 64]", _shape_param_0, tangents_3: "bf16[4, 2, 512, 64]", getitem_283: "bf16[4, 14, 512, 64]", _shape_param_1, tangents_4: "bf16[4, 2, 512, 64]", unsqueeze_12: "bf16[1, 1, 512, 64]", full_default_53: "bf16[4, 2, 512, 64]", unsqueeze_11: "bf16[1, 1, 512, 64]", getitem_282: "bf16[4, 14, 512, 64]", full_default_55: "bf16[4, 14, 512, 64]", _shape_param_2, _shape_param_3, primals_21: "bf16[128, 896]", _shape_param_4, _shape_param_5, primals_19: "bf16[128, 896]", _shape_param_6, _shape_param_7, primals_17: "bf16[896, 896]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        reshape_default: "bf16[4, 2, 7, 512, 64]" = torch.ops.aten.reshape.default(getitem_284, _shape_param_0);  getitem_284 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_dim_int_list: "bf16[4, 2, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(reshape_default, [2], True);  reshape_default = None
        squeeze_dim: "bf16[4, 2, 512, 64]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 2);  sum_dim_int_list = None
        add_tensor: "bf16[4, 2, 512, 64]" = torch.ops.aten.add.Tensor(tangents_3, squeeze_dim);  tangents_3 = squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        reshape_default_1: "bf16[4, 2, 7, 512, 64]" = torch.ops.aten.reshape.default(getitem_283, _shape_param_1);  getitem_283 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_dim_int_list_1: "bf16[4, 2, 1, 512, 64]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [2], True);  reshape_default_1 = None
        squeeze_dim_1: "bf16[4, 2, 512, 64]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 2);  sum_dim_int_list_1 = None
        add_tensor_1: "bf16[4, 2, 512, 64]" = torch.ops.aten.add.Tensor(tangents_4, squeeze_dim_1);  tangents_4 = squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:145 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor: "bf16[4, 2, 512, 64]" = torch.ops.aten.mul.Tensor(add_tensor_1, unsqueeze_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:120 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_tensor: "bf16[4, 2, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor, 3, 0, 32)
        slice_tensor_1: "bf16[4, 2, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor, 3, 32, 64);  mul_tensor = None
        neg_default: "bf16[4, 2, 512, 32]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:119 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_default: "bf16[4, 2, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_53, neg_default, 3, 32, 9223372036854775807);  neg_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:118 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_default_1: "bf16[4, 2, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_53, slice_tensor_1, 3, 0, 32);  full_default_53 = slice_tensor_1 = None
        add_tensor_2: "bf16[4, 2, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1);  slice_scatter_default = slice_scatter_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:145 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_1: "bf16[4, 2, 512, 64]" = torch.ops.aten.mul.Tensor(add_tensor_1, unsqueeze_11);  add_tensor_1 = None
        add_tensor_3: "bf16[4, 2, 512, 64]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_1);  add_tensor_2 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:144 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_2: "bf16[4, 14, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_282, unsqueeze_12);  unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:120 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_tensor_2: "bf16[4, 14, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor_2, 3, 0, 32)
        slice_tensor_3: "bf16[4, 14, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor_2, 3, 32, 64);  mul_tensor_2 = None
        neg_default_1: "bf16[4, 14, 512, 32]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:119 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_default_2: "bf16[4, 14, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_55, neg_default_1, 3, 32, 9223372036854775807);  neg_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:118 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_default_3: "bf16[4, 14, 512, 64]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_tensor_3, 3, 0, 32);  full_default_55 = slice_tensor_3 = None
        add_tensor_4: "bf16[4, 14, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default_2, slice_scatter_default_3);  slice_scatter_default_2 = slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:144 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_3: "bf16[4, 14, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_282, unsqueeze_11);  getitem_282 = unsqueeze_11 = None
        add_tensor_5: "bf16[4, 14, 512, 64]" = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_3);  add_tensor_4 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default: "bf16[4, 512, 2, 64]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1, 3]);  add_tensor = None
        clone_default: "bf16[4, 512, 2, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "bf16[4, 512, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        reshape_default_3: "bf16[2048, 128]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "bf16[896, 128]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_default_2: "bf16[128, 896]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_3: "bf16[4, 512, 2, 64]" = torch.ops.aten.permute.default(add_tensor_3, [0, 2, 1, 3]);  add_tensor_3 = None
        clone_default_1: "bf16[4, 512, 2, 64]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        reshape_default_4: "bf16[4, 512, 128]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None
        reshape_default_5: "bf16[2048, 128]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_4: "bf16[896, 128]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_default_5: "bf16[128, 896]" = torch.ops.aten.permute.default(permute_default_4, [1, 0]);  permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_6: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(add_tensor_5, [0, 2, 1, 3]);  add_tensor_5 = None
        clone_default_2: "bf16[4, 512, 14, 64]" = torch.ops.aten.clone.default(permute_default_6, memory_format = torch.contiguous_format);  permute_default_6 = None
        reshape_default_6: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_6);  clone_default_2 = _shape_param_6 = None
        reshape_default_7: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_6, _shape_param_7);  reshape_default_6 = _shape_param_7 = None
        permute_default_7: "bf16[896, 896]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_default_8: "bf16[896, 896]" = torch.ops.aten.permute.default(permute_default_7, [1, 0]);  permute_default_7 = None
        return (reshape_default_3, permute_default_2, reshape_default_5, permute_default_5, reshape_default_7, permute_default_8)


def _default_make_inputs():
    return [
    torch.randn([4, 14, 512, 64], dtype=torch.bfloat16, device='cuda'),
    [4, 2, 7, 512, 64],  # _shape_param_0
    torch.randn([4, 2, 512, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 14, 512, 64], dtype=torch.bfloat16, device='cuda'),
    [4, 2, 7, 512, 64],  # _shape_param_1
    torch.randn([4, 2, 512, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1, 512, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 2, 512, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1, 512, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_282
    torch.randn([4, 14, 512, 64], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 128],  # _shape_param_2
    [2048, 128],  # _shape_param_3
    torch.randn([128, 896], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 128],  # _shape_param_4
    [2048, 128],  # _shape_param_5
    torch.randn([128, 896], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 896],  # _shape_param_6
    [2048, 896],  # _shape_param_7
    torch.randn([896, 896], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
