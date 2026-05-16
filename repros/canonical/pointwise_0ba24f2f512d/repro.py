"""
Standalone repro captured via capture_hook.
Label: hf_qwen2_0.5b_train
Pattern hash: 0ba24f2f512d
Shape hash: 17e069db
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_69: "bf16[2048, 896]", _shape_param_0, _shape_param_1, addmm_70: "bf16[2048, 128]", _shape_param_2, _shape_param_3, addmm_71: "bf16[2048, 128]", _shape_param_4, _shape_param_5, unsqueeze_11: "bf16[1, 1, 512, 64]", unsqueeze_12: "bf16[1, 1, 512, 64]", _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(addmm_69, _shape_param_0);  addmm_69 = _shape_param_0 = None
        reshape_default_1: "bf16[4, 512, 14, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "bf16[4, 14, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "bf16[4, 512, 128]" = torch.ops.aten.reshape.default(addmm_70, _shape_param_2);  addmm_70 = _shape_param_2 = None
        reshape_default_3: "bf16[4, 512, 2, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "bf16[4, 2, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_4: "bf16[4, 512, 128]" = torch.ops.aten.reshape.default(addmm_71, _shape_param_4);  addmm_71 = _shape_param_4 = None
        reshape_default_5: "bf16[4, 512, 2, 64]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "bf16[4, 2, 512, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:144 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor: "bf16[4, 14, 512, 64]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:118 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor: "bf16[4, 14, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:119 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_1: "bf16[4, 14, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 32, 9223372036854775807);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:120 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "bf16[4, 14, 512, 32]" = torch.ops.aten.neg.default(slice_tensor_1);  slice_tensor_1 = None
        cat_default: "bf16[4, 14, 512, 64]" = torch.ops.aten.cat.default([neg_default, slice_tensor], -1);  neg_default = slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:144 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_1: "bf16[4, 14, 512, 64]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_12);  cat_default = None
        add_tensor: "bf16[4, 14, 512, 64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:145 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_2: "bf16[4, 2, 512, 64]" = torch.ops.aten.mul.Tensor(permute_default_1, unsqueeze_11);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:118 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_2: "bf16[4, 2, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:119 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_3: "bf16[4, 2, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 32, 9223372036854775807);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:120 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "bf16[4, 2, 512, 32]" = torch.ops.aten.neg.default(slice_tensor_3);  slice_tensor_3 = None
        cat_default_1: "bf16[4, 2, 512, 64]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_2], -1);  neg_default_1 = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:145 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_3: "bf16[4, 2, 512, 64]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_12);  cat_default_1 = unsqueeze_12 = None
        add_tensor_1: "bf16[4, 2, 512, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default: "bf16[4, 2, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 2);  add_tensor_1 = None
        expand_default: "bf16[4, 2, 7, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_6);  unsqueeze_default = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default: "bf16[4, 2, 7, 512, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_6: "bf16[4, 14, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_7);  clone_default = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_1: "bf16[4, 2, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_default_2, 2);  permute_default_2 = None
        expand_default_1: "bf16[4, 2, 7, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_default_1, _shape_param_8);  unsqueeze_default_1 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default_1: "bf16[4, 2, 7, 512, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_7: "bf16[4, 14, 512, 64]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_9);  clone_default_1 = _shape_param_9 = None
        return (add_tensor, reshape_default_6, reshape_default_7)


def _default_make_inputs():
    return [
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 896],  # _shape_param_0
    [4, 512, -1, 64],  # _shape_param_1
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 128],  # _shape_param_2
    [4, 512, -1, 64],  # _shape_param_3
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 128],  # _shape_param_4
    [4, 512, -1, 64],  # _shape_param_5
    torch.randn([1, 1, 512, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1, 512, 64], dtype=torch.bfloat16, device='cuda'),
    [4, 2, 7, 512, 64],  # _shape_param_6
    [4, 14, 512, 64],  # _shape_param_7
    [4, 2, 7, 512, 64],  # _shape_param_8
    [4, 14, 512, 64],  # _shape_param_9
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
