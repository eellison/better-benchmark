"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg8_1: "f32[1, 512, 32]", getitem: "f32[4, 64, 512, 32]", arg9_1: "f32[1, 512, 32]", getitem_1: "f32[4, 64, 512, 32]", getitem_2: "f32[4, 8, 512, 32]", getitem_3: "f32[4, 8, 512, 32]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:234 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default: "f32[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(arg8_1, 1);  arg8_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:228 in _apply_rotary_emb, code: first_ = first_half * cos - second_half * sin
        mul_tensor: "f32[4, 64, 512, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:235 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_1: "f32[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(arg9_1, 1);  arg9_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:228 in _apply_rotary_emb, code: first_ = first_half * cos - second_half * sin
        mul_tensor_1: "f32[4, 64, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_default_1)
        sub_tensor: "f32[4, 64, 512, 32]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:229 in _apply_rotary_emb, code: second_ = second_half * cos + first_half * sin
        mul_tensor_2: "f32[4, 64, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_default);  getitem_1 = None
        mul_tensor_3: "f32[4, 64, 512, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_default_1);  getitem = None
        add_tensor: "f32[4, 64, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:230 in _apply_rotary_emb, code: return torch.cat((first_, second_), dim=-1)
        cat_default: "f32[4, 64, 512, 64]" = torch.ops.aten.cat.default([sub_tensor, add_tensor], -1);  sub_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:253 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        expand_default: "f32[4, 64, 512, 64]" = torch.ops.aten.expand.default(cat_default, [4, 64, 512, 64]);  cat_default = None
        reshape_default: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(expand_default, [256, 512, 64]);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:228 in _apply_rotary_emb, code: first_ = first_half * cos - second_half * sin
        mul_tensor_4: "f32[4, 8, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_2, unsqueeze_default)
        mul_tensor_5: "f32[4, 8, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_3, unsqueeze_default_1)
        sub_tensor_1: "f32[4, 8, 512, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:229 in _apply_rotary_emb, code: second_ = second_half * cos + first_half * sin
        mul_tensor_6: "f32[4, 8, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_3, unsqueeze_default);  getitem_3 = unsqueeze_default = None
        mul_tensor_7: "f32[4, 8, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_2, unsqueeze_default_1);  getitem_2 = unsqueeze_default_1 = None
        add_tensor_1: "f32[4, 8, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:230 in _apply_rotary_emb, code: return torch.cat((first_, second_), dim=-1)
        cat_default_1: "f32[4, 8, 512, 64]" = torch.ops.aten.cat.default([sub_tensor_1, add_tensor_1], -1);  sub_tensor_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:218 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_2: "f32[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(cat_default_1, 2);  cat_default_1 = None
        expand_default_1: "f32[4, 8, 8, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_default_2, [4, 8, 8, 512, 64]);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:219 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default: "f32[4, 8, 8, 512, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_1: "f32[4, 64, 512, 64]" = torch.ops.aten.reshape.default(clone_default, [4, 64, 512, 64]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:253 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        permute_default: "f32[4, 64, 64, 512]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None
        expand_default_2: "f32[4, 64, 64, 512]" = torch.ops.aten.expand.default(permute_default, [4, 64, 64, 512]);  permute_default = None
        reshape_default_2: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(expand_default_2, [256, 64, 512]);  expand_default_2 = None
        return (reshape_default, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([1, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 64, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 64, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 8, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 8, 512, 32], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
