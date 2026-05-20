"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train
Pattern hash: 60c3211d2081
Shape hash: ebc48f9f
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([2048, 64], f32), T([1, 128, 64], i64, gen=Index(2048)), T([16384, 4096], f32), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 16, 256]), S([1, 128, 16, 256]), S([1, 128, 16, 256]), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 16, 64]), S([1, 128, 16, 64]))"

class Repro(torch.nn.Module):
    def forward(self, mm_108: "f32[128, 4096]", mm_109: "f32[128, 4096]", mm_110: "f32[128, 4096]", primals_305: "f32[2048, 64]", repeat_1: "i64[1, 128, 64]", primals_307: "f32[16384, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        reshape_default: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_108, _shape_param_0);  mm_108 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        reshape_default_1: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_109, _shape_param_1);  mm_109 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        reshape_default_2: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_110, _shape_param_2);  mm_110 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        reshape_default_3: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_3);  reshape_default = _shape_param_3 = None
        reshape_default_4: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_4);  reshape_default_1 = _shape_param_4 = None
        reshape_default_5: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_5);  reshape_default_2 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_default: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_305, [1, 1, 1]);  primals_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_default: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_default, 1, repeat_1);  repeat_default = repeat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_tensor = torch.ops.aten.split.Tensor(gather_default, 32, -1);  gather_default = None
        getitem: "f32[1, 128, 32]" = split_tensor[0]
        getitem_1: "f32[1, 128, 32]" = split_tensor[1];  split_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(reshape_default_4, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_tensor_1: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(reshape_default_4, 3, 64, 9223372036854775807);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_tensor_2: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(reshape_default_3, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_tensor_3: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(reshape_default_3, 3, 64, 9223372036854775807);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_default: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem, 2);  getitem = None
        unsqueeze_default_1: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        expand_default: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_default_1, _shape_param_6);  unsqueeze_default_1 = _shape_param_6 = None
        clone_default: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_6: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_7);  clone_default = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_default_2: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_1, 2);  getitem_1 = None
        unsqueeze_default_3: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        expand_default_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_default_3, _shape_param_8);  unsqueeze_default_3 = _shape_param_8 = None
        clone_default_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_7: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_9);  clone_default_1 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor, reshape_default_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_tensor_4: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_tensor_5: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 1, 9223372036854775807, 2);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_default: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_tensor_5);  slice_tensor_5 = None
        unsqueeze_default_4: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_default, 4);  neg_default = None
        unsqueeze_default_5: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_4, 4);  slice_tensor_4 = None
        cat_default: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_default_4, unsqueeze_default_5], -1);  unsqueeze_default_4 = unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        reshape_default_8: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_default, _shape_param_10);  cat_default = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_tensor_1: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(reshape_default_8, reshape_default_6);  reshape_default_8 = None
        add_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        mul_tensor_2: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_2, reshape_default_7);  reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_tensor_6: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_tensor_7: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 1, 9223372036854775807, 2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_default_1: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_tensor_7);  slice_tensor_7 = None
        unsqueeze_default_6: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_default_1, 4);  neg_default_1 = None
        unsqueeze_default_7: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_6, 4);  slice_tensor_6 = None
        cat_default_1: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_default_6, unsqueeze_default_7], -1);  unsqueeze_default_6 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        reshape_default_9: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_default_1, _shape_param_11);  cat_default_1 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_tensor_3: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(reshape_default_9, reshape_default_6);  reshape_default_9 = reshape_default_6 = None
        add_tensor_1: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_default_2: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_tensor, slice_tensor_1], -1);  add_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_default_3: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_tensor_1, slice_tensor_3], -1);  add_tensor_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_default_1: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_default_2, [0, 2, 1, 3]);  cat_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_default_2: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_default_3, [0, 2, 1, 3]);  cat_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_3: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
        return (permute_default, permute_default_1, permute_default_2, permute_default_3)


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
