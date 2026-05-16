"""
Standalone repro captured via capture_hook.
Label: hf_pythia_410m_train
Pattern hash: 43361b728ea1
Shape hash: 7f40d19e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_370: "f16[4, 16, 512, 64]", getitem_371: "f16[4, 16, 512, 64]", unsqueeze_11: "f16[1, 1, 512, 16]", unsqueeze_12: "f16[1, 1, 512, 16]", getitem_372: "f16[4, 16, 512, 64]", mul_256: "f32[4, 512, 1024]", primals_286: "f16[1024]", primals_287: "f16[1024]", _shape_param_0, primals_288: "f16[4096, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:149 in apply_rotary_pos_emb, code: q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        slice_tensor: "f16[4, 16, 512, 16]" = torch.ops.aten.slice.Tensor(getitem_370, 3, 0, 16)
        slice_tensor_1: "f16[4, 16, 512, 48]" = torch.ops.aten.slice.Tensor(getitem_370, 3, 16, 9223372036854775807);  getitem_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:150 in apply_rotary_pos_emb, code: k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        slice_tensor_2: "f16[4, 16, 512, 16]" = torch.ops.aten.slice.Tensor(getitem_371, 3, 0, 16)
        slice_tensor_3: "f16[4, 16, 512, 48]" = torch.ops.aten.slice.Tensor(getitem_371, 3, 16, 9223372036854775807);  getitem_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:153 in apply_rotary_pos_emb, code: q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        mul_tensor: "f16[4, 16, 512, 16]" = torch.ops.aten.mul.Tensor(slice_tensor, unsqueeze_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:121 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_4: "f16[4, 16, 512, 8]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:122 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_5: "f16[4, 16, 512, 8]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 8, 9223372036854775807);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:123 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "f16[4, 16, 512, 8]" = torch.ops.aten.neg.default(slice_tensor_5);  slice_tensor_5 = None
        cat_default: "f16[4, 16, 512, 16]" = torch.ops.aten.cat.default([neg_default, slice_tensor_4], -1);  neg_default = slice_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:153 in apply_rotary_pos_emb, code: q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        mul_tensor_1: "f16[4, 16, 512, 16]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_12);  cat_default = None
        add_tensor: "f16[4, 16, 512, 16]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:154 in apply_rotary_pos_emb, code: k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        mul_tensor_2: "f16[4, 16, 512, 16]" = torch.ops.aten.mul.Tensor(slice_tensor_2, unsqueeze_11);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:121 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_6: "f16[4, 16, 512, 8]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 0, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:122 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_7: "f16[4, 16, 512, 8]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 8, 9223372036854775807);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:123 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "f16[4, 16, 512, 8]" = torch.ops.aten.neg.default(slice_tensor_7);  slice_tensor_7 = None
        cat_default_1: "f16[4, 16, 512, 16]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_6], -1);  neg_default_1 = slice_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:154 in apply_rotary_pos_emb, code: k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        mul_tensor_3: "f16[4, 16, 512, 16]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_12);  cat_default_1 = unsqueeze_12 = None
        add_tensor_1: "f16[4, 16, 512, 16]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:157 in apply_rotary_pos_emb, code: q_embed = torch.cat([q_embed, q_pass], dim=-1)
        cat_default_2: "f16[4, 16, 512, 64]" = torch.ops.aten.cat.default([add_tensor, slice_tensor_1], -1);  add_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:158 in apply_rotary_pos_emb, code: k_embed = torch.cat([k_embed, k_pass], dim=-1)
        cat_default_3: "f16[4, 16, 512, 64]" = torch.ops.aten.cat.default([add_tensor_1, slice_tensor_3], -1);  add_tensor_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:144 in update, code: self.values = torch.cat([self.values, value_states], dim=-2)
        clone_default: "f16[4, 16, 512, 64]" = torch.ops.aten.clone.default(getitem_372);  getitem_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_4: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_256, primals_286);  mul_256 = primals_286 = None
        add_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_4, primals_287);  mul_tensor_4 = primals_287 = None
        convert_element_type_default: "f16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        reshape_default: "f16[2048, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_0);  convert_element_type_default = _shape_param_0 = None
        permute_default: "f16[1024, 4096]" = torch.ops.aten.permute.default(primals_288, [1, 0]);  primals_288 = None
        return (cat_default_2, cat_default_3, clone_default, reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn(6291328, dtype=torch.float16, device='cuda').as_strided([4, 16, 512, 64], [1572864, 192, 3072, 1]),  # getitem_370
    torch.randn(6291328, dtype=torch.float16, device='cuda').as_strided([4, 16, 512, 64], [1572864, 192, 3072, 1]),  # getitem_371
    torch.randn([1, 1, 512, 16], dtype=torch.float16, device='cuda'),
    torch.randn([1, 1, 512, 16], dtype=torch.float16, device='cuda'),
    torch.randn(6291328, dtype=torch.float16, device='cuda').as_strided([4, 16, 512, 64], [1572864, 192, 3072, 1]),  # getitem_372
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    [2048, 1024],  # _shape_param_0
    torch.randn([4096, 1024], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
