"""
Standalone repro captured via capture_hook.
Label: torchbench_moondream_infer
Pattern hash: 4f528368dfe5
Shape hash: 06ed411f
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
_shapes_config = "(T([512, 2048], f16), T([1, 512, 32], f16), T([1, 512, 32], f16), T([512, 2048], f16), S([1, 512, 2048]), S([1, 512, -1, 64]), S([1, 512, 2048]), S([1, 512, -1, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_138: "f16[512, 2048]", convert_element_type_2: "f16[1, 512, 32]", convert_element_type_3: "f16[1, 512, 32]", addmm_139: "f16[512, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_138, _shape_param_0);  addmm_138 = _shape_param_0 = None
        reshape_default_1: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:219 in forward, code: query_states[..., : self.rotary_ndims],
        slice_tensor: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:126 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_tensor, unsqueeze_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_1: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_tensor_1);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_2: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 16);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_default, slice_tensor_2], -1);  neg_default = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:127 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_1: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:128 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_1: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = None
        add_tensor: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:220 in forward, code: query_states[..., self.rotary_ndims :],
        slice_tensor_3: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 32, 9223372036854775807);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:230 in forward, code: query_states = torch.cat((query_rot, query_pass), dim=-1)
        cat_default_1: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_tensor, slice_tensor_3], -1);  add_tensor = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_139, _shape_param_2);  addmm_139 = _shape_param_2 = None
        reshape_default_3: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:223 in forward, code: key_states[..., : self.rotary_ndims],
        slice_tensor_4: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_2: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_tensor_4, unsqueeze_default);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:103 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_5: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 16, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_tensor_5);  slice_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:102 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_6: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 0, 16);  slice_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:104 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default_2: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_6], -1);  neg_default_1 = slice_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:129 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_3: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_default_2, unsqueeze_default_1);  cat_default_2 = unsqueeze_default_1 = None
        add_tensor_1: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:224 in forward, code: key_states[..., self.rotary_ndims :],
        slice_tensor_7: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 32, 9223372036854775807);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:231 in forward, code: key_states = torch.cat((key_rot, key_pass), dim=-1)
        cat_default_3: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_tensor_1, slice_tensor_7], -1);  add_tensor_1 = slice_tensor_7 = None
        return (cat_default_1, cat_default_3)



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
