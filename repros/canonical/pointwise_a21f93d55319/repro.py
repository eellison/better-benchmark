"""
Standalone repro captured via capture_hook.
Label: torchbench_llava_infer
Pattern hash: a21f93d55319
Shape hash: 34aef19d
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
_shapes_config = "(T([512, 4096], f16), T([64], f16), T([512, 4096], f16), S([1, 512, 4096]), S([1, 512, -1, 128]), S([1, 64, 1]), S([1, 1, 512]), S([1, 512, 2, 64]), S([1, 512, 128]), S([1, 512, 4096]), S([1, 512, -1, 128]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f16[512, 4096]", arg2_1: "f16[64]", mm_1: "f16[512, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:262 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        reshape_default_1: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:125 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_default: "f16[1, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_default_1: "f16[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        convert_element_type_default: "f32[1, 64, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float32);  unsqueeze_default_1 = None
        expand_default: "f32[1, 64, 1]" = torch.ops.aten.expand.default(convert_element_type_default, [1, -1, 1]);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:130 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_1: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:396 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:397 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default_2: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:126 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_default_3: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 1);  unsqueeze_default_2 = None
        convert_element_type_default_1: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_3, torch.float32);  unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:130 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_2: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_default_1, _shape_param_3);  convert_element_type_default_1 = _shape_param_3 = None
        mul_tensor: "f32[1, 64, 512]" = torch.ops.aten.mul.Tensor(expand_default_1, expand_default_2);  expand_default_1 = expand_default_2 = None
        permute_default_1: "f32[1, 512, 64]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:131 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_default_4: "f32[1, 512, 1, 64]" = torch.ops.aten.unsqueeze.default(permute_default_1, 2);  permute_default_1 = None
        expand_default_3: "f32[1, 512, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_default_4, _shape_param_4);  unsqueeze_default_4 = _shape_param_4 = None
        clone_default: "f32[1, 512, 2, 64]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        reshape_default_2: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:132 in forward, code: cos = emb.cos() * self.attention_scaling
        cos_default: "f32[1, 512, 128]" = torch.ops.aten.cos.default(reshape_default_2)
        mul_tensor_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(cos_default, 1.0);  cos_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:135 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_2: "f16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:164 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default_5: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 1);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_2: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_1: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 64);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:133 in forward, code: sin = emb.sin() * self.attention_scaling
        sin_default: "f32[1, 512, 128]" = torch.ops.aten.sin.default(reshape_default_2);  reshape_default_2 = None
        mul_tensor_3: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sin_default, 1.0);  sin_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:135 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_3: "f16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.float16);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:165 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_6: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, 1);  convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:166 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_4: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_6);  cat_default = None
        add_tensor_1: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:263 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_3: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_1, _shape_param_6);  mm_1 = _shape_param_6 = None
        reshape_default_4: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_7);  reshape_default_3 = _shape_param_7 = None
        permute_default_2: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_5: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default_2, unsqueeze_default_5);  unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:141 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_2: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:140 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_3: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 0, 64);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:142 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default_1: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:167 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_6: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_6);  cat_default_1 = unsqueeze_default_6 = None
        add_tensor_2: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        return (add_tensor_1, add_tensor_2)



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
