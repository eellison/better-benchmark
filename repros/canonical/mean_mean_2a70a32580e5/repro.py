"""
Standalone repro captured via capture_hook.
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([2048, 2048], bf16), T([128], bf16), T([1, 512, 128], bf16), T([1, 512, 128], bf16), T([2048, 1024], bf16), T([128], bf16), T([2048, 1024], bf16), T([4, 1, 512, 512], b8))"

class Repro(torch.nn.Module):
    def forward(self, mm_189: "bf16[2048, 2048]", arg302_1: "bf16[128]", convert_element_type_1: "bf16[1, 512, 128]", convert_element_type_2: "bf16[1, 512, 128]", mm_190: "bf16[2048, 1024]", arg304_1: "bf16[128]", mm_191: "bf16[2048, 1024]", expand_1: "b8[4, 1, 512, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_189, [4, 512, 2048]);  mm_189 = None
        reshape_default_1: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(reshape_default, [4, 512, -1, 128]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg302_1, convert_element_type_default_1);  arg302_1 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_tensor_1, [0, 2, 1, 3]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:177 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_2: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_1: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 64);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:178 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_1: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:179 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_3: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = None
        add_tensor_1: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default_2: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_190, [4, 512, 1024]);  mm_190 = None
        reshape_default_3: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(reshape_default_2, [4, 512, -1, 128]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_2: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(reshape_default_3, torch.float32);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar_1: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2)
        mean_dim_1: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True);  pow_tensor_scalar_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_2: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06);  mean_dim_1 = None
        rsqrt_default_1: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_4: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_default_1);  convert_element_type_default_2 = rsqrt_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_3: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        mul_tensor_5: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg304_1, convert_element_type_default_3);  arg304_1 = convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_tensor_5, [0, 2, 1, 3]);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_6: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default_1, unsqueeze_default);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:154 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_2: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:153 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_3: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:155 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:180 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_7: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_1);  cat_default_1 = unsqueeze_default_1 = None
        add_tensor_3: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_2: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_tensor_3, 2);  add_tensor_3 = None
        expand_default: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_default_2, [4, 8, 2, 512, 128]);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_4: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_default, [4, 16, 512, 128]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_5: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_191, [4, 512, 1024]);  mm_191 = None
        reshape_default_6: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(reshape_default_5, [4, 512, -1, 128]);  reshape_default_5 = None
        permute_default_2: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_3: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_default_2, 2);  permute_default_2 = None
        expand_default_1: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_default_3, [4, 8, 2, 512, 128]);  unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default_1: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_7: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_default_1, [4, 16, 512, 128]);  clone_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default, full_default_1);  expand_1 = full_default = full_default_1 = None
        return (add_tensor_1, reshape_default_4, reshape_default_7, where_self)


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
