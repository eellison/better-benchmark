"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['4', '8', '1', '512', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4', '512', '8', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_334: "bf16[4, 16, 512, 128]", unsqueeze_6: "bf16[1, 1, 512, 128]", full_default_62: "bf16[4, 8, 512, 128]", unsqueeze_5: "bf16[1, 1, 512, 128]", primals_8: "bf16[128]", mm_1: "bf16[2048, 1024]", rsqrt_2: "f32[4, 512, 8, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        reshape_default: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.reshape.default(getitem_334, [4, 8, 2, 512, 128]);  getitem_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_dim_int_list: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(reshape_default, [2], True);  reshape_default = None
        squeeze_dim: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 2);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:116 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_dim, unsqueeze_6);  unsqueeze_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:90 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_tensor: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_tensor, 3, 0, 64)
        slice_tensor_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_tensor, 3, 64, 128);  mul_tensor = None
        neg_default: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:89 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_default: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_62, neg_default, 3, 64, 9223372036854775807);  neg_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:88 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_default_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_62, slice_tensor_1, 3, 0, 64);  full_default_62 = slice_tensor_1 = None
        add_tensor: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1);  slice_scatter_default = slice_scatter_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:116 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_dim, unsqueeze_5);  squeeze_dim = unsqueeze_5 = None
        add_tensor_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_1);  add_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:201 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(add_tensor_1, [0, 2, 1, 3]);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor_2: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_default, primals_8);  primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:201 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default_1: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_1, [4, 512, 1024]);  mm_1 = None
        reshape_default_2: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(reshape_default_1, [4, 512, -1, 128]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_3: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        mul_tensor_4: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_default, convert_element_type_default_1);  permute_default = convert_element_type_default_1 = None
        sum_dim_int_list_1: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1, 2], True);  mul_tensor_4 = None
        reshape_default_3: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [128]);  sum_dim_int_list_1 = None
        convert_element_type_default_2: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float32);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_5: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default)
        mul_tensor_6: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_2);  convert_element_type_default_2 = None
        sum_dim_int_list_2: "f32[4, 512, 8, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [3], True);  mul_tensor_5 = None
        pow_tensor_scalar: "f32[4, 512, 8, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_scalar: "f32[4, 512, 8, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_2, -0.5);  sum_dim_int_list_2 = None
        mul_tensor_7: "f32[4, 512, 8, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 512, 8, 128]" = torch.ops.aten.expand.default(mul_tensor_7, [4, 512, 8, 128]);  mul_tensor_7 = None
        div_scalar: "f32[4, 512, 8, 128]" = torch.ops.aten.div.Scalar(expand_default, 128);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_8: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 8, 128]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_8);  mul_tensor_6 = mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_3: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:201 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        clone_default: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(convert_element_type_default_3, memory_format = torch.contiguous_format);  convert_element_type_default_3 = None
        reshape_default_4: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_default, [4, 512, 1024]);  clone_default = None
        reshape_default_5: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(reshape_default_4, [2048, 1024]);  reshape_default_4 = None
        permute_default_1: "bf16[1024, 2048]" = torch.ops.aten.permute.default(reshape_default_5, [1, 0]);  reshape_default_5 = None
        return (reshape_default_3, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([4, 16, 512, 128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1, 512, 128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 8, 512, 128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1, 512, 128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 8, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
