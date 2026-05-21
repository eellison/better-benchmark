"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['4', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import glob
import os
import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 2048], bf16), T([2048, 2048], bf16), T([2048, 2048], bf16), T([2048], bf16), T([4, 512, 2048], bf16), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([16384, 2048], bf16), T([16384], i64, gen=Index(100)), T([16384, 1], b8), T([], bf16), T([2048, 8], f32), T([16384], i64, gen=Index(100)), T([128, 2048, 768], bf16), T([16384, 1536], bf16))"

class Repro(torch.nn.Module):
    def forward(self, mm_48: "bf16[2048, 2048]", mm_50: "bf16[2048, 2048]", mm_52: "bf16[2048, 2048]", primals_15: "bf16[2048]", add_11: "bf16[4, 512, 2048]", rsqrt_4: "f32[4, 512, 1]", add_79: "bf16[4, 512, 2048]", full_default_22: "bf16[16384, 2048]", index_put: "i64[16384]", unsqueeze_18: "b8[16384, 1]", full_default_2: "bf16[]", div_1: "f32[2048, 8]", getitem_12: "i64[16384]", primals_14: "bf16[128, 2048, 768]", _grouped_mm: "bf16[16384, 1536]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_48, [4, 512, 2048]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default_1: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_50, [4, 512, 2048]);  mm_50 = None
        add_tensor: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default_2: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_52, [4, 512, 2048]);  mm_52 = None
        add_tensor_1: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_15);  add_tensor_1 = primals_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default)
        mul_tensor_2: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_4);  convert_element_type_default_1 = None
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_4, 3);  rsqrt_4 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_tensor_3, [4, 512, 2048]);  mul_tensor_3 = None
        div_scalar: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_default, 2048);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_2: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        add_tensor_3: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_79, convert_element_type_default_2);  add_79 = convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:286 in forward, code: return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        reshape_default_3: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_tensor_3, [2048, 2048]);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:472 in grouped_mm_experts_forward, code: final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        unsqueeze_default: "bf16[2048, 1, 2048]" = torch.ops.aten.unsqueeze.default(reshape_default_3, 1);  reshape_default_3 = None
        expand_default_1: "bf16[2048, 8, 2048]" = torch.ops.aten.expand.default(unsqueeze_default, [2048, 8, 2048]);  unsqueeze_default = None
        clone_default: "bf16[2048, 8, 2048]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "bf16[16384, 2048]" = torch.ops.aten.reshape.default(clone_default, [16384, 2048]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:466 in grouped_mm_experts_forward, code: weighted_out = weighted_out[inv_perm]  # (S, hidden_dim)
        index_put_default: "bf16[16384, 2048]" = torch.ops.aten.index_put.default(full_default_22, [index_put], reshape_default_4, True);  full_default_22 = index_put = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        where_self: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_18, full_default_2, index_put_default);  unsqueeze_18 = full_default_2 = index_put_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_default_3: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        reshape_default_5: "bf16[16384]" = torch.ops.aten.reshape.default(convert_element_type_default_3, [-1]);  convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_tensor: "bf16[16384]" = torch.ops.aten.index.Tensor(reshape_default_5, [getitem_12]);  reshape_default_5 = getitem_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        unsqueeze_default_1: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_tensor, -1);  index_tensor = None
        mul_tensor_5: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_self, unsqueeze_default_1);  where_self = unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_default: "bf16[128, 768, 2048]" = torch.ops.aten.permute.default(primals_14, [0, 2, 1]);  primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:6782 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_default_1: "bf16[128, 2048, 768]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        split_tensor = torch.ops.aten.split.Tensor(_grouped_mm, 768, -1);  _grouped_mm = None
        getitem: "bf16[16384, 768]" = split_tensor[0]
        getitem_13: "bf16[16384, 768]" = split_tensor[1];  split_tensor = None
        return (mul_tensor_5, permute_default_1, getitem, getitem_13)


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
