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

_shapes_config = "(T([2048, 8], f32), T([16384], i64, gen=Index(16384)), T([16384, 2048], bf16), T([16384, 1], b8), T([4, 512, 2048], bf16), T([2048], bf16), T([151936, 2048], bf16))"

class Repro(torch.nn.Module):
    def forward(self, getitem_54: "f32[2048, 8]", getitem_57: "i64[16384]", _grouped_mm_7: "bf16[16384, 2048]", unsqueeze_36: "b8[16384, 1]", add_35: "bf16[4, 512, 2048]", arg47_1: "bf16[2048]", arg48_1: "bf16[151936, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        full_default: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:269 in forward, code: router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        sum_dim_int_list: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(getitem_54, [-1], True)
        div_tensor: "f32[2048, 8]" = torch.ops.aten.div.Tensor(getitem_54, sum_dim_int_list);  getitem_54 = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_default: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        reshape_default: "bf16[16384]" = torch.ops.aten.reshape.default(convert_element_type_default, [-1]);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        index_tensor: "bf16[16384]" = torch.ops.aten.index.Tensor(reshape_default, [getitem_57]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        unsqueeze_default: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_tensor, -1);  index_tensor = None
        mul_tensor: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(_grouped_mm_7, unsqueeze_default);  _grouped_mm_7 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:461 in grouped_mm_experts_forward, code: weighted_out.masked_fill_(sentinel_mask, 0.0)
        where_self: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_36, full_default, mul_tensor);  unsqueeze_36 = full_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:464 in grouped_mm_experts_forward, code: inv_perm = torch.empty_like(perm)
        empty_memory_format: "i64[16384]" = torch.ops.aten.empty.memory_format([16384], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:465 in grouped_mm_experts_forward, code: inv_perm[perm] = torch.arange(perm.size(0), device=device)
        iota_default: "i64[16384]" = torch.ops.prims.iota.default(16384, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put_default: "i64[16384]" = torch.ops.aten.index_put.default(empty_memory_format, [getitem_57], iota_default);  empty_memory_format = getitem_57 = iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:466 in grouped_mm_experts_forward, code: weighted_out = weighted_out[inv_perm]  # (S, hidden_dim)
        index_tensor_1: "bf16[16384, 2048]" = torch.ops.aten.index.Tensor(where_self, [index_put_default]);  where_self = index_put_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:472 in grouped_mm_experts_forward, code: final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        reshape_default_1: "bf16[2048, 8, 2048]" = torch.ops.aten.reshape.default(index_tensor_1, [2048, 8, 2048]);  index_tensor_1 = None
        sum_dim_int_list_1: "bf16[2048, 2048]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [1]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:286 in forward, code: return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        reshape_default_2: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [4, 512, 2048]);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:352 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_35, reshape_default_2);  add_35 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_1: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_1, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_default);  convert_element_type_default_1 = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_2: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg47_1, convert_element_type_default_2);  arg47_1 = convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:684 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        reshape_default_3: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(mul_tensor_2, [2048, 2048]);  mul_tensor_2 = None
        permute_default: "bf16[2048, 151936]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        return (reshape_default_3, permute_default)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    import torch
    inputs = parse_shapes_config(_shapes_config)
    # getitem_57 (input 1) must be a true permutation of [0, 16384) for inverse-perm logic
    inputs[1] = torch.randperm(16384, dtype=torch.int64, device='cuda')
    return inputs


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
