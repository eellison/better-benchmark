"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_infer
Pattern hash: 678e04f46657
Shape hash: d2350de5
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
_shapes_config = "(T([8, 2048, 2048], f16), T([32, 8], f16), T([2048, 512], f16), T([2048, 512], f16), T([1, 2048, 512], f16), T([512], f16), T([2048, 512], f16), S([1, 8, 2048, 2048]), S([1, -1, 2048, 2048]), S([1, 8, 2048, 2048]), S([8, 2048, 2048]), S([1, 2048, 512]), S([1, 2048, -1, 64]), S([1, 8, 2048, 64]), S([8, 2048, 64]), S([1, 2048, 512]), S([2048, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_12: "f16[8, 2048, 2048]", arg56_1: "f16[32, 8]", mm_38: "f16[2048, 512]", mm_33: "f16[2048, 512]", convert_element_type_132: "f16[1, 2048, 512]", arg48_1: "f16[512]", arg49_1: "f16[2048, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_12, _shape_param_0);  bmm_12 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:228 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        iota_default: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:227 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_default_1: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, 1);  iota_default_1 = None
        add_tensor: "i64[2048, 1]" = torch.ops.aten.add.Tensor(unsqueeze_default_1, 0);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:229 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub_tensor: "i64[2048, 2048]" = torch.ops.aten.sub.Tensor(unsqueeze_default, add_tensor);  unsqueeze_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:203 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_default: "i64[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default: "i64[2048, 2048]" = torch.ops.aten.minimum.default(sub_tensor, full_default);  sub_tensor = full_default = None
        neg_default: "i64[2048, 2048]" = torch.ops.aten.neg.default(minimum_default);  minimum_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_scalar: "b8[2048, 2048]" = torch.ops.aten.lt.Scalar(neg_default, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_default: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(neg_default, torch.float32)
        div_tensor: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_default, 16);  convert_element_type_default = None
        log_default: "f32[2048, 2048]" = torch.ops.aten.log.default(div_tensor);  div_tensor = None
        div_tensor_1: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(log_default, 2.0794415416798357);  log_default = None
        mul_tensor: "f32[2048, 2048]" = torch.ops.aten.mul.Tensor(div_tensor_1, 16);  div_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_default_1: "i64[2048, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_tensor_1: "i64[2048, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 16);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_1: "i64[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_default_1: "i64[2048, 2048]" = torch.ops.aten.minimum.default(add_tensor_1, full_default_1);  add_tensor_1 = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_self: "i64[2048, 2048]" = torch.ops.aten.where.self(lt_scalar, neg_default, minimum_default_1);  lt_scalar = neg_default = minimum_default_1 = None
        add_tensor_2: "i64[2048, 2048]" = torch.ops.aten.add.Tensor(where_self, 0);  where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_default: "f16[2048, 2048, 8]" = torch.ops.aten.embedding.default(arg56_1, add_tensor_2);  arg56_1 = add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_default: "f16[8, 2048, 2048]" = torch.ops.aten.permute.default(embedding_default, [2, 0, 1]);  embedding_default = None
        unsqueeze_default_2: "f16[1, 8, 2048, 2048]" = torch.ops.aten.unsqueeze.default(permute_default, 0);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_default_2: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_3: "i64[2048]" = torch.ops.aten.add.Tensor(iota_default_2, 0);  iota_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_default_3: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(add_tensor_3, 0);  add_tensor_3 = None
        unsqueeze_default_4: "i64[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "i64[1, 1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_default_3: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_4: "i64[2048]" = torch.ops.aten.add.Tensor(iota_default_3, 0);  iota_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_default_6: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(add_tensor_4, 0);  add_tensor_4 = None
        unsqueeze_default_7: "i64[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 1);  unsqueeze_default_6 = None
        unsqueeze_default_8: "i64[1, 1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le_tensor: "b8[1, 1, 2048, 2048]" = torch.ops.aten.le.Tensor(unsqueeze_default_5, unsqueeze_default_8);  unsqueeze_default_5 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_default: "b8[1, 1, 2048, 2048]" = torch.ops.aten.expand.default(le_tensor, _shape_param_1);  le_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_2: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f16[1, 1, 2048, 2048]" = torch.ops.aten.where.self(expand_default, full_default_2, full_default_3);  expand_default = full_default_2 = full_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_tensor_5: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(unsqueeze_default_2, where_self_1);  unsqueeze_default_2 = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_6: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(reshape_default, add_tensor_5);  reshape_default = add_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_default_2: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.float32);  add_tensor_6 = None
        amax_default: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_default_2, [-1], True)
        sub_tensor_1: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, amax_default);  convert_element_type_default_2 = amax_default = None
        exp_default: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_2: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_3: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_tensor_2, torch.float16);  div_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default_1: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_default_3, _shape_param_2);  convert_element_type_default_3 = _shape_param_2 = None
        reshape_default_1: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_3);  expand_default_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_2: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_38, _shape_param_4);  mm_38 = _shape_param_4 = None
        reshape_default_3: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_5);  reshape_default_2 = _shape_param_5 = None
        permute_default_1: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default_2: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_6);  permute_default_1 = _shape_param_6 = None
        reshape_default_4: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_default_2, _shape_param_7);  expand_default_2 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        reshape_default_5: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_33, _shape_param_8);  mm_33 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_tensor_7: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_132, reshape_default_5);  convert_element_type_132 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_default_4: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_default: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_tensor_7);  add_tensor_7 = None
        any_default: "b8[]" = torch.ops.aten.any.default(isinf_default);  isinf_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "f32[]" = torch.ops.aten.where.self(any_default, full_default_4, full_default_5);  any_default = full_default_4 = full_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_default_1: "f32[]" = torch.ops.aten.neg.default(where_self_2)
        clamp_min_tensor: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_default_4, neg_default_1);  convert_element_type_default_4 = neg_default_1 = None
        clamp_max_tensor: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_tensor, where_self_2);  clamp_min_tensor = where_self_2 = None
        convert_element_type_default_5: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_tensor, torch.float16);  clamp_max_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_default_6: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float32)
        pow_tensor_scalar: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_6, 2);  convert_element_type_default_6 = None
        mean_dim: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_8: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_tensor_8);  add_tensor_8 = None
        mul_tensor_1: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, rsqrt_default);  convert_element_type_default_5 = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_default_7: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_2: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg48_1, convert_element_type_default_7);  arg48_1 = convert_element_type_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        reshape_default_6: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_9);  mul_tensor_2 = _shape_param_9 = None
        permute_default_2: "f16[512, 2048]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        return (reshape_default_1, reshape_default_4, reshape_default_6, permute_default_2)



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
