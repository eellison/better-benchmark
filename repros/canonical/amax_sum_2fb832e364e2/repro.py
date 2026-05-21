"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_base_train
Pattern hash: 2fb832e364e2
Shape hash: 0bd948b5
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
_shapes_config = "(T([1, 1, 1024], i64), T([1, 1, 1024, 1], i64), T([], f32), T([], f32), T([96, 1024, 1024], f32), T([1024, 1024], i64), T([32, 12], f32), T([124], i64), S([8, -1, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([96, 1024, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, unsqueeze_1: "i64[1, 1, 1024]", unsqueeze_2: "i64[1, 1, 1024, 1]", full_default: "f32[]", full_default_1: "f32[]", bmm_24: "f32[96, 1024, 1024]", sub: "i64[1024, 1024]", primals_106: "f32[32, 12]", inductor_seeds_default: "i64[124]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_default: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le_tensor: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_default, unsqueeze_2);  unsqueeze_default = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_default: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(le_tensor, _shape_param_0);  le_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        where_self: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default, full_default_1);  expand_default = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_24, _shape_param_1);  bmm_24 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:203 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_default_2: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub, full_default_2);  sub = full_default_2 = None
        neg_default: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum_default);  minimum_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_scalar: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg_default, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_default: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg_default, torch.float32)
        div_tensor: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_default, 16);  convert_element_type_default = None
        log_default: "f32[1024, 1024]" = torch.ops.aten.log.default(div_tensor);  div_tensor = None
        div_tensor_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log_default, 2.0794415416798357);  log_default = None
        mul_tensor: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_tensor_1, 16);  div_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_default_1: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_tensor: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 16);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_3: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_default_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_tensor, full_default_3);  add_tensor = full_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_self_1: "i64[1024, 1024]" = torch.ops.aten.where.self(lt_scalar, neg_default, minimum_default_1);  lt_scalar = neg_default = minimum_default_1 = None
        add_tensor_1: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where_self_1, 0);  where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_default: "f32[1024, 1024, 12]" = torch.ops.aten.embedding.default(primals_106, add_tensor_1);  primals_106 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_default: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(embedding_default, [2, 0, 1]);  embedding_default = None
        unsqueeze_default_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_default, 0);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_tensor_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default_1, where_self);  unsqueeze_default_1 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_3: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(reshape_default, add_tensor_2);  reshape_default = add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_default: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_tensor_3, [-1], True)
        sub_tensor: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_3, amax_default);  add_tensor_3 = amax_default = None
        exp_default: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 51);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_1: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor_2);  gt_scalar = div_tensor_2 = None
        mul_tensor_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.1111111111111112);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default_1: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_tensor_2, _shape_param_2);  mul_tensor_2 = _shape_param_2 = None
        reshape_default_1: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_3);  expand_default_1 = _shape_param_3 = None
        return reshape_default_1



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
