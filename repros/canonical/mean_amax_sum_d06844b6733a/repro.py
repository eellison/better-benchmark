"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_training
Pattern hash: d06844b6733a
Shape hash: 5f1a20c4
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_52: "f32[1024, 512]", _shape_param_0, inductor_seeds_default: "i64[84]", add_55: "f32[8, 128, 512]", primals_72: "f32[512]", primals_73: "f32[1024, 512]", _shape_param_1, primals_74: "f32[1024, 512]", unsqueeze_1: "i64[1, 1, 128]", unsqueeze_2: "i64[1, 1, 128, 1]", _shape_param_2, full_default: "f32[]", full_default_1: "f32[]", mm_58: "f32[1024, 384]", _shape_param_3, _shape_param_4, bmm_16: "f32[48, 128, 128]", _shape_param_5, sub: "i64[128, 128]", primals_82: "f32[32, 6]", _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        reshape_default: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(mm_52, _shape_param_0);  mm_52 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default: "f32[8, 128, 512]" = torch.ops.prims.inductor_random.default([8, 128, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(add_55, mul_tensor_1);  add_55 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[8, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[8, 128, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(primals_72, mul_tensor_2);  primals_72 = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        reshape_default_1: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_1: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_default: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le_tensor: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_default, unsqueeze_2);  unsqueeze_default = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_default: "b8[8, 1, 128, 128]" = torch.ops.aten.expand.default(le_tensor, _shape_param_2);  le_tensor = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        where_self: "f32[8, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default, full_default_1);  expand_default = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(mm_58, _shape_param_3);  mm_58 = _shape_param_3 = None
        reshape_default_3: "f32[8, 128, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_4);  reshape_default_2 = _shape_param_4 = None
        permute_default_2: "f32[8, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_4: "f32[8, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, _shape_param_5);  bmm_16 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:208 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_default_2: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default: "i64[128, 128]" = torch.ops.aten.minimum.default(sub, full_default_2);  sub = full_default_2 = None
        neg_default: "i64[128, 128]" = torch.ops.aten.neg.default(minimum_default);  minimum_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:213 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_scalar: "b8[128, 128]" = torch.ops.aten.lt.Scalar(neg_default, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:217 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_default: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(neg_default, torch.float32)
        div_tensor: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_default, 16);  convert_element_type_default = None
        log_default: "f32[128, 128]" = torch.ops.aten.log.default(div_tensor);  div_tensor = None
        div_tensor_1: "f32[128, 128]" = torch.ops.aten.div.Tensor(log_default, 2.0794415416798357);  log_default = None
        mul_tensor_4: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_tensor_1, 16);  div_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:220 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_default_1: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.int64);  mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:216 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_tensor_2: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 16);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:222 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_3: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:221 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_default_1: "i64[128, 128]" = torch.ops.aten.minimum.default(add_tensor_2, full_default_3);  add_tensor_2 = full_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:225 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_self_1: "i64[128, 128]" = torch.ops.aten.where.self(lt_scalar, neg_default, minimum_default_1);  lt_scalar = neg_default = minimum_default_1 = None
        add_tensor_3: "i64[128, 128]" = torch.ops.aten.add.Tensor(where_self_1, 0);  where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:241 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_default: "f32[128, 128, 6]" = torch.ops.aten.embedding.default(primals_82, add_tensor_3);  primals_82 = add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_default_3: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_default, [2, 0, 1]);  embedding_default = None
        unsqueeze_default_1: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_default_3, 0);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        add_tensor_4: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_default_1, where_self);  unsqueeze_default_1 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_5: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default_4, add_tensor_4);  reshape_default_4 = add_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_default: "f32[8, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_5, [-1], True)
        sub_tensor: "f32[8, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_5, amax_default);  add_tensor_5 = amax_default = None
        exp_default: "f32[8, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_2: "f32[8, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35);  inductor_seeds_default = None
        inductor_random_default_1: "f32[8, 6, 128, 128]" = torch.ops.prims.inductor_random.default([8, 6, 128, 128], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_scalar_1: "b8[8, 6, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_tensor_5: "f32[8, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_scalar_1, div_tensor_2);  gt_scalar_1 = div_tensor_2 = None
        mul_tensor_6: "f32[8, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_5, 1.1111111111111112);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default_1: "f32[8, 6, 128, 128]" = torch.ops.aten.expand.default(mul_tensor_6, _shape_param_6);  mul_tensor_6 = _shape_param_6 = None
        reshape_default_5: "f32[48, 128, 128]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_7);  expand_default_1 = _shape_param_7 = None
        expand_default_2: "f32[8, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_2, _shape_param_8);  permute_default_2 = _shape_param_8 = None
        clone_default: "f32[8, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        reshape_default_6: "f32[48, 128, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_9);  clone_default = _shape_param_9 = None
        return (permute_default, reshape_default_1, permute_default_1, reshape_default_5, reshape_default_6)


def _default_make_inputs():
    return [
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_0
    torch.randint(0, 2, [84], dtype=torch.int64, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_1
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 1, 128], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1, 1, 128, 1], dtype=torch.int64, device='cuda'),
    [8, -1, 128, 128],  # _shape_param_2
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [8, 128, 384],  # _shape_param_3
    [8, 128, -1, 64],  # _shape_param_4
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 128],  # _shape_param_5
    torch.randint(0, 2, [128, 128], dtype=torch.int64, device='cuda'),
    torch.randn([32, 6], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 128],  # _shape_param_6
    [48, 128, 128],  # _shape_param_7
    [8, 6, 128, 64],  # _shape_param_8
    [48, 128, 64],  # _shape_param_9
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
