"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=any, ranges=['32', '6', '128', '1'], reduction_ranges=[]
#   origins: ['aten.any.dim']
#   type=amax, ranges=['32', '6', '128', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['32', '6', '128', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_default_14: "f32[192, 128, 128]", arg81_1: "f32[32, 6]"):
        # No stacktrace found for following nodes
        reshape_default: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default_14, [32, 6, 128, 128]);  bmm_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:331 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1013 in forward, code: cache_position = torch.arange(
        iota_default_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in compute_bias, code: context_position = cache_position[:, None].to(device)
        unsqueeze_default_1: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:332 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub_tensor: "i64[128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze_default, unsqueeze_default_1);  unsqueeze_default = unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:303 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_default: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default: "i64[128, 128]" = torch.ops.aten.minimum.default(sub_tensor, full_default);  sub_tensor = full_default = None
        neg_default: "i64[128, 128]" = torch.ops.aten.neg.default(minimum_default);  minimum_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:308 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_scalar: "b8[128, 128]" = torch.ops.aten.lt.Scalar(neg_default, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:312 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_default: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(neg_default, torch.float32)
        div_tensor: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_default, 16);  convert_element_type_default = None
        log_default: "f32[128, 128]" = torch.ops.aten.log.default(div_tensor);  div_tensor = None
        div_tensor_1: "f32[128, 128]" = torch.ops.aten.div.Tensor(log_default, 2.0794415416798357);  log_default = None
        mul_tensor: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_tensor_1, 16);  div_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:315 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_default_1: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:311 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_tensor: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 16);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_1: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:316 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_default_1: "i64[128, 128]" = torch.ops.aten.minimum.default(add_tensor, full_default_1);  add_tensor = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_self: "i64[128, 128]" = torch.ops.aten.where.self(lt_scalar, neg_default, minimum_default_1);  lt_scalar = neg_default = minimum_default_1 = None
        add_tensor_1: "i64[128, 128]" = torch.ops.aten.add.Tensor(where_self, 0);  where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:339 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_default: "f32[128, 128, 6]" = torch.ops.aten.embedding.default(arg81_1, add_tensor_1);  arg81_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:340 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_default: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_default, [2, 0, 1]);  embedding_default = None
        unsqueeze_default_2: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_default, 0);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:420 in forward, code: position_bias = position_bias[:, :, -seq_length:, :]
        slice_tensor: "f32[1, 6, 128, 128]" = torch.ops.aten.slice.Tensor(unsqueeze_default_2, 2, -128, 9223372036854775807);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1259 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.triu(causal_mask, diagonal=1)
        iota_default_2: "i64[129]" = torch.ops.prims.iota.default(129, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_3: "i64[1, 129]" = torch.ops.aten.unsqueeze.default(iota_default_2, -2);  iota_default_2 = None
        iota_default_3: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_4: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(iota_default_3, -1);  iota_default_3 = None
        sub_tensor_1: "i64[128, 129]" = torch.ops.aten.sub.Tensor(unsqueeze_default_3, unsqueeze_default_4);  unsqueeze_default_3 = unsqueeze_default_4 = None
        ge_scalar: "b8[128, 129]" = torch.ops.aten.ge.Scalar(sub_tensor_1, 1);  sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1255 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.full(
        full_default_2: "f32[128, 129]" = torch.ops.aten.full.default([128, 129], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1259 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.triu(causal_mask, diagonal=1)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[128, 129]" = torch.ops.aten.where.self(ge_scalar, full_default_2, full_default_3);  ge_scalar = full_default_2 = full_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1260 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        iota_default_4: "i64[129]" = torch.ops.prims.iota.default(129, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_1: "i64[128, 1]" = torch.ops.aten.reshape.default(iota_default_1, [-1, 1]);  iota_default_1 = None
        gt_tensor: "b8[128, 129]" = torch.ops.aten.gt.Tensor(iota_default_4, reshape_default_1);  iota_default_4 = reshape_default_1 = None
        mul_tensor_1: "f32[128, 129]" = torch.ops.aten.mul.Tensor(where_self_1, gt_tensor);  where_self_1 = gt_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:423 in forward, code: causal_mask = mask[:, :, :, : key_states.shape[-2]]
        unsqueeze_default_5: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_6: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 1);  unsqueeze_default_5 = None
        expand_default: "f32[32, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_6, [32, 1, -1, -1]);  unsqueeze_default_6 = None
        slice_tensor_1: "f32[32, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default, 3, 0, 128);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:424 in forward, code: position_bias = position_bias + causal_mask
        add_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None

        # No stacktrace found for following nodes
        add_tensor_3: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default, add_tensor_2);  reshape_default = add_tensor_2 = None
        eq_scalar: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_3, -inf)
        logical_not_default: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_4: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_3, [-1], True)
        sub_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_3, amax_default);  add_tensor_3 = amax_default = None
        exp_default: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_2);  sub_tensor_2 = None
        sum_dim_int_list: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self_2: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_1, full_default_4, div_tensor_2);  logical_not_default_1 = full_default_4 = div_tensor_2 = None
        expand_default_1: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_2, [32, 6, 128, 128]);  where_self_2 = None
        reshape_default_2: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default_1, [192, 128, 128]);  expand_default_1 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 6], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
