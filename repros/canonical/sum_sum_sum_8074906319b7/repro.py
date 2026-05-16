"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['8', '8', '1024', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '8', '1024', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '8', '1024', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '8', '1024', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '8', '1024', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '8', '1024', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '8', '1024', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['8', '8', '1024', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '8', '1024', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_473: "f32[8, 8, 1024, 1024]", view_524: "f32[8, 8, 1024, 1024]", view_575: "f32[8, 8, 1024, 1024]", view_626: "f32[8, 8, 1024, 1024]", view_677: "f32[8, 8, 1024, 1024]", bmm_81: "f32[64, 1024, 1024]", gt_29: "b8[8, 8, 1024, 1024]", ge: "b8[1024, 1025]", full_default_2: "f32[]", bmm_12: "f32[64, 1024, 1024]", embedding_3: "f32[1024, 1024, 8]", amax_6: "f32[8, 8, 1024, 1]", sum_7: "f32[8, 8, 1024, 1]", add_38: "i64[1024, 1024]", view_757: "f32[8, 8, 1024, 1024]", view_785: "f32[8, 8, 1024, 1024]", view_813: "f32[8, 8, 1024, 1024]", view_841: "f32[8, 8, 1024, 1024]", view_869: "f32[8, 8, 1024, 1024]", bmm_105: "f32[64, 1024, 1024]", gt_2: "b8[8, 8, 1024, 1024]", bmm: "f32[64, 1024, 1024]", embedding_1: "f32[1024, 1024, 8]", amax: "f32[8, 8, 1024, 1]", sum_1: "f32[8, 8, 1024, 1]", add_4: "i64[1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:558 in forward, code: scores += position_bias_masked
        add_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_473, view_524);  view_473 = view_524 = None
        add_tensor_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor, view_575);  add_tensor = view_575 = None
        add_tensor_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, view_626);  add_tensor_1 = view_626 = None
        add_tensor_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_2, view_677);  add_tensor_2 = view_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:568 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_81, [8, 8, 1024, 1024]);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:562 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1026 in forward, code: cache_position = torch.arange(
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1268 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.full(
        full_default: "f32[1024, 1025]" = torch.ops.aten.full.default([1024, 1025], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1272 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.triu(causal_mask, diagonal=1)
        iota_default_1: "i64[1025]" = torch.ops.prims.iota.default(1025, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        where_self: "f32[1024, 1025]" = torch.ops.aten.where.self(ge, full_default, full_default_2);  ge = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1273 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        reshape_default_1: "i64[1024, 1]" = torch.ops.aten.reshape.default(iota_default, [-1, 1]);  iota_default = None
        gt_tensor: "b8[1024, 1025]" = torch.ops.aten.gt.Tensor(iota_default_1, reshape_default_1);  iota_default_1 = reshape_default_1 = None
        mul_tensor_2: "f32[1024, 1025]" = torch.ops.aten.mul.Tensor(where_self, gt_tensor);  where_self = gt_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:529 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_12, [8, 8, 1024, 1024]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:465 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_default: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_default: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_default, 0);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:545 in forward, code: position_bias = position_bias[:, :, -seq_length:, :]
        slice_tensor: "f32[1, 8, 1024, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_default, 2, -1024, 9223372036854775807);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:548 in forward, code: causal_mask = mask[:, :, :, : key_states.shape[-2]]
        unsqueeze_default_1: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_2, 0);  mul_tensor_2 = None
        unsqueeze_default_2: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 1);  unsqueeze_default_1 = None
        expand_default: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_2, [8, 1, -1, -1]);  unsqueeze_default_2 = None
        slice_tensor_1: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default, 3, 0, 1024);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:549 in forward, code: position_bias = position_bias + causal_mask
        add_tensor_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:558 in forward, code: scores += position_bias_masked
        add_tensor_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(reshape_default_2, add_tensor_4);  reshape_default_2 = add_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:561 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        sub_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_5, amax_6);  add_tensor_5 = amax_6 = None
        exp_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default, sum_7);  exp_default = sum_7 = None
        mul_tensor_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_tensor);  mul_tensor_1 = None
        sum_dim_int_list: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [-1], True)
        neg_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_3);  neg_default = sum_dim_int_list = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:558 in forward, code: scores += position_bias_masked
        reshape_default_3: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_default, [64, 1024, 1024]);  fma_default = None
        reshape_default_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_3, [8, 8, 1024, 1024]);  reshape_default_3 = None
        reshape_default_5: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_4, [64, 1024, 1024])
        add_tensor_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_4);  add_tensor_3 = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:549 in forward, code: position_bias = position_bias + causal_mask
        sum_dim_int_list_1: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_6, [0], True);  add_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:545 in forward, code: position_bias = position_bias[:, :, -seq_length:, :]
        full_default_3: "f32[1, 8, 1024, 1024]" = torch.ops.aten.full.default([1, 8, 1024, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[1, 8, 1024, 1024]" = torch.ops.aten.slice_scatter.default(full_default_3, sum_dim_int_list_1, 2, -1024, 9223372036854775807);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:465 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_dim: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(slice_scatter_default, 0);  slice_scatter_default = None
        permute_default_1: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_scalar: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_38, -1)
        unsqueeze_default_3: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self_1: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_default_3, full_default_2, permute_default_1);  unsqueeze_default_3 = permute_default_1 = None
        clone_default: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_self_1, memory_format = torch.contiguous_format);  where_self_1 = None
        full_default_4: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 8]" = torch.ops.aten.index_put.default(full_default_4, [add_38], clone_default, True);  add_38 = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:558 in forward, code: scores += position_bias_masked
        sum_dim_int_list_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(view_757, [0], True);  view_757 = None
        sum_dim_int_list_3: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(view_785, [0], True);  view_785 = None
        add_tensor_7: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(sum_dim_int_list_2, sum_dim_int_list_3);  sum_dim_int_list_2 = sum_dim_int_list_3 = None
        sum_dim_int_list_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(view_813, [0], True);  view_813 = None
        add_tensor_8: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_7, sum_dim_int_list_4);  add_tensor_7 = sum_dim_int_list_4 = None
        sum_dim_int_list_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(view_841, [0], True);  view_841 = None
        add_tensor_9: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_8, sum_dim_int_list_5);  add_tensor_8 = sum_dim_int_list_5 = None
        sum_dim_int_list_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(view_869, [0], True);  view_869 = None
        add_tensor_10: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_9, sum_dim_int_list_6);  add_tensor_9 = sum_dim_int_list_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:568 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_105, [8, 8, 1024, 1024]);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:562 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_default_1: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_tensor_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(reshape_default_6, mul_tensor_4);  reshape_default_6 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:529 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm, [8, 8, 1024, 1024]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:465 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_default_2: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_default_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_default_2, 0);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:545 in forward, code: position_bias = position_bias[:, :, -seq_length:, :]
        slice_tensor_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_default_4, 2, -1024, 9223372036854775807);  unsqueeze_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:558 in forward, code: scores += position_bias_masked
        add_tensor_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(reshape_default_7, slice_tensor_2);  reshape_default_7 = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:561 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        sub_tensor_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_11, amax);  add_tensor_11 = amax = None
        exp_default_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        div_tensor_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default_1, sum_1);  exp_default_1 = sum_1 = None
        mul_tensor_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_5, div_tensor_1);  mul_tensor_5 = None
        sum_dim_int_list_7: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [-1], True)
        neg_default_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_tensor_1);  div_tensor_1 = None
        fma_default_1: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_default_1, sum_dim_int_list_7, mul_tensor_6);  neg_default_1 = sum_dim_int_list_7 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:558 in forward, code: scores += position_bias_masked
        reshape_default_8: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_default_1, [64, 1024, 1024]);  fma_default_1 = None
        reshape_default_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_8, [8, 8, 1024, 1024]);  reshape_default_8 = None
        reshape_default_10: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_9, [64, 1024, 1024])
        sum_dim_int_list_8: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(reshape_default_9, [0], True);  reshape_default_9 = None
        add_tensor_12: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_10, sum_dim_int_list_8);  add_tensor_10 = sum_dim_int_list_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:545 in forward, code: position_bias = position_bias[:, :, -seq_length:, :]
        slice_scatter_default_1: "f32[1, 8, 1024, 1024]" = torch.ops.aten.slice_scatter.default(full_default_3, add_tensor_12, 2, -1024, 9223372036854775807);  full_default_3 = add_tensor_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:465 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_dim_1: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(slice_scatter_default_1, 0);  slice_scatter_default_1 = None
        permute_default_3: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:464 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_scalar_1: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_4, -1)
        unsqueeze_default_5: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_2: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_default_5, full_default_2, permute_default_3);  unsqueeze_default_5 = full_default_2 = permute_default_3 = None
        clone_default_1: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_self_2, memory_format = torch.contiguous_format);  where_self_2 = None
        index_put_default_1: "f32[32, 8]" = torch.ops.aten.index_put.default(full_default_4, [add_4], clone_default_1, True);  full_default_4 = add_4 = clone_default_1 = None
        return (reshape_default_5, index_put_default, reshape_default_10, index_put_default_1)


def _default_make_inputs():
    return [
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 8, 1024, 1024], dtype=torch.bool, device='cuda'),
    torch.randint(0, 2, [1024, 1025], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024, 8], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1024, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 8, 1024, 1024], dtype=torch.bool, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024, 8], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1024, 1024], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
