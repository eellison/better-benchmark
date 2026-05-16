"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=var_mean, ranges=[], reduction_ranges=[]
#   origins: ['aten.var_mean.correction']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[50265, 1024]", arg0_1: "i64[8, 1024]", arg2_1: "f32[1026, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:116 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[8, 1024, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:1062 in forward, code: cache_position = torch.arange(
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:101 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_default, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:103 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor: "i64[1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default, 2);  unsqueeze_default = None
        embedding_default_1: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg2_1, add_tensor);  arg2_1 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:1094 in forward, code: hidden_states = inputs_embeds + positions
        add_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1);  mul_tensor = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:1095 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True);  add_tensor_1 = None
        getitem: "f32[8, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:665 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.triu(causal_mask, diagonal=1)
        iota_default_1: "i64[1025]" = torch.ops.prims.iota.default(1025, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[1, 1025]" = torch.ops.aten.unsqueeze.default(iota_default_1, -2);  iota_default_1 = None
        iota_default_2: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_2: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, -1);  iota_default_2 = None
        sub_tensor: "i64[1024, 1025]" = torch.ops.aten.sub.Tensor(unsqueeze_default_1, unsqueeze_default_2);  unsqueeze_default_1 = unsqueeze_default_2 = None
        ge_scalar: "b8[1024, 1025]" = torch.ops.aten.ge.Scalar(sub_tensor, 1);  sub_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:661 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.full(
        full_default: "f32[1024, 1025]" = torch.ops.aten.full.default([1024, 1025], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:665 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.triu(causal_mask, diagonal=1)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1024, 1025]" = torch.ops.aten.where.self(ge_scalar, full_default, full_default_1);  ge_scalar = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:666 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        iota_default_3: "i64[1025]" = torch.ops.prims.iota.default(1025, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default: "i64[1024, 1]" = torch.ops.aten.reshape.default(iota_default, [-1, 1]);  iota_default = None
        gt_tensor: "b8[1024, 1025]" = torch.ops.aten.gt.Tensor(iota_default_3, reshape_default);  iota_default_3 = reshape_default = None
        mul_tensor_1: "f32[1024, 1025]" = torch.ops.aten.mul.Tensor(where_self, gt_tensor);  where_self = gt_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_3: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_4: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 1);  unsqueeze_default_3 = None
        expand_default: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_4, [8, 1, -1, -1]);  unsqueeze_default_4 = None
        slice_tensor: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default, 3, 0, 1024);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default: "f32[8, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor, [0, 8], 0.0);  slice_tensor = None
        slice_tensor_1: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default, -1, 0, 1024);  constant_pad_nd_default = None
        expand_default_1: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_1, [8, 16, 1024, 1024]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_5: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_6: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 1);  unsqueeze_default_5 = None
        expand_default_2: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_6, [8, 1, -1, -1]);  unsqueeze_default_6 = None
        slice_tensor_2: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_2, 3, 0, 1024);  expand_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_1: "f32[8, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_2, [0, 8], 0.0);  slice_tensor_2 = None
        slice_tensor_3: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_1, -1, 0, 1024);  constant_pad_nd_default_1 = None
        expand_default_3: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_3, [8, 16, 1024, 1024]);  slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_7: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_8: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 1);  unsqueeze_default_7 = None
        expand_default_4: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_8, [8, 1, -1, -1]);  unsqueeze_default_8 = None
        slice_tensor_4: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_4, 3, 0, 1024);  expand_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_2: "f32[8, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_4, [0, 8], 0.0);  slice_tensor_4 = None
        slice_tensor_5: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_2, -1, 0, 1024);  constant_pad_nd_default_2 = None
        expand_default_5: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_5, [8, 16, 1024, 1024]);  slice_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_9: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_10: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 1);  unsqueeze_default_9 = None
        expand_default_6: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_10, [8, 1, -1, -1]);  unsqueeze_default_10 = None
        slice_tensor_6: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_6, 3, 0, 1024);  expand_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_3: "f32[8, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_6, [0, 8], 0.0);  slice_tensor_6 = None
        slice_tensor_7: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_3, -1, 0, 1024);  constant_pad_nd_default_3 = None
        expand_default_7: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_7, [8, 16, 1024, 1024]);  slice_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_11: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_12: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 1);  unsqueeze_default_11 = None
        expand_default_8: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_12, [8, 1, -1, -1]);  unsqueeze_default_12 = None
        slice_tensor_8: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_8, 3, 0, 1024);  expand_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_4: "f32[8, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_8, [0, 8], 0.0);  slice_tensor_8 = None
        slice_tensor_9: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_4, -1, 0, 1024);  constant_pad_nd_default_4 = None
        expand_default_9: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_9, [8, 16, 1024, 1024]);  slice_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_13: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_14: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 1);  unsqueeze_default_13 = None
        expand_default_10: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_14, [8, 1, -1, -1]);  unsqueeze_default_14 = None
        slice_tensor_10: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_10, 3, 0, 1024);  expand_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_5: "f32[8, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_10, [0, 8], 0.0);  slice_tensor_10 = None
        slice_tensor_11: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_5, -1, 0, 1024);  constant_pad_nd_default_5 = None
        expand_default_11: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_11, [8, 16, 1024, 1024]);  slice_tensor_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_15: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_16: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 1);  unsqueeze_default_15 = None
        expand_default_12: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_16, [8, 1, -1, -1]);  unsqueeze_default_16 = None
        slice_tensor_12: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_12, 3, 0, 1024);  expand_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_6: "f32[8, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_12, [0, 8], 0.0);  slice_tensor_12 = None
        slice_tensor_13: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_6, -1, 0, 1024);  constant_pad_nd_default_6 = None
        expand_default_13: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_13, [8, 16, 1024, 1024]);  slice_tensor_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_17: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_18: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 1);  unsqueeze_default_17 = None
        expand_default_14: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_18, [8, 1, -1, -1]);  unsqueeze_default_18 = None
        slice_tensor_14: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_14, 3, 0, 1024);  expand_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_7: "f32[8, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_14, [0, 8], 0.0);  slice_tensor_14 = None
        slice_tensor_15: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_7, -1, 0, 1024);  constant_pad_nd_default_7 = None
        expand_default_15: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_15, [8, 16, 1024, 1024]);  slice_tensor_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_19: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_20: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 1);  unsqueeze_default_19 = None
        expand_default_16: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_20, [8, 1, -1, -1]);  unsqueeze_default_20 = None
        slice_tensor_16: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_16, 3, 0, 1024);  expand_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_8: "f32[8, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_16, [0, 8], 0.0);  slice_tensor_16 = None
        slice_tensor_17: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_8, -1, 0, 1024);  constant_pad_nd_default_8 = None
        expand_default_17: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_17, [8, 16, 1024, 1024]);  slice_tensor_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_21: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_22: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 1);  unsqueeze_default_21 = None
        expand_default_18: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_22, [8, 1, -1, -1]);  unsqueeze_default_22 = None
        slice_tensor_18: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_18, 3, 0, 1024);  expand_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_9: "f32[8, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_18, [0, 8], 0.0);  slice_tensor_18 = None
        slice_tensor_19: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_9, -1, 0, 1024);  constant_pad_nd_default_9 = None
        expand_default_19: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_19, [8, 16, 1024, 1024]);  slice_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_23: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_24: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 1);  unsqueeze_default_23 = None
        expand_default_20: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_24, [8, 1, -1, -1]);  unsqueeze_default_24 = None
        slice_tensor_20: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_20, 3, 0, 1024);  expand_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_10: "f32[8, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_20, [0, 8], 0.0);  slice_tensor_20 = None
        slice_tensor_21: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_10, -1, 0, 1024);  constant_pad_nd_default_10 = None
        expand_default_21: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_21, [8, 16, 1024, 1024]);  slice_tensor_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_25: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_26: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 1);  unsqueeze_default_25 = None
        expand_default_22: "f32[8, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_26, [8, 1, -1, -1]);  unsqueeze_default_26 = None
        slice_tensor_22: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_22, 3, 0, 1024);  expand_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_11: "f32[8, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_22, [0, 8], 0.0);  slice_tensor_22 = None
        slice_tensor_23: "f32[8, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_11, -1, 0, 1024);  constant_pad_nd_default_11 = None
        expand_default_23: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_23, [8, 16, 1024, 1024]);  slice_tensor_23 = None
        return (expand_default_1, expand_default_3, expand_default_5, expand_default_7, expand_default_9, expand_default_11, expand_default_13, expand_default_15, expand_default_17, expand_default_19, expand_default_21, expand_default_23, getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([50265, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([1026, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
