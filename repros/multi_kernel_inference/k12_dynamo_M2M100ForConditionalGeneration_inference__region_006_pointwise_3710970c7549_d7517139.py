"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config
from torch import device

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:689 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.triu(causal_mask, diagonal=1)
        iota_default: "i64[129]" = torch.ops.prims.iota.default(129, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 129]" = torch.ops.aten.unsqueeze.default(iota_default, -2);  iota_default = None
        iota_default_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, -1);  iota_default_1 = None
        sub_tensor: "i64[128, 129]" = torch.ops.aten.sub.Tensor(unsqueeze_default, unsqueeze_default_1);  unsqueeze_default = unsqueeze_default_1 = None
        ge_scalar: "b8[128, 129]" = torch.ops.aten.ge.Scalar(sub_tensor, 1);  sub_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:685 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.full(
        full_default: "f32[128, 129]" = torch.ops.aten.full.default([128, 129], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:689 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.triu(causal_mask, diagonal=1)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 129]" = torch.ops.aten.where.self(ge_scalar, full_default, full_default_1);  ge_scalar = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:690 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        iota_default_2: "i64[129]" = torch.ops.prims.iota.default(129, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1070 in forward, code: cache_position = torch.arange(
        iota_default_3: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:690 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        reshape_default: "i64[128, 1]" = torch.ops.aten.reshape.default(iota_default_3, [-1, 1]);  iota_default_3 = None
        gt_tensor: "b8[128, 129]" = torch.ops.aten.gt.Tensor(iota_default_2, reshape_default);  iota_default_2 = reshape_default = None
        mul_tensor: "f32[128, 129]" = torch.ops.aten.mul.Tensor(where_self, gt_tensor);  where_self = gt_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_2: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0)
        unsqueeze_default_3: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 1);  unsqueeze_default_2 = None
        expand_default: "f32[64, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_3, [64, 1, -1, -1]);  unsqueeze_default_3 = None
        slice_tensor: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default, 3, 0, 128);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default: "f32[64, 1, 128, 136]" = torch.ops.aten.constant_pad_nd.default(slice_tensor, [0, 8], 0.0);  slice_tensor = None
        slice_tensor_1: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default, -1, 0, 128);  constant_pad_nd_default = None
        expand_default_1: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(slice_tensor_1, [64, 16, 128, 128]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_4: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0)
        unsqueeze_default_5: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 1);  unsqueeze_default_4 = None
        expand_default_2: "f32[64, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_5, [64, 1, -1, -1]);  unsqueeze_default_5 = None
        slice_tensor_2: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default_2, 3, 0, 128);  expand_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_1: "f32[64, 1, 128, 136]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_2, [0, 8], 0.0);  slice_tensor_2 = None
        slice_tensor_3: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_1, -1, 0, 128);  constant_pad_nd_default_1 = None
        expand_default_3: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(slice_tensor_3, [64, 16, 128, 128]);  slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_6: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0)
        unsqueeze_default_7: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 1);  unsqueeze_default_6 = None
        expand_default_4: "f32[64, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_7, [64, 1, -1, -1]);  unsqueeze_default_7 = None
        slice_tensor_4: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default_4, 3, 0, 128);  expand_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_2: "f32[64, 1, 128, 136]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_4, [0, 8], 0.0);  slice_tensor_4 = None
        slice_tensor_5: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_2, -1, 0, 128);  constant_pad_nd_default_2 = None
        expand_default_5: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(slice_tensor_5, [64, 16, 128, 128]);  slice_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_8: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0)
        unsqueeze_default_9: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 1);  unsqueeze_default_8 = None
        expand_default_6: "f32[64, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_9, [64, 1, -1, -1]);  unsqueeze_default_9 = None
        slice_tensor_6: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default_6, 3, 0, 128);  expand_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_3: "f32[64, 1, 128, 136]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_6, [0, 8], 0.0);  slice_tensor_6 = None
        slice_tensor_7: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_3, -1, 0, 128);  constant_pad_nd_default_3 = None
        expand_default_7: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(slice_tensor_7, [64, 16, 128, 128]);  slice_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_10: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0)
        unsqueeze_default_11: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 1);  unsqueeze_default_10 = None
        expand_default_8: "f32[64, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_11, [64, 1, -1, -1]);  unsqueeze_default_11 = None
        slice_tensor_8: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default_8, 3, 0, 128);  expand_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_4: "f32[64, 1, 128, 136]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_8, [0, 8], 0.0);  slice_tensor_8 = None
        slice_tensor_9: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_4, -1, 0, 128);  constant_pad_nd_default_4 = None
        expand_default_9: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(slice_tensor_9, [64, 16, 128, 128]);  slice_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_12: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0)
        unsqueeze_default_13: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 1);  unsqueeze_default_12 = None
        expand_default_10: "f32[64, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_13, [64, 1, -1, -1]);  unsqueeze_default_13 = None
        slice_tensor_10: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default_10, 3, 0, 128);  expand_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_5: "f32[64, 1, 128, 136]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_10, [0, 8], 0.0);  slice_tensor_10 = None
        slice_tensor_11: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_5, -1, 0, 128);  constant_pad_nd_default_5 = None
        expand_default_11: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(slice_tensor_11, [64, 16, 128, 128]);  slice_tensor_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_14: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0)
        unsqueeze_default_15: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 1);  unsqueeze_default_14 = None
        expand_default_12: "f32[64, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_15, [64, 1, -1, -1]);  unsqueeze_default_15 = None
        slice_tensor_12: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default_12, 3, 0, 128);  expand_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_6: "f32[64, 1, 128, 136]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_12, [0, 8], 0.0);  slice_tensor_12 = None
        slice_tensor_13: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_6, -1, 0, 128);  constant_pad_nd_default_6 = None
        expand_default_13: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(slice_tensor_13, [64, 16, 128, 128]);  slice_tensor_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_16: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0)
        unsqueeze_default_17: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 1);  unsqueeze_default_16 = None
        expand_default_14: "f32[64, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_17, [64, 1, -1, -1]);  unsqueeze_default_17 = None
        slice_tensor_14: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default_14, 3, 0, 128);  expand_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_7: "f32[64, 1, 128, 136]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_14, [0, 8], 0.0);  slice_tensor_14 = None
        slice_tensor_15: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_7, -1, 0, 128);  constant_pad_nd_default_7 = None
        expand_default_15: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(slice_tensor_15, [64, 16, 128, 128]);  slice_tensor_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_18: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0)
        unsqueeze_default_19: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 1);  unsqueeze_default_18 = None
        expand_default_16: "f32[64, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_19, [64, 1, -1, -1]);  unsqueeze_default_19 = None
        slice_tensor_16: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default_16, 3, 0, 128);  expand_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_8: "f32[64, 1, 128, 136]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_16, [0, 8], 0.0);  slice_tensor_16 = None
        slice_tensor_17: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_8, -1, 0, 128);  constant_pad_nd_default_8 = None
        expand_default_17: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(slice_tensor_17, [64, 16, 128, 128]);  slice_tensor_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_20: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0)
        unsqueeze_default_21: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 1);  unsqueeze_default_20 = None
        expand_default_18: "f32[64, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_21, [64, 1, -1, -1]);  unsqueeze_default_21 = None
        slice_tensor_18: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default_18, 3, 0, 128);  expand_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_9: "f32[64, 1, 128, 136]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_18, [0, 8], 0.0);  slice_tensor_18 = None
        slice_tensor_19: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_9, -1, 0, 128);  constant_pad_nd_default_9 = None
        expand_default_19: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(slice_tensor_19, [64, 16, 128, 128]);  slice_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_22: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0)
        unsqueeze_default_23: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 1);  unsqueeze_default_22 = None
        expand_default_20: "f32[64, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_23, [64, 1, -1, -1]);  unsqueeze_default_23 = None
        slice_tensor_20: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default_20, 3, 0, 128);  expand_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_10: "f32[64, 1, 128, 136]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_20, [0, 8], 0.0);  slice_tensor_20 = None
        slice_tensor_21: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_10, -1, 0, 128);  constant_pad_nd_default_10 = None
        expand_default_21: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(slice_tensor_21, [64, 16, 128, 128]);  slice_tensor_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_24: "f32[1, 128, 129]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0);  mul_tensor = None
        unsqueeze_default_25: "f32[1, 1, 128, 129]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 1);  unsqueeze_default_24 = None
        expand_default_22: "f32[64, 1, 128, 129]" = torch.ops.aten.expand.default(unsqueeze_default_25, [64, 1, -1, -1]);  unsqueeze_default_25 = None
        slice_tensor_22: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(expand_default_22, 3, 0, 128);  expand_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_11: "f32[64, 1, 128, 136]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_22, [0, 8], 0.0);  slice_tensor_22 = None
        slice_tensor_23: "f32[64, 1, 128, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_11, -1, 0, 128);  constant_pad_nd_default_11 = None
        expand_default_23: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(slice_tensor_23, [64, 16, 128, 128]);  slice_tensor_23 = None
        return (expand_default_1, expand_default_3, expand_default_5, expand_default_7, expand_default_9, expand_default_11, expand_default_13, expand_default_15, expand_default_17, expand_default_19, expand_default_21, expand_default_23)



def make_inputs():
    return [

    ]


def _count_bytes(inputs, outputs):
    """Count total read + write bytes for SOL calculation."""
    total = 0
    for t in inputs:
        if isinstance(t, torch.Tensor):
            total += t.nelement() * t.element_size()
    if isinstance(outputs, torch.Tensor):
        total += outputs.nelement() * outputs.element_size()
    elif isinstance(outputs, (tuple, list)):
        for o in outputs:
            if isinstance(o, torch.Tensor):
                total += o.nelement() * o.element_size()
    return total


def _measure_memcopy_sol(total_bytes, n_warmup=25, n_iter=200):
    """Measure memcopy speed-of-light at multiple transfer sizes.

    Returns dict mapping size label to time in microseconds.
    We measure at the exact kernel size plus a few reference sizes
    so we can see where on the bandwidth curve this kernel sits.
    """
    results = {}
    sizes = set()
    sizes.add(total_bytes)
    # Add reference sizes: 1KB, 64KB, 1MB, 16MB, 64MB, 256MB
    for ref in [1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 256*1024*1024]:
        sizes.add(ref)

    for nbytes in sorted(sizes):
        n_elems = max(nbytes // 4, 1)
        src = torch.empty(n_elems, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)
        for _ in range(n_warmup):
            dst.copy_(src)
        torch.cuda.synchronize()

        import time
        start = time.perf_counter()
        for _ in range(n_iter):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iter

        bw_gbps = (nbytes * 2) / elapsed / 1e9  # read + write
        label = f"{nbytes / 1024:.0f}KB" if nbytes < 1024*1024 else f"{nbytes / 1024 / 1024:.1f}MB"
        is_kernel_size = (nbytes == total_bytes)
        results[nbytes] = {
            "label": label,
            "time_us": elapsed * 1e6,
            "bw_gbps": bw_gbps,
            "is_kernel_size": is_kernel_size,
        }
    return results


def benchmark(n_warmup=25, n_iter=200):
    mod = Repro()
    inputs = make_inputs()

    # Eager baseline
    with torch.no_grad():
        eager_out = mod(*inputs)

    total_bytes = _count_bytes(inputs, eager_out)

    # Compiled (default heuristics)
    compiled = torch.compile(mod)
    with torch.no_grad():
        for _ in range(n_warmup):
            compiled(*inputs)
        torch.cuda.synchronize()

        import time
        start = time.perf_counter()
        for _ in range(n_iter):
            compiled(*inputs)
        torch.cuda.synchronize()
        compiled_time = (time.perf_counter() - start) / n_iter * 1e6

    # Compiled with coordinate descent tuning
    inductor_config.coordinate_descent_tuning = True
    torch._dynamo.reset()
    compiled_cd = torch.compile(mod)
    with torch.no_grad():
        for _ in range(n_warmup):
            compiled_cd(*inputs)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iter):
            compiled_cd(*inputs)
        torch.cuda.synchronize()
        cd_time = (time.perf_counter() - start) / n_iter * 1e6

    # Memcopy SOL at multiple sizes
    sol_results = _measure_memcopy_sol(total_bytes)
    kernel_sol = sol_results[total_bytes]

    print(f"\nKernel data: {total_bytes / 1024:.1f} KB (read+write)")
    print(f"Compiled (default):      {compiled_time:8.1f} us")
    print(f"Compiled (coord descent):{cd_time:8.1f} us")
    print(f"Memcopy SOL (same size): {kernel_sol['time_us']:8.1f} us  ({kernel_sol['bw_gbps']:.1f} GB/s)")
    print(f"Gap (default / SOL):     {compiled_time / kernel_sol['time_us']:8.2f}x")
    print(f"Gap (CD / SOL):          {cd_time / kernel_sol['time_us']:8.2f}x")
    print(f"\nMemcopy bandwidth curve:")
    for nbytes in sorted(sol_results):
        r = sol_results[nbytes]
        marker = " <-- kernel size" if r["is_kernel_size"] else ""
        print(f"  {r['label']:>10s}: {r['time_us']:8.1f} us  {r['bw_gbps']:8.1f} GB/s{marker}")

    return {
        "compiled_us": compiled_time,
        "coord_descent_us": cd_time,
        "memcopy_sol_us": kernel_sol["time_us"],
        "total_bytes": total_bytes,
        "sol_curve": sol_results,
    }


if __name__ == "__main__":
    benchmark()
