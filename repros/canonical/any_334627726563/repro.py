"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=any, ranges=['4', '1', '512', '1'], reduction_ranges=[]
#   origins: ['aten.any.dim']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:604 in forward, code: attention_mask = torch.ones(inputs_embeds.shape[0], seq_length, device=inputs_embeds.device)
        full_default: "f32[4, 512]" = torch.ops.aten.full.default([4, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:483 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.triu(causal_mask, diagonal=1)
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_default, -2);  iota_default = None
        iota_default_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, -1);  iota_default_1 = None
        sub_tensor: "i64[512, 512]" = torch.ops.aten.sub.Tensor(unsqueeze_default, unsqueeze_default_1);  unsqueeze_default = unsqueeze_default_1 = None
        ge_scalar: "b8[512, 512]" = torch.ops.aten.ge.Scalar(sub_tensor, 1);  sub_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:479 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.full(
        full_default_1: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:483 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.triu(causal_mask, diagonal=1)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[512, 512]" = torch.ops.aten.where.self(ge_scalar, full_default_1, full_default_2);  ge_scalar = full_default_1 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:484 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        iota_default_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:598 in forward, code: cache_position = torch.arange(
        iota_default_3: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:484 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        reshape_default: "i64[512, 1]" = torch.ops.aten.reshape.default(iota_default_3, [-1, 1]);  iota_default_3 = None
        gt_tensor: "b8[512, 512]" = torch.ops.aten.gt.Tensor(iota_default_2, reshape_default);  iota_default_2 = reshape_default = None
        mul_tensor: "f32[512, 512]" = torch.ops.aten.mul.Tensor(where_self, gt_tensor);  where_self = gt_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:485 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        unsqueeze_default_2: "f32[1, 512, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor, 0);  mul_tensor = None
        unsqueeze_default_3: "f32[1, 1, 512, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 1);  unsqueeze_default_2 = None
        expand_default: "f32[4, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_default_3, [4, 1, -1, -1]);  unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:487 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        clone_default: "f32[4, 1, 512, 512]" = torch.ops.aten.clone.default(expand_default);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:489 in _prepare_4d_causal_attention_mask_with_cache_position, code: padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
        unsqueeze_default_4: "f32[4, 1, 512]" = torch.ops.aten.unsqueeze.default(full_default, 1);  full_default = None
        unsqueeze_default_5: "f32[4, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        add_tensor: "f32[4, 1, 512, 512]" = torch.ops.aten.add.Tensor(clone_default, unsqueeze_default_5);  unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:492 in _prepare_4d_causal_attention_mask_with_cache_position, code: padding_mask = padding_mask == 0
        eq_scalar: "b8[4, 1, 512, 512]" = torch.ops.aten.eq.Scalar(add_tensor, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:493 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
        full_default_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[4, 1, 512, 512]" = torch.ops.aten.where.self(eq_scalar, full_default_3, clone_default);  eq_scalar = full_default_3 = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:247 in _unmask_unattended, code: return expanded_mask.mul(~torch.all(expanded_mask == min_dtype, dim=-1, keepdim=True))
        eq_scalar_1: "b8[4, 1, 512, 512]" = torch.ops.aten.eq.Scalar(where_self_1, -3.4028234663852886e+38)
        logical_not_default: "b8[4, 1, 512, 512]" = torch.ops.aten.logical_not.default(eq_scalar_1);  eq_scalar_1 = None
        any_dim: "b8[4, 1, 512, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[4, 1, 512, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        bitwise_not_default: "b8[4, 1, 512, 1]" = torch.ops.aten.bitwise_not.default(logical_not_default_1);  logical_not_default_1 = None
        mul_tensor_1: "f32[4, 1, 512, 512]" = torch.ops.aten.mul.Tensor(where_self_1, bitwise_not_default);  where_self_1 = bitwise_not_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_default_1: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_2: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_3: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_4: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_5: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_6: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_7: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_8: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_9: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_10: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_11: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_12: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_13: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_14: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_15: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_16: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_17: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_18: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_19: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_20: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_21: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_22: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_23: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512])
        expand_default_24: "f32[4, 16, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 16, 512, 512]);  mul_tensor_1 = None
        return (expand_default_1, expand_default_2, expand_default_3, expand_default_4, expand_default_5, expand_default_6, expand_default_7, expand_default_8, expand_default_9, expand_default_10, expand_default_11, expand_default_12, expand_default_13, expand_default_14, expand_default_15, expand_default_16, expand_default_17, expand_default_18, expand_default_19, expand_default_20, expand_default_21, expand_default_22, expand_default_23, expand_default_24)


def _default_make_inputs():
    return [

    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
