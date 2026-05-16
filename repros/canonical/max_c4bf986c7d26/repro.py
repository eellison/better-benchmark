"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=max, ranges=[], reduction_ranges=[]
#   origins: ['aten.max.dim']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm: "f32[256, 512, 512]", iota: "i64[512]", arg10_1: "f32[64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:253 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        reshape_default: "f32[4, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm, [4, 64, 512, 512]);  bmm = None
        mul_tensor: "f32[4, 64, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, 0.125);  reshape_default = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full_default: "b8[512, 1]" = torch.ops.aten.full.default([512, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:379 in sdpa_mask_recent_torch, code: kv_arange = torch.arange(kv_length, device=cache_position.device)
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:380 in sdpa_mask_recent_torch, code: kv_arange += kv_offset
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        sub_tensor: "i64[512]" = torch.ops.aten.sub.Tensor(iota, 128)
        reshape_default_1: "i64[512, 1]" = torch.ops.aten.reshape.default(sub_tensor, [512, 1]);  sub_tensor = None
        gt_tensor: "b8[512, 512]" = torch.ops.aten.gt.Tensor(add_tensor, reshape_default_1);  reshape_default_1 = None
        bitwise_and_tensor: "b8[512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, gt_tensor);  full_default = gt_tensor = None
        reshape_default_2: "i64[512, 1]" = torch.ops.aten.reshape.default(iota, [512, 1]);  iota = None
        le_tensor: "b8[512, 512]" = torch.ops.aten.le.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None
        bitwise_and_tensor_1: "b8[512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, le_tensor);  bitwise_and_tensor = le_tensor = None

        # File: /tmp/pytorch-work/torch/_functorch/vmap.py:204 in _maybe_remove_batch_dim, code: return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        expand_default: "b8[1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_tensor_1, [1, 512, 512]);  bitwise_and_tensor_1 = None
        expand_default_1: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(expand_default, [4, 1, 512, 512]);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:521 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default_1, full_default_1, full_default_2);  expand_default_1 = full_default_1 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:256 in eager_attention_forward, code: attn_weights = attn_weights + causal_mask
        add_tensor_1: "f32[4, 64, 512, 512]" = torch.ops.aten.add.Tensor(mul_tensor, where_self);  mul_tensor = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:258 in eager_attention_forward, code: sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        reshape_default_3: "f32[1, 64, 1, 1]" = torch.ops.aten.reshape.default(arg10_1, [1, -1, 1, 1]);  arg10_1 = None
        expand_default_2: "f32[4, 64, 512, 1]" = torch.ops.aten.expand.default(reshape_default_3, [4, -1, 512, -1]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:259 in eager_attention_forward, code: combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        cat_default: "f32[4, 64, 512, 513]" = torch.ops.aten.cat.default([add_tensor_1, expand_default_2], -1);  add_tensor_1 = expand_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:264 in eager_attention_forward, code: combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        max_dim = torch.ops.aten.max.dim(cat_default, -1, True);  cat_default = None
        getitem: "f32[4, 64, 512, 1]" = max_dim[0];  max_dim = None
        return getitem


def _default_make_inputs():
    return [
    torch.randn([256, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [512], dtype=torch.int64, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
