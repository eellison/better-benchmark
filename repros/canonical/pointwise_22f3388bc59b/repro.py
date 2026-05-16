"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, iota: "i64[1024]", cumsum: "i64[8, 1024]"):
        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full_default: "b8[1024, 1]" = torch.ops.aten.full.default([1024, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:379 in sdpa_mask_recent_torch, code: kv_arange = torch.arange(kv_length, device=cache_position.device)
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:380 in sdpa_mask_recent_torch, code: kv_arange += kv_offset
        add_tensor: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default: "i64[1024, 1]" = torch.ops.aten.reshape.default(iota, [1024, 1])
        le_tensor: "b8[1024, 1024]" = torch.ops.aten.le.Tensor(add_tensor, reshape_default);  reshape_default = None
        bitwise_and_tensor: "b8[1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:386 in sdpa_mask_recent_torch, code: batch_arange = torch.arange(batch_size, device=cache_position.device)
        iota_default_1: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:100 in forward, code: return torch.ops.aten.index(x, indices)
        reshape_default_1: "i64[8, 1]" = torch.ops.aten.reshape.default(iota_default_1, [8, 1])
        index_tensor: "i64[8, 1024]" = torch.ops.aten.index.Tensor(cumsum, [reshape_default_1, iota]);  reshape_default_1 = iota = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default_2: "i64[8, 1024, 1]" = torch.ops.aten.reshape.default(index_tensor, [8, 1024, 1]);  index_tensor = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:100 in forward, code: return torch.ops.aten.index(x, indices)
        reshape_default_3: "i64[8, 1]" = torch.ops.aten.reshape.default(iota_default_1, [8, 1]);  iota_default_1 = None
        index_tensor_1: "i64[8, 1024]" = torch.ops.aten.index.Tensor(cumsum, [reshape_default_3, add_tensor]);  cumsum = reshape_default_3 = add_tensor = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default_4: "i64[8, 1, 1024]" = torch.ops.aten.reshape.default(index_tensor_1, [8, 1, 1024]);  index_tensor_1 = None
        eq_tensor: "b8[8, 1024, 1024]" = torch.ops.aten.eq.Tensor(reshape_default_2, reshape_default_4);  reshape_default_2 = reshape_default_4 = None
        bitwise_and_tensor_1: "b8[8, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None

        # File: /tmp/pytorch-work/torch/_functorch/vmap.py:204 in _maybe_remove_batch_dim, code: return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        reshape_default_5: "b8[8, 1, 1024, 1024]" = torch.ops.aten.reshape.default(bitwise_and_tensor_1, [8, 1, 1024, 1024]);  bitwise_and_tensor_1 = None
        expand_default: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(reshape_default_5, [8, 1, 1024, 1024]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default_2);  full_default_1 = full_default_2 = None
        expand_default_1: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self, [8, 12, 1024, 1024]);  where_self = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_3, full_default_4);  full_default_3 = full_default_4 = None
        expand_default_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_1, [8, 12, 1024, 1024]);  where_self_1 = None
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_5, full_default_6);  full_default_5 = full_default_6 = None
        expand_default_3: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_2, [8, 12, 1024, 1024]);  where_self_2 = None
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_3: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_7, full_default_8);  full_default_7 = full_default_8 = None
        expand_default_4: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_3, [8, 12, 1024, 1024]);  where_self_3 = None
        full_default_9: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_4: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_9, full_default_10);  full_default_9 = full_default_10 = None
        expand_default_5: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_4, [8, 12, 1024, 1024]);  where_self_4 = None
        full_default_11: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_5: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_11, full_default_12);  full_default_11 = full_default_12 = None
        expand_default_6: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_5, [8, 12, 1024, 1024]);  where_self_5 = None
        full_default_13: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_6: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_13, full_default_14);  full_default_13 = full_default_14 = None
        expand_default_7: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_6, [8, 12, 1024, 1024]);  where_self_6 = None
        full_default_15: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_7: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_15, full_default_16);  full_default_15 = full_default_16 = None
        expand_default_8: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_7, [8, 12, 1024, 1024]);  where_self_7 = None
        full_default_17: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_8: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_17, full_default_18);  full_default_17 = full_default_18 = None
        expand_default_9: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_8, [8, 12, 1024, 1024]);  where_self_8 = None
        full_default_19: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_9: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_19, full_default_20);  full_default_19 = full_default_20 = None
        expand_default_10: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_9, [8, 12, 1024, 1024]);  where_self_9 = None
        full_default_21: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_10: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_21, full_default_22);  full_default_21 = full_default_22 = None
        expand_default_11: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_10, [8, 12, 1024, 1024]);  where_self_10 = None
        full_default_23: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_11: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_23, full_default_24);  expand_default = full_default_23 = full_default_24 = None
        expand_default_12: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_11, [8, 12, 1024, 1024]);  where_self_11 = None
        return (expand_default_1, expand_default_2, expand_default_3, expand_default_4, expand_default_5, expand_default_6, expand_default_7, expand_default_8, expand_default_9, expand_default_10, expand_default_11, expand_default_12)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [1024], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [8, 1024], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
