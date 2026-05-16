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
    def forward(self, iota: "i64[512]", cumsum: "i64[4, 512]"):
        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full_default: "b8[512, 1]" = torch.ops.aten.full.default([512, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:379 in sdpa_mask_recent_torch, code: kv_arange = torch.arange(kv_length, device=cache_position.device)
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:380 in sdpa_mask_recent_torch, code: kv_arange += kv_offset
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default: "i64[512, 1]" = torch.ops.aten.reshape.default(iota, [512, 1])
        le_tensor: "b8[512, 512]" = torch.ops.aten.le.Tensor(add_tensor, reshape_default);  reshape_default = None
        bitwise_and_tensor: "b8[512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:386 in sdpa_mask_recent_torch, code: batch_arange = torch.arange(batch_size, device=cache_position.device)
        iota_default_1: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:100 in forward, code: return torch.ops.aten.index(x, indices)
        reshape_default_1: "i64[4, 1]" = torch.ops.aten.reshape.default(iota_default_1, [4, 1])
        index_tensor: "i64[4, 512]" = torch.ops.aten.index.Tensor(cumsum, [reshape_default_1, iota]);  reshape_default_1 = iota = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default_2: "i64[4, 512, 1]" = torch.ops.aten.reshape.default(index_tensor, [4, 512, 1]);  index_tensor = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:100 in forward, code: return torch.ops.aten.index(x, indices)
        reshape_default_3: "i64[4, 1]" = torch.ops.aten.reshape.default(iota_default_1, [4, 1]);  iota_default_1 = None
        index_tensor_1: "i64[4, 512]" = torch.ops.aten.index.Tensor(cumsum, [reshape_default_3, add_tensor]);  cumsum = reshape_default_3 = add_tensor = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default_4: "i64[4, 1, 512]" = torch.ops.aten.reshape.default(index_tensor_1, [4, 1, 512]);  index_tensor_1 = None
        eq_tensor: "b8[4, 512, 512]" = torch.ops.aten.eq.Tensor(reshape_default_2, reshape_default_4);  reshape_default_2 = reshape_default_4 = None
        bitwise_and_tensor_1: "b8[4, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None

        # File: /tmp/pytorch-work/torch/_functorch/vmap.py:204 in _maybe_remove_batch_dim, code: return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        reshape_default_5: "b8[4, 1, 512, 512]" = torch.ops.aten.reshape.default(bitwise_and_tensor_1, [4, 1, 512, 512]);  bitwise_and_tensor_1 = None
        expand_default: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(reshape_default_5, [4, 1, 512, 512]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default_2);  full_default_1 = full_default_2 = None
        full_default_3: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_3, full_default_4);  full_default_3 = full_default_4 = None
        full_default_5: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_5, full_default_6);  full_default_5 = full_default_6 = None
        full_default_7: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_3: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_7, full_default_8);  full_default_7 = full_default_8 = None
        full_default_9: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_4: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_9, full_default_10);  full_default_9 = full_default_10 = None
        full_default_11: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_5: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_11, full_default_12);  full_default_11 = full_default_12 = None
        full_default_13: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_6: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_13, full_default_14);  full_default_13 = full_default_14 = None
        full_default_15: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_7: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_15, full_default_16);  full_default_15 = full_default_16 = None
        full_default_17: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_8: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_17, full_default_18);  full_default_17 = full_default_18 = None
        full_default_19: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_9: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_19, full_default_20);  full_default_19 = full_default_20 = None
        full_default_21: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_10: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_21, full_default_22);  full_default_21 = full_default_22 = None
        full_default_23: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_11: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_23, full_default_24);  full_default_23 = full_default_24 = None
        full_default_25: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_12: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_25, full_default_26);  full_default_25 = full_default_26 = None
        full_default_27: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_13: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_27, full_default_28);  full_default_27 = full_default_28 = None
        full_default_29: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_14: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_29, full_default_30);  full_default_29 = full_default_30 = None
        full_default_31: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_15: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_31, full_default_32);  full_default_31 = full_default_32 = None
        full_default_33: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_34: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_16: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_33, full_default_34);  full_default_33 = full_default_34 = None
        full_default_35: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_17: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_35, full_default_36);  full_default_35 = full_default_36 = None
        full_default_37: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_18: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_37, full_default_38);  full_default_37 = full_default_38 = None
        full_default_39: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_19: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_39, full_default_40);  full_default_39 = full_default_40 = None
        full_default_41: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_20: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_41, full_default_42);  full_default_41 = full_default_42 = None
        full_default_43: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_21: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_43, full_default_44);  full_default_43 = full_default_44 = None
        full_default_45: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_22: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_45, full_default_46);  full_default_45 = full_default_46 = None
        full_default_47: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_23: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_47, full_default_48);  full_default_47 = full_default_48 = None
        full_default_49: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_50: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_24: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_49, full_default_50);  full_default_49 = full_default_50 = None
        full_default_51: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_25: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_51, full_default_52);  full_default_51 = full_default_52 = None
        full_default_53: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_26: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_53, full_default_54);  full_default_53 = full_default_54 = None
        full_default_55: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_27: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_55, full_default_56);  expand_default = full_default_55 = full_default_56 = None
        return (where_self, where_self_1, where_self_2, where_self_3, where_self_4, where_self_5, where_self_6, where_self_7, where_self_8, where_self_9, where_self_10, where_self_11, where_self_12, where_self_13, where_self_14, where_self_15, where_self_16, where_self_17, where_self_18, where_self_19, where_self_20, where_self_21, where_self_22, where_self_23, where_self_24, where_self_25, where_self_26, where_self_27)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [4, 512], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
