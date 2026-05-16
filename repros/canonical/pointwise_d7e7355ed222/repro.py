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
    def forward(self, arg0_1: "i64[512]"):
        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full_default: "b8[512, 1]" = torch.ops.aten.full.default([512, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:379 in sdpa_mask_recent_torch, code: kv_arange = torch.arange(kv_length, device=cache_position.device)
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:380 in sdpa_mask_recent_torch, code: kv_arange += kv_offset
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        sub_tensor: "i64[512]" = torch.ops.aten.sub.Tensor(arg0_1, 128)
        reshape_default: "i64[512, 1]" = torch.ops.aten.reshape.default(sub_tensor, [512, 1]);  sub_tensor = None
        gt_tensor: "b8[512, 512]" = torch.ops.aten.gt.Tensor(add_tensor, reshape_default);  reshape_default = None
        bitwise_and_tensor: "b8[512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, gt_tensor);  full_default = gt_tensor = None
        reshape_default_1: "i64[512, 1]" = torch.ops.aten.reshape.default(arg0_1, [512, 1]);  arg0_1 = None
        le_tensor: "b8[512, 512]" = torch.ops.aten.le.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None
        bitwise_and_tensor_1: "b8[512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, le_tensor);  bitwise_and_tensor = le_tensor = None

        # File: /tmp/pytorch-work/torch/_functorch/vmap.py:204 in _maybe_remove_batch_dim, code: return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        expand_default: "b8[1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_tensor_1, [1, 512, 512]);  bitwise_and_tensor_1 = None
        expand_default_1: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(expand_default, [4, 1, 512, 512]);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:521 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default_1, full_default_1, full_default_2);  expand_default_1 = full_default_1 = full_default_2 = None
        return where_self


def _default_make_inputs():
    return [
    torch.randint(0, 2, [512], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
