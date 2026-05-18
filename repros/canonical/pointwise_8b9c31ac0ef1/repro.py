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
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "i64[512]", cumsum: "i64[4, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full_default: "b8[512, 1]" = torch.ops.aten.full.default([512, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:379 in sdpa_mask_recent_torch, code: kv_arange = torch.arange(kv_length, device=cache_position.device)
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:380 in sdpa_mask_recent_torch, code: kv_arange += kv_offset
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default: "i64[512, 1]" = torch.ops.aten.reshape.default(arg1_1, _shape_param_0);  _shape_param_0 = None
        le_tensor: "b8[512, 512]" = torch.ops.aten.le.Tensor(add_tensor, reshape_default);  reshape_default = None
        bitwise_and_tensor: "b8[512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:386 in sdpa_mask_recent_torch, code: batch_arange = torch.arange(batch_size, device=cache_position.device)
        iota_default_1: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:100 in forward, code: return torch.ops.aten.index(x, indices)
        reshape_default_1: "i64[4, 1]" = torch.ops.aten.reshape.default(iota_default_1, _shape_param_1);  _shape_param_1 = None
        index_tensor: "i64[4, 512]" = torch.ops.aten.index.Tensor(cumsum, [reshape_default_1, arg1_1]);  reshape_default_1 = arg1_1 = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default_2: "i64[4, 512, 1]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_2);  index_tensor = _shape_param_2 = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:100 in forward, code: return torch.ops.aten.index(x, indices)
        reshape_default_3: "i64[4, 1]" = torch.ops.aten.reshape.default(iota_default_1, _shape_param_3);  iota_default_1 = _shape_param_3 = None
        index_tensor_1: "i64[4, 512]" = torch.ops.aten.index.Tensor(cumsum, [reshape_default_3, add_tensor]);  cumsum = reshape_default_3 = add_tensor = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default_4: "i64[4, 1, 512]" = torch.ops.aten.reshape.default(index_tensor_1, _shape_param_4);  index_tensor_1 = _shape_param_4 = None
        eq_tensor: "b8[4, 512, 512]" = torch.ops.aten.eq.Tensor(reshape_default_2, reshape_default_4);  reshape_default_2 = reshape_default_4 = None
        bitwise_and_tensor_1: "b8[4, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None

        # File: /tmp/pytorch-work/torch/_functorch/vmap.py:204 in _maybe_remove_batch_dim, code: return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        reshape_default_5: "b8[4, 1, 512, 512]" = torch.ops.aten.reshape.default(bitwise_and_tensor_1, _shape_param_5);  bitwise_and_tensor_1 = _shape_param_5 = None
        expand_default: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(reshape_default_5, _shape_param_6);  reshape_default_5 = _shape_param_6 = None
        return expand_default


def _default_make_inputs():
    return [
    torch.randint(0, 2, [512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [4, 512], dtype=torch.int64, device='cuda'),
    [512, 1],  # _shape_param_0
    [4, 1],  # _shape_param_1
    [4, 512, 1],  # _shape_param_2
    [4, 1],  # _shape_param_3
    [4, 1, 512],  # _shape_param_4
    [4, 1, 512, 512],  # _shape_param_5
    [4, 1, 512, 512],  # _shape_param_6
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
