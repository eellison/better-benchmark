"""
Standalone repro captured via capture_hook.
Label: falcon_rw_1b
Pattern hash: f75f7ac37fa0
Shape hash: e841b48c
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, _tensor_constant1: "b8[]", mm_96: "f16[2048, 50304]", _shape_param_0):
        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full_default: "b8[512]" = torch.ops.aten.full.default([512], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "b8[512]" = torch.ops.aten.full.default([512], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        lift_fresh_copy_default: "b8[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:906 in forward, code: lm_logits = self.lm_head(hidden_states[:, slice_indices, :])
        reshape_default: "f16[4, 512, 50304]" = torch.ops.aten.reshape.default(mm_96, _shape_param_0);  mm_96 = _shape_param_0 = None
        return (full_default, full_default_1, lift_fresh_copy_default, reshape_default)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [], dtype=torch.bool, device='cpu'),
    torch.randn([2048, 50304], dtype=torch.float16, device='cuda'),
    [4, 512, 50304],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
