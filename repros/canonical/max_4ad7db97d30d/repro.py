"""
Standalone repro captured via capture_hook.
Label: hf_Reformer_inference
Pattern hash: 4ad7db97d30d
Shape hash: c6ff54f3
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[8, 4096]", arg1_1: "f32[64, 1, 64]", _shape_param_0, arg2_1: "f32[1, 64, 192]", _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:266 in forward, code: max_position_id = position_ids.max().item()
        max_default: "i64[]" = torch.ops.aten.max.default(arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:227 in forward, code: weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights
        expand_default: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        expand_default_1: "f32[8, 64, 64, 192]" = torch.ops.aten.expand.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        return (max_default, expand_default, expand_default_1)


def _default_make_inputs():
    return [
    torch.randint(0, 2, (4096,), dtype=torch.int64, device='cuda').as_strided([8, 4096], [0, 1]),  # arg0_1
    torch.randn([64, 1, 64], dtype=torch.float32, device='cuda'),
    [8, 64, 64, 64],  # _shape_param_0
    torch.randn([1, 64, 192], dtype=torch.float32, device='cuda'),
    [8, 64, 64, 192],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
