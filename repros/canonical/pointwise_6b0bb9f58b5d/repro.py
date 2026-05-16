"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_training
Pattern hash: 6b0bb9f58b5d
Shape hash: bd518b66
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[32, 1280]", _shape_param_0, le: "b8[32, 1280, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:834 in forward_head, code: x = self.flatten(x)
        reshape_default: "f32[32, 1280, 1, 1]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:833 in forward_head, code: x = self.act2(x)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[32, 1280, 1, 1]" = torch.ops.aten.where.self(le, full_default, reshape_default);  le = full_default = reshape_default = None
        return where_self


def _default_make_inputs():
    return [
    torch.randn([32, 1280], dtype=torch.float32, device='cuda'),
    [32, 1280, 1, 1],  # _shape_param_0
    torch.randint(0, 2, [32, 1280, 1, 1], dtype=torch.bool, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
