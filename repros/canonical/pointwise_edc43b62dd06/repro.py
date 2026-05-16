"""
Standalone repro captured via capture_hook.
Label: vit_b_16_training
Pattern hash: edc43b62dd06
Shape hash: e06247f6
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_90: "f32[4, 12, 197, 64]", _shape_param_0, primals_142: "f32[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default: "f32[197, 4, 12, 64]" = torch.ops.aten.permute.default(getitem_90, [2, 0, 1, 3]);  getitem_90 = None
        clone_default: "f32[197, 4, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default: "f32[788, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        return (reshape_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_90
    [788, 768],  # _shape_param_0
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
