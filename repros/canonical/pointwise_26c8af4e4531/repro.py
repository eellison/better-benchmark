"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_inference
Pattern hash: 26c8af4e4531
Shape hash: 5a695a43
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_21: "f32[32, 6, 49, 128]", _shape_param_0, _shape_param_1, getitem_22: "f32[32, 6, 49, 128]", _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_default: "f32[32, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_21, _shape_param_0);  getitem_21 = _shape_param_0 = None
        clone_default: "f32[32, 6, 49, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default: "f32[192, 49, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        permute_default: "f32[32, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_22, [0, 1, 3, 2]);  getitem_22 = None
        expand_default_1: "f32[32, 6, 128, 49]" = torch.ops.aten.expand.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None
        clone_default_1: "f32[32, 6, 128, 49]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_1: "f32[192, 128, 49]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None
        return (reshape_default, reshape_default_1)


def _default_make_inputs():
    return [
    torch.randn(3537408, dtype=torch.float32, device='cuda').as_strided([32, 6, 49, 128], [112896, 6272, 1, 49]),  # getitem_21
    [32, 6, 49, 128],  # _shape_param_0
    [192, 49, 128],  # _shape_param_1
    torch.randn(3537408, dtype=torch.float32, device='cuda').as_strided([32, 6, 49, 128], [112896, 6272, 1, 49]),  # getitem_22
    [32, 6, 128, 49],  # _shape_param_2
    [192, 128, 49],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
