"""
Standalone repro captured via capture_hook.
Label: vgg16_training
Pattern hash: 5c0ff7e50403
Shape hash: cbdb1860
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
    def forward(self, getitem_40: "f32[4, 64, 112, 112]", _shape_param_0, getitem_1: "i8[4, 64, 112, 112]", _shape_param_1, _shape_param_2, le_13: "b8[4, 64, 224, 224]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        full_default: "f32[256, 50176]" = torch.ops.aten.full.default([256, 50176], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        reshape_default: "f32[256, 12544]" = torch.ops.aten.reshape.default(getitem_40, _shape_param_0);  getitem_40 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[4, 64, 112, 112]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_1, [2, 2], [224, 224], [2, 2], [0, 0], [1, 1]);  getitem_1 = None
        reshape_default_1: "i64[256, 12544]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[256, 50176]" = torch.ops.aten.scatter_add.default(full_default, 1, reshape_default_1, reshape_default);  full_default = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[4, 64, 224, 224]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        full_default_1 = full_default
        where_self: "f32[4, 64, 224, 224]" = torch.ops.aten.where.self(le_13, full_default_1, reshape_default_2);  le_13 = full_default_1 = reshape_default_2 = None
        return where_self


def _default_make_inputs():
    return [
    torch.randn(3211264, dtype=torch.float32, device='cuda').as_strided([4, 64, 112, 112], [802816, 1, 7168, 64]),  # getitem_40
    [256, 12544],  # _shape_param_0
    torch.randint(0, 2, (3211264,), dtype=torch.int8, device='cuda').as_strided([4, 64, 112, 112], [802816, 1, 7168, 64]),  # getitem_1
    [256, 12544],  # _shape_param_1
    [4, 64, 224, 224],  # _shape_param_2
    torch.randn(12845056, dtype=torch.bool, device='cuda').as_strided([4, 64, 224, 224], [3211264, 1, 14336, 64]),  # le_13
    torch.tensor(1),  # full_default_1 (unknown shape)
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
