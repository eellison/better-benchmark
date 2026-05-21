"""
Standalone repro captured via capture_hook.
Label: torchbench_vgg16_train
Pattern hash: adf6fadb8efb
Shape hash: db593c8d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, _adaptive_avg_pool2d_backward: "f32[128, 512, 7, 7]", getitem_9: "i8[128, 512, 7, 7]", le_2: "b8[128, 512, 14, 14]", full_default: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        full_default_1: "f32[65536, 196]" = torch.ops.aten.full.default([65536, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        reshape_default: "f32[65536, 49]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d_backward, _shape_param_0);  _adaptive_avg_pool2d_backward = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[128, 512, 7, 7]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_9, [2, 2], [14, 14], [2, 2], [0, 0], [1, 1]);  getitem_9 = None
        reshape_default_1: "i64[65536, 49]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[65536, 196]" = torch.ops.aten.scatter_add.default(full_default_1, 1, reshape_default_1, reshape_default);  full_default_1 = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[128, 512, 14, 14]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        where_self: "f32[128, 512, 14, 14]" = torch.ops.aten.where.self(le_2, full_default, reshape_default_2);  le_2 = full_default = reshape_default_2 = None
        return where_self


def _default_make_inputs():
    return [
    torch.randn([128, 512, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randint(0, 4, [128, 512, 7, 7], dtype=torch.int8, device='cuda'),
    torch.randint(0, 2, [128, 512, 14, 14], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    [65536, 49],  # _shape_param_0
    [65536, 49],  # _shape_param_1
    [128, 512, 14, 14],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
