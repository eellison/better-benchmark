"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train
Pattern hash: ddc049989f0a
Shape hash: 690a3b7d
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
    def forward(self, getitem_78: "f32[512, 64, 55, 55]", getitem_1: "i8[512, 64, 55, 55]", le_25: "b8[512, 64, 111, 111]", full_default: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        full_default_1: "f32[32768, 12321]" = torch.ops.aten.full.default([32768, 12321], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        reshape_default: "f32[32768, 3025]" = torch.ops.aten.reshape.default(getitem_78, _shape_param_0);  getitem_78 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[512, 64, 55, 55]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_1, [3, 3], [111, 111], [2, 2], [0, 0], [1, 1]);  getitem_1 = None
        reshape_default_1: "i64[32768, 3025]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[32768, 12321]" = torch.ops.aten.scatter_add.default(full_default_1, 1, reshape_default_1, reshape_default);  full_default_1 = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[512, 64, 111, 111]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        where_self: "f32[512, 64, 111, 111]" = torch.ops.aten.where.self(le_25, full_default, reshape_default_2);  le_25 = full_default = reshape_default_2 = None
        return where_self


def _default_make_inputs():
    return [
    torch.randn(99123200, dtype=torch.float32, device='cuda').as_strided([512, 64, 55, 55], [193600, 1, 3520, 64]),  # getitem_78
    torch.randint(0, 9, (99123200,), dtype=torch.int8, device='cuda').as_strided([512, 64, 55, 55], [193600, 1, 3520, 64]),  # getitem_1
    torch.randint(0, 2, (403734528,), dtype=torch.bool, device='cuda').as_strided([512, 64, 111, 111], [788544, 1, 7104, 64]),  # le_25
    torch.randn([], dtype=torch.float32, device='cuda'),
    [32768, 3025],  # _shape_param_0
    [32768, 3025],  # _shape_param_1
    [512, 64, 111, 111],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
