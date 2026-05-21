"""
Standalone repro captured via capture_hook.
Label: torchbench_vgg16_train
Pattern hash: 86b5e5e44a3e
Shape hash: d3ba6265
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 256, 28, 28], f32, stride=(200704, 1, 7168, 256)), T([128, 256, 28, 28], i8, stride=(200704, 1, 7168, 256), gen=Index(4)), T([128, 256, 56, 56], b8, stride=(802816, 1, 14336, 256)), T([], f32), S([32768, 784]), S([32768, 784]), S([128, 256, 56, 56]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_25: "f32[128, 256, 28, 28]", getitem_5: "i8[128, 256, 28, 28]", le_8: "b8[128, 256, 56, 56]", full_default: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        full_default_1: "f32[32768, 3136]" = torch.ops.aten.full.default([32768, 3136], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        reshape_default: "f32[32768, 784]" = torch.ops.aten.reshape.default(getitem_25, _shape_param_0);  getitem_25 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[128, 256, 28, 28]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_5, [2, 2], [56, 56], [2, 2], [0, 0], [1, 1]);  getitem_5 = None
        reshape_default_1: "i64[32768, 784]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[32768, 3136]" = torch.ops.aten.scatter_add.default(full_default_1, 1, reshape_default_1, reshape_default);  full_default_1 = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[128, 256, 56, 56]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        where_self: "f32[128, 256, 56, 56]" = torch.ops.aten.where.self(le_8, full_default, reshape_default_2);  le_8 = full_default = reshape_default_2 = None
        return where_self



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
