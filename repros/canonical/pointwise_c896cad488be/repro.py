"""
Standalone repro captured via capture_hook.
Label: torchbench_vgg16_train
Pattern hash: c896cad488be
Shape hash: 85b70ad5
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
_shapes_config = "(T([128, 512, 14, 14], f32, stride=(100352, 1, 7168, 512)), T([128, 512, 14, 14], i8, stride=(100352, 1, 7168, 512), gen=Index(4)), T([128, 512, 28, 28], b8, stride=(401408, 1, 14336, 512)), T([], f32), S([65536, 196]), S([65536, 196]), S([128, 512, 28, 28]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_16: "f32[128, 512, 14, 14]", getitem_7: "i8[128, 512, 14, 14]", le_5: "b8[128, 512, 28, 28]", full_default: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        full_default_1: "f32[65536, 784]" = torch.ops.aten.full.default([65536, 784], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        reshape_default: "f32[65536, 196]" = torch.ops.aten.reshape.default(getitem_16, _shape_param_0);  getitem_16 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[128, 512, 14, 14]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_7, [2, 2], [28, 28], [2, 2], [0, 0], [1, 1]);  getitem_7 = None
        reshape_default_1: "i64[65536, 196]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[65536, 784]" = torch.ops.aten.scatter_add.default(full_default_1, 1, reshape_default_1, reshape_default);  full_default_1 = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[128, 512, 28, 28]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        where_self: "f32[128, 512, 28, 28]" = torch.ops.aten.where.self(le_5, full_default, reshape_default_2);  le_5 = full_default = reshape_default_2 = None
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
