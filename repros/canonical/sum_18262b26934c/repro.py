"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train_001
Pattern hash: 18262b26934c
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

_repro_version = 2
_shapes_config = "(T([512, 64, 55, 55], f32), T([512, 64, 55, 55], i8, gen=Index(9)), T([512, 64, 111, 111], b8), T([], f32), S([32768, 3025]), S([32768, 3025]), S([512, 64, 111, 111]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_72: "f32[512, 64, 55, 55]", arg28_1: "i8[512, 64, 55, 55]", arg65_1: "b8[512, 64, 111, 111]", full: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        full_default: "f32[32768, 12321]" = torch.ops.aten.full.default([32768, 12321], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default: "f32[32768, 3025]" = torch.ops.aten.view.default(getitem_72, _shape_param_0);  getitem_72 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[512, 64, 55, 55]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg28_1, [3, 3], [111, 111], [2, 2], [0, 0], [1, 1]);  arg28_1 = None
        view_default_1: "i64[32768, 3025]" = torch.ops.aten.view.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[32768, 12321]" = torch.ops.aten.scatter_add.default(full_default, 1, view_default_1, view_default);  full_default = view_default_1 = view_default = None
        view_default_2: "f32[512, 64, 111, 111]" = torch.ops.aten.view.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        where_self: "f32[512, 64, 111, 111]" = torch.ops.aten.where.self(arg65_1, full, view_default_2);  arg65_1 = full = view_default_2 = None
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
        return sum_dim_int_list



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
