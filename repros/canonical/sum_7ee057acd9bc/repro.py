"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train_001
Pattern hash: 7ee057acd9bc
Shape hash: 0c4f2a09
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
_shapes_config = "(T([1024, 192, 13, 13], f32), T([1024, 192, 13, 13], i8, gen=Index(9)), T([1024, 192, 27, 27], b8), T([], f32), S([196608, 169]), S([196608, 169]), S([1024, 192, 27, 27]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_6: "f32[1024, 192, 13, 13]", arg12_1: "i8[1024, 192, 13, 13]", arg24_1: "b8[1024, 192, 27, 27]", full: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        full_default: "f32[196608, 729]" = torch.ops.aten.full.default([196608, 729], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default: "f32[196608, 169]" = torch.ops.aten.view.default(getitem_6, _shape_param_0);  getitem_6 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[1024, 192, 13, 13]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg12_1, [3, 3], [27, 27], [2, 2], [0, 0], [1, 1]);  arg12_1 = None
        view_default_1: "i64[196608, 169]" = torch.ops.aten.view.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[196608, 729]" = torch.ops.aten.scatter_add.default(full_default, 1, view_default_1, view_default);  full_default = view_default_1 = view_default = None
        view_default_2: "f32[1024, 192, 27, 27]" = torch.ops.aten.view.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        where_self: "f32[1024, 192, 27, 27]" = torch.ops.aten.where.self(arg24_1, full, view_default_2);  arg24_1 = full = view_default_2 = None
        sum_dim_int_list: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
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
