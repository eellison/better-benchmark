"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train_001
Pattern hash: 8bcd6e12dcd4
Shape hash: c1f46ce0
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
_shapes_config = "(T([512, 128, 27, 27], f32), T([512, 128, 27, 27], i8, gen=Index(9)), T([512, 64, 55, 55], b8), T([], f32), T([512, 64, 55, 55], b8), S([65536, 729]), S([65536, 729]), S([512, 128, 55, 55]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_54: "f32[512, 128, 27, 27]", arg33_1: "i8[512, 128, 27, 27]", arg61_1: "b8[512, 64, 55, 55]", full: "f32[]", arg62_1: "b8[512, 64, 55, 55]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        full_default: "f32[65536, 3025]" = torch.ops.aten.full.default([65536, 3025], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default: "f32[65536, 729]" = torch.ops.aten.view.default(getitem_54, _shape_param_0);  getitem_54 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[512, 128, 27, 27]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg33_1, [3, 3], [55, 55], [2, 2], [0, 0], [1, 1]);  arg33_1 = None
        view_default_1: "i64[65536, 729]" = torch.ops.aten.view.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[65536, 3025]" = torch.ops.aten.scatter_add.default(full_default, 1, view_default_1, view_default);  full_default = view_default_1 = view_default = None
        view_default_2: "f32[512, 128, 55, 55]" = torch.ops.aten.view.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        slice_tensor: "f32[512, 64, 55, 55]" = torch.ops.aten.slice.Tensor(view_default_2, 1, 0, 64)
        slice_tensor_1: "f32[512, 64, 55, 55]" = torch.ops.aten.slice.Tensor(view_default_2, 1, 64, 128);  view_default_2 = None
        where_self: "f32[512, 64, 55, 55]" = torch.ops.aten.where.self(arg61_1, full, slice_tensor_1);  arg61_1 = slice_tensor_1 = None
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
        where_self_1: "f32[512, 64, 55, 55]" = torch.ops.aten.where.self(arg62_1, full, slice_tensor);  arg62_1 = full = slice_tensor = None
        sum_dim_int_list_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3]);  where_self_1 = None
        return (sum_dim_int_list, sum_dim_int_list_1)



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
