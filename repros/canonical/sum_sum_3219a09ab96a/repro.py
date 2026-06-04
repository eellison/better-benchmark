"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train_001
Pattern hash: 3219a09ab96a
Shape hash: fdfcb0af
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 256, 13, 13], f32), T([512, 256, 13, 13], i8, gen=Index(9)), T([512, 128, 27, 27], b8), T([], f32), T([512, 128, 27, 27], b8), S([131072, 169]), S([131072, 169]), S([512, 256, 27, 27]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_36: "f32[512, 256, 13, 13]", arg38_1: "i8[512, 256, 13, 13]", arg57_1: "b8[512, 128, 27, 27]", full: "f32[]", arg58_1: "b8[512, 128, 27, 27]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        full_default: "f32[131072, 729]" = torch.ops.aten.full.default([131072, 729], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default: "f32[131072, 169]" = torch.ops.aten.view.default(getitem_36, _shape_param_0);  getitem_36 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[512, 256, 13, 13]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg38_1, [3, 3], [27, 27], [2, 2], [0, 0], [1, 1]);  arg38_1 = None
        view_default_1: "i64[131072, 169]" = torch.ops.aten.view.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[131072, 729]" = torch.ops.aten.scatter_add.default(full_default, 1, view_default_1, view_default);  full_default = view_default_1 = view_default = None
        view_default_2: "f32[512, 256, 27, 27]" = torch.ops.aten.view.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        slice_tensor: "f32[512, 128, 27, 27]" = torch.ops.aten.slice.Tensor(view_default_2, 1, 0, 128)
        slice_tensor_1: "f32[512, 128, 27, 27]" = torch.ops.aten.slice.Tensor(view_default_2, 1, 128, 256);  view_default_2 = None
        where_self: "f32[512, 128, 27, 27]" = torch.ops.aten.where.self(arg57_1, full, slice_tensor_1);  arg57_1 = slice_tensor_1 = None
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
        where_self_1: "f32[512, 128, 27, 27]" = torch.ops.aten.where.self(arg58_1, full, slice_tensor);  arg58_1 = full = slice_tensor = None
        sum_dim_int_list_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3]);  where_self_1 = None
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
