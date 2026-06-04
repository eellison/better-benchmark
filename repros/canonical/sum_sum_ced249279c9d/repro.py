"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train_001
Pattern hash: ced249279c9d
Shape hash: ed34fe42
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 128, 55, 55], f32), T([512, 64, 55, 55], b8), T([], f32), T([512, 64, 55, 55], b8))"

class Repro(torch.nn.Module):
    def forward(self, getitem_63: "f32[512, 128, 55, 55]", arg63_1: "b8[512, 64, 55, 55]", full: "f32[]", arg64_1: "b8[512, 64, 55, 55]"):
        # No stacktrace found for following nodes
        slice_tensor: "f32[512, 64, 55, 55]" = torch.ops.aten.slice.Tensor(getitem_63, 1, 0, 64)
        slice_tensor_1: "f32[512, 64, 55, 55]" = torch.ops.aten.slice.Tensor(getitem_63, 1, 64, 128);  getitem_63 = None
        where_self: "f32[512, 64, 55, 55]" = torch.ops.aten.where.self(arg63_1, full, slice_tensor_1);  arg63_1 = slice_tensor_1 = None
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
        where_self_1: "f32[512, 64, 55, 55]" = torch.ops.aten.where.self(arg64_1, full, slice_tensor);  arg64_1 = full = slice_tensor = None
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
