"""
Standalone repro captured via capture_hook.
Label: torchbench_vgg16_train_001
Pattern hash: f4f71b921e96
Shape hash: db593c8d
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 512, 7, 7], f32), T([128, 512, 7, 7], i8, gen=Index(4)), T([128, 512, 14, 14], b8), T([], f32), S([65536, 49]), S([65536, 49]), S([128, 512, 14, 14]))"

class Repro(torch.nn.Module):
    def forward(self, _adaptive_avg_pool2d_backward: "f32[128, 512, 7, 7]", arg34_1: "i8[128, 512, 7, 7]", arg42_1: "b8[128, 512, 14, 14]", full: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        full_default: "f32[65536, 196]" = torch.ops.aten.full.default([65536, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default: "f32[65536, 49]" = torch.ops.aten.view.default(_adaptive_avg_pool2d_backward, _shape_param_0);  _adaptive_avg_pool2d_backward = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[128, 512, 7, 7]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg34_1, [2, 2], [14, 14], [2, 2], [0, 0], [1, 1]);  arg34_1 = None
        view_default_1: "i64[65536, 49]" = torch.ops.aten.view.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[65536, 196]" = torch.ops.aten.scatter_add.default(full_default, 1, view_default_1, view_default);  full_default = view_default_1 = view_default = None
        view_default_2: "f32[128, 512, 14, 14]" = torch.ops.aten.view.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        where_self: "f32[128, 512, 14, 14]" = torch.ops.aten.where.self(arg42_1, full, view_default_2);  arg42_1 = full = view_default_2 = None
        sum_dim_int_list: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
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
