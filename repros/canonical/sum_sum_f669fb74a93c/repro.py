"""
Standalone repro captured via capture_hook.
Label: torchbench_dlrm_train_001
Pattern hash: f669fb74a93c
Shape hash: 2de803a0
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 1], b8), T([2048, 1], f32), T([1, 1024], f32), T([2048, 1024], f32), S([1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg52_1: "b8[2048, 1]", arg54_1: "f32[2048, 1]", arg13_1: "f32[1, 1024]", arg51_1: "f32[2048, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[2048, 1]" = torch.ops.aten.where.self(arg52_1, full_default, arg54_1);  arg52_1 = arg54_1 = None
        permute_default: "f32[1024, 1]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        permute_default_1: "f32[1, 1024]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        mul_tensor: "f32[2048, 1024]" = torch.ops.aten.mul.Tensor(where_self, permute_default_1);  permute_default_1 = None
        permute_default_2: "f32[1, 2048]" = torch.ops.aten.permute.default(where_self, [1, 0])
        sum_dim_int_list: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(where_self, [0], True);  where_self = None
        view_default: "f32[1]" = torch.ops.aten.view.default(sum_dim_int_list, [1]);  sum_dim_int_list = None
        le_scalar: "b8[2048, 1024]" = torch.ops.aten.le.Scalar(arg51_1, 0);  arg51_1 = None
        where_self_1: "f32[2048, 1024]" = torch.ops.aten.where.self(le_scalar, full_default, mul_tensor);  le_scalar = full_default = mul_tensor = None
        permute_default_3: "f32[1024, 2048]" = torch.ops.aten.permute.default(where_self_1, [1, 0])
        sum_dim_int_list_1: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0], True);  where_self_1 = None
        view_default_1: "f32[1024]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_0);  sum_dim_int_list_1 = _shape_param_0 = None
        return (permute_default_2, view_default, permute_default_3, view_default_1)

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
