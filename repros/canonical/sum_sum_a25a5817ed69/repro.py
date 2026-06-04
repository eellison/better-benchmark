"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train_001
Pattern hash: a25a5817ed69
Shape hash: 20fa88dc
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 120, 28, 28], f32), T([512, 120, 28, 28], f32), T([512, 120, 1, 1], f32), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_195: "f32[512, 120, 28, 28]", arg277_1: "f32[512, 120, 28, 28]", arg280_1: "f32[512, 120, 1, 1]", full: "f32[]"):
        # No stacktrace found for following nodes
        mul_tensor: "f32[512, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_195, arg277_1);  getitem_195 = arg277_1 = None
        sum_dim_int_list: "f32[512, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2, 3], True);  mul_tensor = None
        gt_scalar: "b8[512, 120, 1, 1]" = torch.ops.aten.gt.Scalar(arg280_1, -3.0)
        lt_scalar: "b8[512, 120, 1, 1]" = torch.ops.aten.lt.Scalar(arg280_1, 3.0);  arg280_1 = None
        bitwise_and_tensor: "b8[512, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_1: "f32[512, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[512, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_1, full);  bitwise_and_tensor = mul_tensor_1 = full = None
        sum_dim_int_list_1: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
        return sum_dim_int_list_1

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
