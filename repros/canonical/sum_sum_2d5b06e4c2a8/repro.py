"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train_001
Pattern hash: 2d5b06e4c2a8
Shape hash: b88dd92d
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
_shapes_config = "(T([512, 72, 28, 28], f32), T([1, 72, 1, 1], f32), T([1, 72, 1, 1], f32), T([72], f32), T([72], f32), T([512, 72, 28, 28], f32), T([512, 72, 1, 1], f32), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg165_1: "f32[512, 72, 28, 28]", arg166_1: "f32[1, 72, 1, 1]", arg167_1: "f32[1, 72, 1, 1]", arg23_1: "f32[72]", arg24_1: "f32[72]", getitem_147: "f32[512, 72, 28, 28]", arg170_1: "f32[512, 72, 1, 1]", full: "f32[]"):
        # No stacktrace found for following nodes
        sub_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(arg165_1, arg166_1);  arg165_1 = arg166_1 = None
        mul_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, arg167_1);  sub_tensor = arg167_1 = None
        unsqueeze_default: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_default_1: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_default_3: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[512, 72, 28, 28]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        mul_tensor_2: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_147, relu_default);  getitem_147 = relu_default = None
        sum_dim_int_list: "f32[512, 72, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True);  mul_tensor_2 = None
        gt_scalar: "b8[512, 72, 1, 1]" = torch.ops.aten.gt.Scalar(arg170_1, -3.0)
        lt_scalar: "b8[512, 72, 1, 1]" = torch.ops.aten.lt.Scalar(arg170_1, 3.0);  arg170_1 = None
        bitwise_and_tensor: "b8[512, 72, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_3: "f32[512, 72, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[512, 72, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_3, full);  bitwise_and_tensor = mul_tensor_3 = full = None
        sum_dim_int_list_1: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
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
