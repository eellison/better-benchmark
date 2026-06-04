"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train_001
Pattern hash: b16afac198fb
Shape hash: ccc1f6d9
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 256, 56, 56], f32), T([], f32), T([64, 256, 56, 56], f32), T([64, 256, 56, 56], f32), T([1, 256, 1, 1], f32), T([256], f32), T([256], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg286_1: "f32[64, 256, 56, 56]", full: "f32[]", getitem_318: "f32[64, 256, 56, 56]", arg284_1: "f32[64, 256, 56, 56]", arg718_1: "f32[1, 256, 1, 1]", arg285_1: "f32[256]", arg28_1: "f32[256]"):
        # No stacktrace found for following nodes
        le_scalar: "b8[64, 256, 56, 56]" = torch.ops.aten.le.Scalar(arg286_1, 0);  arg286_1 = None
        where_self: "f32[64, 256, 56, 56]" = torch.ops.aten.where.self(le_scalar, full, getitem_318);  le_scalar = full = getitem_318 = None
        sum_dim_int_list: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[64, 256, 56, 56]" = torch.ops.aten.sub.Tensor(arg284_1, arg718_1);  arg284_1 = arg718_1 = None
        mul_tensor: "f32[64, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 4.982461734693877e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 4.982461734693877e-06)
        mul_tensor_3: "f32[256]" = torch.ops.aten.mul.Tensor(arg285_1, arg285_1)
        mul_tensor_4: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[256]" = torch.ops.aten.mul.Tensor(arg285_1, arg28_1);  arg28_1 = None
        unsqueeze_default_6: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[64, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[64, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[64, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[64, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg285_1);  sum_dim_int_list_1 = arg285_1 = None
        slice_tensor: "f32[64, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 224, 256);  mul_tensor_7 = None
        return (mul_tensor_8, slice_tensor)

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
