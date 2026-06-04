"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv2_100_train_001
Pattern hash: d550f2d61c6c
Shape hash: c551a545
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 32, 112, 112], f32), T([1, 32, 1, 1], f32), T([1, 32, 1, 1], f32), T([32], f32), T([32], f32), T([], f32), T([128, 32, 112, 112], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg141_1: "f32[128, 32, 112, 112]", arg142_1: "f32[1, 32, 1, 1]", arg143_1: "f32[1, 32, 1, 1]", arg2_1: "f32[32]", arg3_1: "f32[32]", full_1: "f32[]", getitem_150: "f32[128, 32, 112, 112]"):
        # No stacktrace found for following nodes
        sub_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(arg141_1, arg142_1)
        mul_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, arg143_1);  sub_tensor = None
        unsqueeze_default: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1)
        unsqueeze_default_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_default_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        le_scalar: "b8[128, 32, 112, 112]" = torch.ops.aten.le.Scalar(add_tensor, 0.0)
        ge_scalar: "b8[128, 32, 112, 112]" = torch.ops.aten.ge.Scalar(add_tensor, 6.0);  add_tensor = None
        bitwise_or_tensor: "b8[128, 32, 112, 112]" = torch.ops.aten.bitwise_or.Tensor(le_scalar, ge_scalar);  le_scalar = ge_scalar = None
        where_self: "f32[128, 32, 112, 112]" = torch.ops.aten.where.self(bitwise_or_tensor, full_1, getitem_150);  bitwise_or_tensor = full_1 = getitem_150 = None
        squeeze_dims: "f32[32]" = torch.ops.aten.squeeze.dims(arg142_1, [0, 2, 3]);  arg142_1 = None
        unsqueeze_default_4: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(arg141_1, unsqueeze_default_6);  arg141_1 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.228077168367346e-07);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.228077168367346e-07)
        squeeze_dims_1: "f32[32]" = torch.ops.aten.squeeze.dims(arg143_1, [0, 2, 3]);  arg143_1 = None
        mul_tensor_5: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg2_1);  arg2_1 = None
        unsqueeze_default_13: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        return (mul_tensor_9, mul_tensor_10)

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
