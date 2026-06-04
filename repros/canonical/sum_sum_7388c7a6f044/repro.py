"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train_001
Pattern hash: 7388c7a6f044
Shape hash: 2334b042
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 240, 28, 28], f32), T([1, 240, 1, 1], f32), T([1, 240, 1, 1], f32), T([240], f32), T([240], f32), T([512, 240, 28, 28], f32), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg201_1: "f32[512, 240, 28, 28]", arg202_1: "f32[1, 240, 1, 1]", arg203_1: "f32[1, 240, 1, 1]", arg48_1: "f32[240]", arg49_1: "f32[240]", getitem_111: "f32[512, 240, 28, 28]", full: "f32[]"):
        # No stacktrace found for following nodes
        sub_tensor: "f32[512, 240, 28, 28]" = torch.ops.aten.sub.Tensor(arg201_1, arg202_1)
        mul_tensor: "f32[512, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, arg203_1);  sub_tensor = None
        unsqueeze_default: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1)
        unsqueeze_default_1: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_default_3: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[512, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        le_scalar: "b8[512, 240, 28, 28]" = torch.ops.aten.le.Scalar(add_tensor, -3)
        lt_scalar: "b8[512, 240, 28, 28]" = torch.ops.aten.lt.Scalar(add_tensor, 3)
        div_tensor: "f32[512, 240, 28, 28]" = torch.ops.aten.div.Tensor(add_tensor, 3);  add_tensor = None
        add_tensor_1: "f32[512, 240, 28, 28]" = torch.ops.aten.add.Tensor(div_tensor, 0.5);  div_tensor = None
        mul_tensor_2: "f32[512, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_111, add_tensor_1);  add_tensor_1 = None
        where_self: "f32[512, 240, 28, 28]" = torch.ops.aten.where.self(lt_scalar, mul_tensor_2, getitem_111);  lt_scalar = mul_tensor_2 = getitem_111 = None
        where_self_1: "f32[512, 240, 28, 28]" = torch.ops.aten.where.self(le_scalar, full, where_self);  le_scalar = full = where_self = None
        squeeze_dims: "f32[240]" = torch.ops.aten.squeeze.dims(arg202_1, [0, 2, 3]);  arg202_1 = None
        unsqueeze_default_4: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_1: "f32[512, 240, 28, 28]" = torch.ops.aten.sub.Tensor(arg201_1, unsqueeze_default_6);  arg201_1 = unsqueeze_default_6 = None
        mul_tensor_3: "f32[512, 240, 28, 28]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_1)
        sum_dim_int_list_1: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 2, 3]);  mul_tensor_3 = None
        mul_tensor_4: "f32[240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_8: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_5: "f32[240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06)
        squeeze_dims_1: "f32[240]" = torch.ops.aten.squeeze.dims(arg203_1, [0, 2, 3]);  arg203_1 = None
        mul_tensor_6: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_7: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        unsqueeze_default_10: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_11: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_8: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg48_1);  arg48_1 = None
        unsqueeze_default_13: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, 0);  mul_tensor_8 = None
        unsqueeze_default_14: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_9: "f32[512, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[512, 240, 28, 28]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_9);  where_self_1 = mul_tensor_9 = None
        sub_tensor_3: "f32[512, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_10: "f32[512, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_11: "f32[240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        return (mul_tensor_10, mul_tensor_11)

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
