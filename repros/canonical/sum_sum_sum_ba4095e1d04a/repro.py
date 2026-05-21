"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train_001
Pattern hash: ba4095e1d04a
Shape hash: c3335a89
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
_shapes_config = "(T([128, 768, 17, 17], f32), T([128, 768, 17, 17], f32), T([128, 768, 17, 17], f32), T([128, 768, 17, 17], f32), T([128, 768, 17, 17], f32), T([128, 96, 17, 17], f32), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([96], f32), T([], f32), T([128, 384, 17, 17], f32), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_162: "f32[128, 768, 17, 17]", arg337_1: "f32[128, 768, 17, 17]", getitem_177: "f32[128, 768, 17, 17]", getitem_186: "f32[128, 768, 17, 17]", getitem_189: "f32[128, 768, 17, 17]", arg333_1: "f32[128, 96, 17, 17]", arg334_1: "f32[1, 96, 1, 1]", arg335_1: "f32[1, 96, 1, 1]", arg75_1: "f32[96]", arg76_1: "f32[96]", full_1: "f32[]", arg324_1: "f32[128, 384, 17, 17]", arg325_1: "f32[1, 384, 1, 1]", arg326_1: "f32[1, 384, 1, 1]", arg68_1: "f32[384]", arg69_1: "f32[384]"):
        # No stacktrace found for following nodes
        avg_pool2d_backward_default: "f32[128, 768, 17, 17]" = torch.ops.aten.avg_pool2d_backward.default(getitem_162, arg337_1, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_162 = arg337_1 = None
        add_tensor: "f32[128, 768, 17, 17]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_default, getitem_177);  avg_pool2d_backward_default = getitem_177 = None
        add_tensor_1: "f32[128, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_tensor, getitem_186);  add_tensor = getitem_186 = None
        add_tensor_2: "f32[128, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_tensor_1, getitem_189);  add_tensor_1 = getitem_189 = None
        slice_tensor: "f32[128, 384, 17, 17]" = torch.ops.aten.slice.Tensor(add_tensor_2, 1, 0, 384)
        slice_tensor_1: "f32[128, 96, 17, 17]" = torch.ops.aten.slice.Tensor(add_tensor_2, 1, 384, 480);  add_tensor_2 = None
        sub_tensor: "f32[128, 96, 17, 17]" = torch.ops.aten.sub.Tensor(arg333_1, arg334_1)
        mul_tensor: "f32[128, 96, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, arg335_1);  sub_tensor = None
        unsqueeze_default: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1)
        unsqueeze_default_1: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 96, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_default_3: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 96, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[128, 96, 17, 17]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        le_scalar: "b8[128, 96, 17, 17]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[128, 96, 17, 17]" = torch.ops.aten.where.self(le_scalar, full_1, slice_tensor_1);  le_scalar = slice_tensor_1 = None
        squeeze_dims: "f32[96]" = torch.ops.aten.squeeze.dims(arg334_1, [0, 2, 3]);  arg334_1 = None
        unsqueeze_default_4: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 96, 17, 17]" = torch.ops.aten.sub.Tensor(arg333_1, unsqueeze_default_6);  arg333_1 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 96, 17, 17]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.703287197231834e-05);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.703287197231834e-05)
        squeeze_dims_1: "f32[96]" = torch.ops.aten.squeeze.dims(arg335_1, [0, 2, 3]);  arg335_1 = None
        mul_tensor_5: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg75_1);  arg75_1 = None
        unsqueeze_default_13: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 96, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 96, 17, 17]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 96, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 96, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        sub_tensor_4: "f32[128, 384, 17, 17]" = torch.ops.aten.sub.Tensor(arg324_1, arg325_1)
        mul_tensor_11: "f32[128, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_4, arg326_1);  sub_tensor_4 = None
        unsqueeze_default_16: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1)
        unsqueeze_default_17: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        mul_tensor_12: "f32[128, 384, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_11, unsqueeze_default_17);  mul_tensor_11 = unsqueeze_default_17 = None
        unsqueeze_default_18: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_default_19: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        add_tensor_4: "f32[128, 384, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_12, unsqueeze_default_19);  mul_tensor_12 = unsqueeze_default_19 = None
        relu_default_1: "f32[128, 384, 17, 17]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        le_scalar_1: "b8[128, 384, 17, 17]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        where_self_1: "f32[128, 384, 17, 17]" = torch.ops.aten.where.self(le_scalar_1, full_1, slice_tensor);  le_scalar_1 = full_1 = slice_tensor = None
        squeeze_dims_2: "f32[384]" = torch.ops.aten.squeeze.dims(arg325_1, [0, 2, 3]);  arg325_1 = None
        unsqueeze_default_20: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_21: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 3);  unsqueeze_default_21 = None
        sum_dim_int_list_2: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_5: "f32[128, 384, 17, 17]" = torch.ops.aten.sub.Tensor(arg324_1, unsqueeze_default_22);  arg324_1 = unsqueeze_default_22 = None
        mul_tensor_13: "f32[128, 384, 17, 17]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_5)
        sum_dim_int_list_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 2, 3]);  mul_tensor_13 = None
        mul_tensor_14: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.703287197231834e-05);  sum_dim_int_list_2 = None
        unsqueeze_default_23: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_24: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 3);  unsqueeze_default_24 = None
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.703287197231834e-05)
        squeeze_dims_3: "f32[384]" = torch.ops.aten.squeeze.dims(arg326_1, [0, 2, 3]);  arg326_1 = None
        mul_tensor_16: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_17: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        unsqueeze_default_26: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_17, 0);  mul_tensor_17 = None
        unsqueeze_default_27: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 3);  unsqueeze_default_27 = None
        mul_tensor_18: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, arg68_1);  arg68_1 = None
        unsqueeze_default_29: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_18, 0);  mul_tensor_18 = None
        unsqueeze_default_30: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        mul_tensor_19: "f32[128, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_28);  sub_tensor_5 = unsqueeze_default_28 = None
        sub_tensor_6: "f32[128, 384, 17, 17]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_19);  where_self_1 = mul_tensor_19 = None
        sub_tensor_7: "f32[128, 384, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_6, unsqueeze_default_25);  sub_tensor_6 = unsqueeze_default_25 = None
        mul_tensor_20: "f32[128, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_31);  sub_tensor_7 = unsqueeze_default_31 = None
        mul_tensor_21: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, squeeze_dims_3);  sum_dim_int_list_3 = squeeze_dims_3 = None
        return (mul_tensor_9, mul_tensor_10, mul_tensor_20, mul_tensor_21)



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
