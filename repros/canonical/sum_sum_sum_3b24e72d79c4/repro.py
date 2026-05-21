"""
Standalone repro captured via capture_hook.
Label: timm_timm_adv_inception_v3_train_train_001
Pattern hash: 3b24e72d79c4
Shape hash: 6b5fd3b9
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
_shapes_config = "(T([128, 2048, 8, 8], f32), T([128, 2048, 8, 8], f32), T([128, 2048, 8, 8], f32), T([128, 2048, 8, 8], f32), T([128, 2048, 8, 8], f32), T([128, 384, 8, 8], f32), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([], f32), T([128, 384, 8, 8], f32), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[128, 2048, 8, 8]", arg514_1: "f32[128, 2048, 8, 8]", getitem_12: "f32[128, 2048, 8, 8]", getitem_21: "f32[128, 2048, 8, 8]", getitem_24: "f32[128, 2048, 8, 8]", arg507_1: "f32[128, 384, 8, 8]", arg508_1: "f32[1, 384, 1, 1]", arg509_1: "f32[1, 384, 1, 1]", arg206_1: "f32[384]", arg207_1: "f32[384]", full_1: "f32[]", arg504_1: "f32[128, 384, 8, 8]", arg505_1: "f32[1, 384, 1, 1]", arg506_1: "f32[1, 384, 1, 1]", arg203_1: "f32[384]", arg204_1: "f32[384]"):
        # No stacktrace found for following nodes
        avg_pool2d_backward_default: "f32[128, 2048, 8, 8]" = torch.ops.aten.avg_pool2d_backward.default(getitem, arg514_1, [3, 3], [1, 1], [1, 1], False, True, None);  getitem = arg514_1 = None
        add_tensor: "f32[128, 2048, 8, 8]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_default, getitem_12);  avg_pool2d_backward_default = getitem_12 = None
        add_tensor_1: "f32[128, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_tensor, getitem_21);  add_tensor = getitem_21 = None
        add_tensor_2: "f32[128, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_tensor_1, getitem_24);  add_tensor_1 = getitem_24 = None
        slice_tensor: "f32[128, 768, 8, 8]" = torch.ops.aten.slice.Tensor(add_tensor_2, 1, 1088, 1856);  add_tensor_2 = None
        slice_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_tensor, 1, 0, 384)
        slice_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_tensor, 1, 384, 768);  slice_tensor = None
        sub_tensor: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(arg507_1, arg508_1)
        mul_tensor: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, arg509_1);  sub_tensor = None
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1)
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        le_scalar: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[128, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar, full_1, slice_tensor_2);  le_scalar = slice_tensor_2 = None
        squeeze_dims: "f32[384]" = torch.ops.aten.squeeze.dims(arg508_1, [0, 2, 3]);  arg508_1 = None
        unsqueeze_default_4: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(arg507_1, unsqueeze_default_6);  arg507_1 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0001220703125);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0001220703125);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[384]" = torch.ops.aten.squeeze.dims(arg509_1, [0, 2, 3]);  arg509_1 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg206_1);  squeeze_dims_1 = arg206_1 = None
        unsqueeze_default_13: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        sub_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(arg504_1, arg505_1)
        mul_tensor_10: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_4, arg506_1);  sub_tensor_4 = None
        unsqueeze_default_16: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1)
        unsqueeze_default_17: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        mul_tensor_11: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_10, unsqueeze_default_17);  mul_tensor_10 = unsqueeze_default_17 = None
        unsqueeze_default_18: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_default_19: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        add_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_11, unsqueeze_default_19);  mul_tensor_11 = unsqueeze_default_19 = None
        relu_default_1: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        le_scalar_1: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        where_self_1: "f32[128, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar_1, full_1, slice_tensor_1);  le_scalar_1 = full_1 = slice_tensor_1 = None
        squeeze_dims_2: "f32[384]" = torch.ops.aten.squeeze.dims(arg505_1, [0, 2, 3]);  arg505_1 = None
        unsqueeze_default_20: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_21: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 3);  unsqueeze_default_21 = None
        sum_dim_int_list_2: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_5: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(arg504_1, unsqueeze_default_22);  arg504_1 = unsqueeze_default_22 = None
        mul_tensor_12: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_5)
        sum_dim_int_list_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 2, 3]);  mul_tensor_12 = None
        mul_tensor_13: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.0001220703125);  sum_dim_int_list_2 = None
        unsqueeze_default_23: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_24: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 3);  unsqueeze_default_24 = None
        mul_tensor_14: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.0001220703125);  sum_dim_int_list_3 = None
        squeeze_dims_3: "f32[384]" = torch.ops.aten.squeeze.dims(arg506_1, [0, 2, 3]);  arg506_1 = None
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_16: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_14, mul_tensor_15);  mul_tensor_14 = mul_tensor_15 = None
        unsqueeze_default_26: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_16, 0);  mul_tensor_16 = None
        unsqueeze_default_27: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 3);  unsqueeze_default_27 = None
        mul_tensor_17: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, arg203_1);  squeeze_dims_3 = arg203_1 = None
        unsqueeze_default_29: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_17, 0);  mul_tensor_17 = None
        unsqueeze_default_30: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        mul_tensor_18: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_28);  sub_tensor_5 = unsqueeze_default_28 = None
        sub_tensor_6: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_18);  where_self_1 = mul_tensor_18 = None
        sub_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_6, unsqueeze_default_25);  sub_tensor_6 = unsqueeze_default_25 = None
        mul_tensor_19: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_31);  sub_tensor_7 = unsqueeze_default_31 = None
        return (mul_tensor_19, mul_tensor_9)



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
