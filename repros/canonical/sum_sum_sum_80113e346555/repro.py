"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_train_001
Pattern hash: 80113e346555
Shape hash: 9ec587e5
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 128, 4, 4], f32), T([64, 128, 4, 4], f32), T([64, 128, 4, 4], f32), T([], f32), T([64, 128, 4, 4], f32), T([128], f32), T([64, 32], f32), T([64, 32], f32), T([64, 128, 4, 4], f32), T([128], f32), T([64, 32], f32), T([64, 32], f32), S([64, 128, 16]), S([64, 128, 16]), S([64, 32, 4]), S([64, 32, 4]), S([1, 32, 4]), S([64, 32, 4, 16]), S([64, 32, 4, 16]), S([64, 128, 4, 4]), S([64, 32, 4]), S([64, 32, 4]), S([128]), S([64, 128, 16]), S([64, 32, 4]), S([64, 32, 4]), S([1, 32, 4]), S([64, 32, 4, 16]), S([64, 128, 4, 4]), S([64, 32, 4]), S([128]))"

class Repro(torch.nn.Module):
    def forward(self, where_8: "f32[64, 128, 4, 4]", getitem_33: "f32[64, 128, 4, 4]", arg74_1: "f32[64, 128, 4, 4]", full: "f32[]", arg71_1: "f32[64, 128, 4, 4]", arg17_1: "f32[128]", arg73_1: "f32[64, 32]", arg72_1: "f32[64, 32]", arg68_1: "f32[64, 128, 4, 4]", arg15_1: "f32[128]", arg70_1: "f32[64, 32]", arg69_1: "f32[64, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18):
        # No stacktrace found for following nodes
        add_tensor: "f32[64, 128, 4, 4]" = torch.ops.aten.add.Tensor(where_8, getitem_33);  where_8 = getitem_33 = None
        le_scalar: "b8[64, 128, 4, 4]" = torch.ops.aten.le.Scalar(arg74_1, 0);  arg74_1 = None
        where_self: "f32[64, 128, 4, 4]" = torch.ops.aten.where.self(le_scalar, full, add_tensor);  le_scalar = full = add_tensor = None
        mul_tensor: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(where_self, arg71_1)
        view_default: "f32[64, 128, 16]" = torch.ops.aten.view.default(mul_tensor, _shape_param_0);  mul_tensor = _shape_param_0 = None
        sum_dim_int_list: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(view_default, [2]);  view_default = None
        view_default_1: "f32[64, 128, 16]" = torch.ops.aten.view.default(where_self, _shape_param_1);  _shape_param_1 = None
        sum_dim_int_list_1: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(view_default_1, [2]);  view_default_1 = None
        unsqueeze_default: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(arg17_1, 0)
        mul_tensor_1: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, unsqueeze_default)
        view_default_2: "f32[64, 32, 4]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_2);  mul_tensor_1 = _shape_param_2 = None
        sum_dim_int_list_2: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_default_2, [2]);  view_default_2 = None
        mul_tensor_2: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, unsqueeze_default);  unsqueeze_default = None
        view_default_3: "f32[64, 32, 4]" = torch.ops.aten.view.default(mul_tensor_2, _shape_param_3);  mul_tensor_2 = _shape_param_3 = None
        sum_dim_int_list_3: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_default_3, [2]);  view_default_3 = None
        unsqueeze_default_1: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1)
        view_default_4: "f32[1, 32, 4]" = torch.ops.aten.view.default(arg17_1, _shape_param_4);  arg17_1 = _shape_param_4 = None
        mul_tensor_3: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(unsqueeze_default_1, view_default_4);  view_default_4 = None
        mul_tensor_4: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, arg72_1)
        sub_tensor: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list_2);  mul_tensor_4 = sum_dim_int_list_2 = None
        mul_tensor_5: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_tensor, arg73_1);  sub_tensor = None
        mul_tensor_6: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_5, arg73_1);  mul_tensor_5 = None
        mul_tensor_7: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_6, arg73_1);  mul_tensor_6 = None
        mul_tensor_8: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.015625);  mul_tensor_7 = None
        neg_default: "f32[64, 32]" = torch.ops.aten.neg.default(mul_tensor_8)
        mul_tensor_9: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_default, arg72_1);  neg_default = None
        mul_tensor_10: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, arg73_1);  sum_dim_int_list_3 = arg73_1 = None
        mul_tensor_11: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.015625);  mul_tensor_10 = None
        sub_tensor_1: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_9, mul_tensor_11);  mul_tensor_9 = mul_tensor_11 = None
        unsqueeze_default_2: "f32[64, 32, 4, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_3: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, -1);  mul_tensor_8 = None
        unsqueeze_default_4: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, -1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_tensor_1, -1);  sub_tensor_1 = None
        unsqueeze_default_6: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, -1);  unsqueeze_default_5 = None
        view_default_5: "f32[64, 32, 4, 16]" = torch.ops.aten.view.default(where_self, _shape_param_5);  _shape_param_5 = None
        mul_tensor_12: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_default_5, unsqueeze_default_2);  unsqueeze_default_2 = None
        view_default_6: "f32[64, 32, 4, 16]" = torch.ops.aten.view.default(arg71_1, _shape_param_6);  arg71_1 = _shape_param_6 = None
        mul_tensor_13: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_default_6, unsqueeze_default_4);  view_default_6 = unsqueeze_default_4 = None
        add_tensor_1: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        add_tensor_2: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(add_tensor_1, unsqueeze_default_6);  add_tensor_1 = unsqueeze_default_6 = None
        view_default_7: "f32[64, 128, 4, 4]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_7);  add_tensor_2 = _shape_param_7 = None
        view_default_8: "f32[64, 32, 4]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_8);  sum_dim_int_list = _shape_param_8 = None
        view_default_9: "f32[64, 32, 4]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_9);  _shape_param_9 = None
        unsqueeze_default_7: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        mul_tensor_14: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(view_default_9, unsqueeze_default_7);  unsqueeze_default_7 = None
        sub_tensor_2: "f32[64, 32, 4]" = torch.ops.aten.sub.Tensor(view_default_8, mul_tensor_14);  view_default_8 = mul_tensor_14 = None
        mul_tensor_15: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_1);  sub_tensor_2 = unsqueeze_default_1 = None
        sum_dim_int_list_4: "f32[32, 4]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0]);  mul_tensor_15 = None
        view_default_10: "f32[128]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_10);  sum_dim_int_list_4 = _shape_param_10 = None
        sum_dim_int_list_5: "f32[128]" = torch.ops.aten.sum.dim_IntList(sum_dim_int_list_1, [0])
        mul_tensor_16: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(where_self, arg68_1);  where_self = None
        view_default_11: "f32[64, 128, 16]" = torch.ops.aten.view.default(mul_tensor_16, _shape_param_11);  mul_tensor_16 = _shape_param_11 = None
        sum_dim_int_list_6: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(view_default_11, [2]);  view_default_11 = None
        unsqueeze_default_8: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(arg15_1, 0)
        mul_tensor_17: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_6, unsqueeze_default_8)
        view_default_12: "f32[64, 32, 4]" = torch.ops.aten.view.default(mul_tensor_17, _shape_param_12);  mul_tensor_17 = _shape_param_12 = None
        sum_dim_int_list_7: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_default_12, [2]);  view_default_12 = None
        mul_tensor_18: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, unsqueeze_default_8);  unsqueeze_default_8 = None
        view_default_13: "f32[64, 32, 4]" = torch.ops.aten.view.default(mul_tensor_18, _shape_param_13);  mul_tensor_18 = _shape_param_13 = None
        sum_dim_int_list_8: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_default_13, [2]);  view_default_13 = None
        unsqueeze_default_9: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1)
        view_default_14: "f32[1, 32, 4]" = torch.ops.aten.view.default(arg15_1, _shape_param_14);  arg15_1 = _shape_param_14 = None
        mul_tensor_19: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(unsqueeze_default_9, view_default_14);  view_default_14 = None
        mul_tensor_20: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_8, arg69_1)
        sub_tensor_3: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_20, sum_dim_int_list_7);  mul_tensor_20 = sum_dim_int_list_7 = None
        mul_tensor_21: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_tensor_3, arg70_1);  sub_tensor_3 = None
        mul_tensor_22: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_21, arg70_1);  mul_tensor_21 = None
        mul_tensor_23: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_22, arg70_1);  mul_tensor_22 = None
        mul_tensor_24: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_23, 0.015625);  mul_tensor_23 = None
        neg_default_1: "f32[64, 32]" = torch.ops.aten.neg.default(mul_tensor_24)
        mul_tensor_25: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_default_1, arg69_1);  neg_default_1 = None
        mul_tensor_26: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_8, arg70_1);  sum_dim_int_list_8 = arg70_1 = None
        mul_tensor_27: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_26, 0.015625);  mul_tensor_26 = None
        sub_tensor_4: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_25, mul_tensor_27);  mul_tensor_25 = mul_tensor_27 = None
        unsqueeze_default_10: "f32[64, 32, 4, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_19, -1);  mul_tensor_19 = None
        unsqueeze_default_11: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_24, -1);  mul_tensor_24 = None
        unsqueeze_default_12: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, -1);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_tensor_4, -1);  sub_tensor_4 = None
        unsqueeze_default_14: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, -1);  unsqueeze_default_13 = None
        mul_tensor_28: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_default_5, unsqueeze_default_10);  view_default_5 = unsqueeze_default_10 = None
        view_default_15: "f32[64, 32, 4, 16]" = torch.ops.aten.view.default(arg68_1, _shape_param_15);  arg68_1 = _shape_param_15 = None
        mul_tensor_29: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_default_15, unsqueeze_default_12);  view_default_15 = unsqueeze_default_12 = None
        add_tensor_3: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None
        add_tensor_4: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(add_tensor_3, unsqueeze_default_14);  add_tensor_3 = unsqueeze_default_14 = None
        view_default_16: "f32[64, 128, 4, 4]" = torch.ops.aten.view.default(add_tensor_4, _shape_param_16);  add_tensor_4 = _shape_param_16 = None
        view_default_17: "f32[64, 32, 4]" = torch.ops.aten.view.default(sum_dim_int_list_6, _shape_param_17);  sum_dim_int_list_6 = _shape_param_17 = None
        unsqueeze_default_15: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        mul_tensor_30: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(view_default_9, unsqueeze_default_15);  view_default_9 = unsqueeze_default_15 = None
        sub_tensor_5: "f32[64, 32, 4]" = torch.ops.aten.sub.Tensor(view_default_17, mul_tensor_30);  view_default_17 = mul_tensor_30 = None
        mul_tensor_31: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_9);  sub_tensor_5 = unsqueeze_default_9 = None
        sum_dim_int_list_9: "f32[32, 4]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0]);  mul_tensor_31 = None
        view_default_18: "f32[128]" = torch.ops.aten.view.default(sum_dim_int_list_9, _shape_param_18);  sum_dim_int_list_9 = _shape_param_18 = None
        sum_dim_int_list_10: "f32[128]" = torch.ops.aten.sum.dim_IntList(sum_dim_int_list_1, [0]);  sum_dim_int_list_1 = None
        return (view_default_7, view_default_10, sum_dim_int_list_5, view_default_16, view_default_18, sum_dim_int_list_10)

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
