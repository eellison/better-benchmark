"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_resnest_train_001
Pattern hash: 109f690634a7
Shape hash: 08c18ca6
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
_shapes_config = "(T([32, 256, 28, 28], f32), T([32, 256, 56, 56], f32), T([32, 256, 56, 56], f32), T([], f32), T([32, 256, 56, 56], f32), T([1, 256, 1, 1], f32), T([256], f32), T([256], f32), T([32, 256, 56, 56], f32), T([1, 256, 1, 1], f32), T([256], f32), T([256], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_36: "f32[32, 256, 28, 28]", arg86_1: "f32[32, 256, 56, 56]", getitem_51: "f32[32, 256, 56, 56]", full: "f32[]", arg84_1: "f32[32, 256, 56, 56]", arg156_1: "f32[1, 256, 1, 1]", arg85_1: "f32[256]", arg19_1: "f32[256]", arg82_1: "f32[32, 256, 56, 56]", arg157_1: "f32[1, 256, 1, 1]", arg83_1: "f32[256]", arg17_1: "f32[256]"):
        # No stacktrace found for following nodes
        avg_pool2d_backward_default: "f32[32, 256, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(getitem_36, arg86_1, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_36 = None
        add_tensor: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_default, getitem_51);  avg_pool2d_backward_default = getitem_51 = None
        le_scalar: "b8[32, 256, 56, 56]" = torch.ops.aten.le.Scalar(arg86_1, 0);  arg86_1 = None
        where_self: "f32[32, 256, 56, 56]" = torch.ops.aten.where.self(le_scalar, full, add_tensor);  le_scalar = full = add_tensor = None
        sum_dim_int_list: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(arg84_1, arg156_1);  arg84_1 = arg156_1 = None
        mul_tensor: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 9.964923469387754e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 9.964923469387754e-06)
        mul_tensor_3: "f32[256]" = torch.ops.aten.mul.Tensor(arg85_1, arg85_1)
        mul_tensor_4: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[256]" = torch.ops.aten.mul.Tensor(arg85_1, arg19_1);  arg19_1 = None
        unsqueeze_default_6: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  mul_tensor_6 = None
        sub_tensor_2: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg85_1);  sum_dim_int_list_1 = arg85_1 = None
        sum_dim_int_list_2: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_3: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(arg82_1, arg157_1);  arg82_1 = arg157_1 = None
        mul_tensor_9: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_3)
        sum_dim_int_list_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 2, 3]);  mul_tensor_9 = None
        mul_tensor_10: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 9.964923469387754e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_10, 0);  mul_tensor_10 = None
        unsqueeze_default_10: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_11: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 9.964923469387754e-06)
        mul_tensor_12: "f32[256]" = torch.ops.aten.mul.Tensor(arg83_1, arg83_1)
        mul_tensor_13: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_12: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_13: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_14: "f32[256]" = torch.ops.aten.mul.Tensor(arg83_1, arg17_1);  arg17_1 = None
        unsqueeze_default_15: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_16: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_15: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_15);  where_self = mul_tensor_15 = None
        sub_tensor_5: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_16: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        mul_tensor_17: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, arg83_1);  sum_dim_int_list_3 = arg83_1 = None
        return (mul_tensor_7, mul_tensor_8, mul_tensor_16, mul_tensor_17)



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
