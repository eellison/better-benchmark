"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train_001
Pattern hash: 95dac16d4328
Shape hash: bbce3ec7
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 40, 28, 28], f32), T([512, 40, 28, 28], f32), T([512, 40, 28, 28], f32), T([1, 40, 1, 1], f32), T([40], f32), T([40], f32), T([512, 20, 28, 28], f32), T([1, 20, 1, 1], f32), T([20], f32), T([20], f32))"

class Repro(torch.nn.Module):
    def forward(self, clone_11: "f32[512, 40, 28, 28]", getitem_207: "f32[512, 40, 28, 28]", arg268_1: "f32[512, 40, 28, 28]", arg522_1: "f32[1, 40, 1, 1]", arg269_1: "f32[40]", arg53_1: "f32[40]", arg263_1: "f32[512, 20, 28, 28]", arg524_1: "f32[1, 20, 1, 1]", arg264_1: "f32[20]", arg49_1: "f32[20]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[512, 40, 28, 28]" = torch.ops.aten.add.Tensor(clone_11, getitem_207);  clone_11 = getitem_207 = None
        new_empty_strided_default: "f32[512, 40, 28, 28]" = torch.ops.aten.new_empty_strided.default(add_tensor, [512, 40, 28, 28], [31360, 1, 1120, 40])
        copy_default: "f32[512, 40, 28, 28]" = torch.ops.aten.copy.default(new_empty_strided_default, add_tensor);  new_empty_strided_default = add_tensor = None
        clone_default: "f32[512, 40, 28, 28]" = torch.ops.aten.clone.default(copy_default, memory_format = torch.contiguous_format)
        copy_default_1: "f32[512, 40, 28, 28]" = torch.ops.aten.copy.default(copy_default, clone_default);  copy_default = None
        sum_dim_int_list: "f32[40]" = torch.ops.aten.sum.dim_IntList(clone_default, [0, 2, 3])
        sub_tensor: "f32[512, 40, 28, 28]" = torch.ops.aten.sub.Tensor(arg268_1, arg522_1);  arg268_1 = arg522_1 = None
        mul_tensor: "f32[512, 40, 28, 28]" = torch.ops.aten.mul.Tensor(clone_default, sub_tensor)
        sum_dim_int_list_1: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[40]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[40]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06)
        mul_tensor_3: "f32[40]" = torch.ops.aten.mul.Tensor(arg269_1, arg269_1)
        mul_tensor_4: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[40]" = torch.ops.aten.mul.Tensor(arg269_1, arg53_1);  arg53_1 = None
        unsqueeze_default_6: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 40, 28, 28]" = torch.ops.aten.sub.Tensor(clone_default, mul_tensor_6);  clone_default = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[40]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg269_1);  sum_dim_int_list_1 = arg269_1 = None
        slice_tensor: "f32[512, 20, 28, 28]" = torch.ops.aten.slice.Tensor(copy_default_1, 1, 20, 40);  copy_default_1 = None
        sum_dim_int_list_2: "f32[20]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0, 2, 3])
        sub_tensor_3: "f32[512, 20, 28, 28]" = torch.ops.aten.sub.Tensor(arg263_1, arg524_1);  arg263_1 = arg524_1 = None
        mul_tensor_9: "f32[512, 20, 28, 28]" = torch.ops.aten.mul.Tensor(slice_tensor, sub_tensor_3)
        sum_dim_int_list_3: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 2, 3]);  mul_tensor_9 = None
        mul_tensor_10: "f32[20]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.4912308673469386e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_tensor_10, 0);  mul_tensor_10 = None
        unsqueeze_default_10: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_11: "f32[20]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.4912308673469386e-06)
        mul_tensor_12: "f32[20]" = torch.ops.aten.mul.Tensor(arg264_1, arg264_1)
        mul_tensor_13: "f32[20]" = torch.ops.aten.mul.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_12: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_13: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_14: "f32[20]" = torch.ops.aten.mul.Tensor(arg264_1, arg49_1);  arg49_1 = None
        unsqueeze_default_15: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_16: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_15: "f32[512, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[512, 20, 28, 28]" = torch.ops.aten.sub.Tensor(slice_tensor, mul_tensor_15);  slice_tensor = mul_tensor_15 = None
        sub_tensor_5: "f32[512, 20, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_16: "f32[512, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        mul_tensor_17: "f32[20]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, arg264_1);  sum_dim_int_list_3 = arg264_1 = None
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
