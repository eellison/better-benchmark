"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train_001
Pattern hash: 62f7c19427cb
Shape hash: bd92fae7
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 256, 56, 56], f32), T([64, 224, 56, 56], f32), T([64, 192, 56, 56], f32), T([64, 160, 56, 56], f32), T([], f32), T([64, 160, 56, 56], f32), T([64, 160, 56, 56], f32), T([1, 160, 1, 1], f32), T([160], f32), T([160], f32))"

class Repro(torch.nn.Module):
    def forward(self, mul_972: "f32[64, 256, 56, 56]", mul_990: "f32[64, 224, 56, 56]", mul_1008: "f32[64, 192, 56, 56]", arg268_1: "f32[64, 160, 56, 56]", full: "f32[]", getitem_336: "f32[64, 160, 56, 56]", arg266_1: "f32[64, 160, 56, 56]", arg724_1: "f32[1, 160, 1, 1]", arg267_1: "f32[160]", arg16_1: "f32[160]"):
        # No stacktrace found for following nodes
        slice_tensor: "f32[64, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_972, 1, 128, 160);  mul_972 = None
        slice_tensor_1: "f32[64, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_990, 1, 128, 160);  mul_990 = None
        add_tensor: "f32[64, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        slice_tensor_2: "f32[64, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1008, 1, 128, 160);  mul_1008 = None
        add_tensor_1: "f32[64, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        le_scalar: "b8[64, 160, 56, 56]" = torch.ops.aten.le.Scalar(arg268_1, 0);  arg268_1 = None
        where_self: "f32[64, 160, 56, 56]" = torch.ops.aten.where.self(le_scalar, full, getitem_336);  le_scalar = full = getitem_336 = None
        sum_dim_int_list: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[64, 160, 56, 56]" = torch.ops.aten.sub.Tensor(arg266_1, arg724_1);  arg266_1 = arg724_1 = None
        mul_tensor: "f32[64, 160, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 4.982461734693877e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 4.982461734693877e-06)
        mul_tensor_3: "f32[160]" = torch.ops.aten.mul.Tensor(arg267_1, arg267_1)
        mul_tensor_4: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[160]" = torch.ops.aten.mul.Tensor(arg267_1, arg16_1);  arg16_1 = None
        unsqueeze_default_6: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[64, 160, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[64, 160, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[64, 160, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[64, 160, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg267_1);  sum_dim_int_list_1 = arg267_1 = None
        slice_tensor_3: "f32[64, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 128, 160);  mul_tensor_7 = None
        add_tensor_2: "f32[64, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None
        return (mul_tensor_8, add_tensor_2)

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
