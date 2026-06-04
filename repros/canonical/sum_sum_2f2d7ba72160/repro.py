"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train_001
Pattern hash: 2f2d7ba72160
Shape hash: d6b8df5b
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 512, 28, 28], f32), T([64, 480, 28, 28], f32), T([64, 448, 28, 28], f32), T([64, 416, 28, 28], f32), T([64, 384, 28, 28], f32), T([64, 352, 28, 28], f32), T([64, 320, 28, 28], f32), T([64, 288, 28, 28], f32), T([64, 256, 28, 28], f32), T([64, 224, 28, 28], f32), T([], f32), T([64, 224, 28, 28], f32), T([64, 224, 28, 28], f32), T([1, 224, 1, 1], f32), T([224], f32), T([224], f32))"

class Repro(torch.nn.Module):
    def forward(self, mul_747: "f32[64, 512, 28, 28]", mul_765: "f32[64, 480, 28, 28]", mul_783: "f32[64, 448, 28, 28]", mul_801: "f32[64, 416, 28, 28]", mul_819: "f32[64, 384, 28, 28]", mul_837: "f32[64, 352, 28, 28]", mul_855: "f32[64, 320, 28, 28]", mul_873: "f32[64, 288, 28, 28]", mul_891: "f32[64, 256, 28, 28]", arg308_1: "f32[64, 224, 28, 28]", full: "f32[]", getitem_297: "f32[64, 224, 28, 28]", arg306_1: "f32[64, 224, 28, 28]", arg711_1: "f32[1, 224, 1, 1]", arg307_1: "f32[224]", arg42_1: "f32[224]"):
        # No stacktrace found for following nodes
        slice_tensor: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_747, 1, 192, 224);  mul_747 = None
        slice_tensor_1: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_765, 1, 192, 224);  mul_765 = None
        add_tensor: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        slice_tensor_2: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_783, 1, 192, 224);  mul_783 = None
        add_tensor_1: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        slice_tensor_3: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_801, 1, 192, 224);  mul_801 = None
        add_tensor_2: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None
        slice_tensor_4: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_819, 1, 192, 224);  mul_819 = None
        add_tensor_3: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None
        slice_tensor_5: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_837, 1, 192, 224);  mul_837 = None
        add_tensor_4: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_tensor_5);  add_tensor_3 = slice_tensor_5 = None
        slice_tensor_6: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_855, 1, 192, 224);  mul_855 = None
        add_tensor_5: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_4, slice_tensor_6);  add_tensor_4 = slice_tensor_6 = None
        slice_tensor_7: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_873, 1, 192, 224);  mul_873 = None
        add_tensor_6: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_5, slice_tensor_7);  add_tensor_5 = slice_tensor_7 = None
        slice_tensor_8: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_891, 1, 192, 224);  mul_891 = None
        add_tensor_7: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_6, slice_tensor_8);  add_tensor_6 = slice_tensor_8 = None
        le_scalar: "b8[64, 224, 28, 28]" = torch.ops.aten.le.Scalar(arg308_1, 0);  arg308_1 = None
        where_self: "f32[64, 224, 28, 28]" = torch.ops.aten.where.self(le_scalar, full, getitem_297);  le_scalar = full = getitem_297 = None
        sum_dim_int_list: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[64, 224, 28, 28]" = torch.ops.aten.sub.Tensor(arg306_1, arg711_1);  arg306_1 = arg711_1 = None
        mul_tensor: "f32[64, 224, 28, 28]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[224]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 1.992984693877551e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[224]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 1.992984693877551e-05)
        mul_tensor_3: "f32[224]" = torch.ops.aten.mul.Tensor(arg307_1, arg307_1)
        mul_tensor_4: "f32[224]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[224]" = torch.ops.aten.mul.Tensor(arg307_1, arg42_1);  arg42_1 = None
        unsqueeze_default_6: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[64, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[64, 224, 28, 28]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[64, 224, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[64, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[224]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg307_1);  sum_dim_int_list_1 = arg307_1 = None
        slice_tensor_9: "f32[64, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 192, 224);  mul_tensor_7 = None
        add_tensor_8: "f32[64, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_7, slice_tensor_9);  add_tensor_7 = slice_tensor_9 = None
        return (mul_tensor_8, add_tensor_8)

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
