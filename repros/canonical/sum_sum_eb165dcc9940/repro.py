"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train_001
Pattern hash: eb165dcc9940
Shape hash: 52c6e8f2
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 1024, 7, 7], f32), T([64, 992, 7, 7], f32), T([64, 960, 7, 7], f32), T([64, 928, 7, 7], f32), T([64, 896, 7, 7], f32), T([64, 864, 7, 7], f32), T([64, 832, 7, 7], f32), T([64, 800, 7, 7], f32), T([64, 768, 7, 7], f32), T([64, 736, 7, 7], f32), T([64, 704, 7, 7], f32), T([64, 672, 7, 7], f32), T([64, 640, 7, 7], f32), T([64, 608, 7, 7], f32), T([64, 576, 7, 7], f32), T([64, 544, 7, 7], f32), T([64, 512, 7, 7], f32), T([], f32), T([64, 512, 7, 7], f32), T([64, 512, 7, 7], f32), T([1, 512, 1, 1], f32), T([512], f32), T([512], f32), T([64, 512, 14, 14], f32))"

class Repro(torch.nn.Module):
    def forward(self, mul_9: "f32[64, 1024, 7, 7]", mul_27: "f32[64, 992, 7, 7]", mul_45: "f32[64, 960, 7, 7]", mul_63: "f32[64, 928, 7, 7]", mul_81: "f32[64, 896, 7, 7]", mul_99: "f32[64, 864, 7, 7]", mul_117: "f32[64, 832, 7, 7]", mul_135: "f32[64, 800, 7, 7]", mul_153: "f32[64, 768, 7, 7]", mul_171: "f32[64, 736, 7, 7]", mul_189: "f32[64, 704, 7, 7]", mul_207: "f32[64, 672, 7, 7]", mul_225: "f32[64, 640, 7, 7]", mul_243: "f32[64, 608, 7, 7]", mul_261: "f32[64, 576, 7, 7]", mul_279: "f32[64, 544, 7, 7]", arg514_1: "f32[64, 512, 7, 7]", full: "f32[]", getitem_93: "f32[64, 512, 7, 7]", arg512_1: "f32[64, 512, 7, 7]", arg643_1: "f32[1, 512, 1, 1]", arg513_1: "f32[512]", arg178_1: "f32[512]", arg511_1: "f32[64, 512, 14, 14]"):
        # No stacktrace found for following nodes
        slice_tensor: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_9, 1, 0, 512);  mul_9 = None
        slice_tensor_1: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_27, 1, 0, 512);  mul_27 = None
        add_tensor: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        slice_tensor_2: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_45, 1, 0, 512);  mul_45 = None
        add_tensor_1: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        slice_tensor_3: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_63, 1, 0, 512);  mul_63 = None
        add_tensor_2: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None
        slice_tensor_4: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_81, 1, 0, 512);  mul_81 = None
        add_tensor_3: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None
        slice_tensor_5: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_99, 1, 0, 512);  mul_99 = None
        add_tensor_4: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_tensor_5);  add_tensor_3 = slice_tensor_5 = None
        slice_tensor_6: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_117, 1, 0, 512);  mul_117 = None
        add_tensor_5: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_4, slice_tensor_6);  add_tensor_4 = slice_tensor_6 = None
        slice_tensor_7: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_135, 1, 0, 512);  mul_135 = None
        add_tensor_6: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_5, slice_tensor_7);  add_tensor_5 = slice_tensor_7 = None
        slice_tensor_8: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_153, 1, 0, 512);  mul_153 = None
        add_tensor_7: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_6, slice_tensor_8);  add_tensor_6 = slice_tensor_8 = None
        slice_tensor_9: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_171, 1, 0, 512);  mul_171 = None
        add_tensor_8: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_7, slice_tensor_9);  add_tensor_7 = slice_tensor_9 = None
        slice_tensor_10: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_189, 1, 0, 512);  mul_189 = None
        add_tensor_9: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_8, slice_tensor_10);  add_tensor_8 = slice_tensor_10 = None
        slice_tensor_11: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_207, 1, 0, 512);  mul_207 = None
        add_tensor_10: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_9, slice_tensor_11);  add_tensor_9 = slice_tensor_11 = None
        slice_tensor_12: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_225, 1, 0, 512);  mul_225 = None
        add_tensor_11: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_10, slice_tensor_12);  add_tensor_10 = slice_tensor_12 = None
        slice_tensor_13: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_243, 1, 0, 512);  mul_243 = None
        add_tensor_12: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_11, slice_tensor_13);  add_tensor_11 = slice_tensor_13 = None
        slice_tensor_14: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_261, 1, 0, 512);  mul_261 = None
        add_tensor_13: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_12, slice_tensor_14);  add_tensor_12 = slice_tensor_14 = None
        slice_tensor_15: "f32[64, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_279, 1, 0, 512);  mul_279 = None
        add_tensor_14: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_13, slice_tensor_15);  add_tensor_13 = slice_tensor_15 = None
        le_scalar: "b8[64, 512, 7, 7]" = torch.ops.aten.le.Scalar(arg514_1, 0);  arg514_1 = None
        where_self: "f32[64, 512, 7, 7]" = torch.ops.aten.where.self(le_scalar, full, getitem_93);  le_scalar = full = getitem_93 = None
        sum_dim_int_list: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[64, 512, 7, 7]" = torch.ops.aten.sub.Tensor(arg512_1, arg643_1);  arg512_1 = arg643_1 = None
        mul_tensor: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00031887755102040814);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00031887755102040814)
        mul_tensor_3: "f32[512]" = torch.ops.aten.mul.Tensor(arg513_1, arg513_1)
        mul_tensor_4: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[512]" = torch.ops.aten.mul.Tensor(arg513_1, arg178_1);  arg178_1 = None
        unsqueeze_default_6: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[64, 512, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[64, 512, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[64, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg513_1);  sum_dim_int_list_1 = arg513_1 = None
        add_tensor_15: "f32[64, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_14, mul_tensor_7);  add_tensor_14 = mul_tensor_7 = None
        avg_pool2d_backward_default: "f32[64, 512, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(add_tensor_15, arg511_1, [2, 2], [2, 2], [0, 0], False, True, None);  add_tensor_15 = arg511_1 = None
        return (mul_tensor_8, avg_pool2d_backward_default)

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
