"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train_001
Pattern hash: ed9e2feedf2a
Shape hash: 0f3133d9
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
_shapes_config = "(T([64, 1024, 14, 14], f32), T([64, 992, 14, 14], f32), T([64, 960, 14, 14], f32), T([64, 928, 14, 14], f32), T([64, 896, 14, 14], f32), T([64, 864, 14, 14], f32), T([64, 832, 14, 14], f32), T([64, 800, 14, 14], f32), T([64, 768, 14, 14], f32), T([64, 736, 14, 14], f32), T([64, 704, 14, 14], f32), T([64, 672, 14, 14], f32), T([64, 640, 14, 14], f32), T([64, 608, 14, 14], f32), T([64, 576, 14, 14], f32), T([64, 544, 14, 14], f32), T([64, 512, 14, 14], f32), T([64, 480, 14, 14], f32), T([64, 448, 14, 14], f32), T([64, 416, 14, 14], f32), T([64, 384, 14, 14], f32), T([], f32), T([64, 384, 14, 14], f32), T([64, 384, 14, 14], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32))"

class Repro(torch.nn.Module):
    def forward(self, mul_306: "f32[64, 1024, 14, 14]", mul_324: "f32[64, 992, 14, 14]", mul_342: "f32[64, 960, 14, 14]", mul_360: "f32[64, 928, 14, 14]", mul_378: "f32[64, 896, 14, 14]", mul_396: "f32[64, 864, 14, 14]", mul_414: "f32[64, 832, 14, 14]", mul_432: "f32[64, 800, 14, 14]", mul_450: "f32[64, 768, 14, 14]", mul_468: "f32[64, 736, 14, 14]", mul_486: "f32[64, 704, 14, 14]", mul_504: "f32[64, 672, 14, 14]", mul_522: "f32[64, 640, 14, 14]", mul_540: "f32[64, 608, 14, 14]", mul_558: "f32[64, 576, 14, 14]", mul_576: "f32[64, 544, 14, 14]", mul_594: "f32[64, 512, 14, 14]", mul_612: "f32[64, 480, 14, 14]", mul_630: "f32[64, 448, 14, 14]", mul_648: "f32[64, 416, 14, 14]", arg390_1: "f32[64, 384, 14, 14]", full: "f32[]", getitem_216: "f32[64, 384, 14, 14]", arg388_1: "f32[64, 384, 14, 14]", arg684_1: "f32[1, 384, 1, 1]", arg389_1: "f32[384]", arg96_1: "f32[384]"):
        # No stacktrace found for following nodes
        slice_tensor: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_306, 1, 352, 384);  mul_306 = None
        slice_tensor_1: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_324, 1, 352, 384);  mul_324 = None
        add_tensor: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        slice_tensor_2: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_342, 1, 352, 384);  mul_342 = None
        add_tensor_1: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        slice_tensor_3: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_360, 1, 352, 384);  mul_360 = None
        add_tensor_2: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None
        slice_tensor_4: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_378, 1, 352, 384);  mul_378 = None
        add_tensor_3: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None
        slice_tensor_5: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_396, 1, 352, 384);  mul_396 = None
        add_tensor_4: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_tensor_5);  add_tensor_3 = slice_tensor_5 = None
        slice_tensor_6: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_414, 1, 352, 384);  mul_414 = None
        add_tensor_5: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_4, slice_tensor_6);  add_tensor_4 = slice_tensor_6 = None
        slice_tensor_7: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_432, 1, 352, 384);  mul_432 = None
        add_tensor_6: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_5, slice_tensor_7);  add_tensor_5 = slice_tensor_7 = None
        slice_tensor_8: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_450, 1, 352, 384);  mul_450 = None
        add_tensor_7: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_6, slice_tensor_8);  add_tensor_6 = slice_tensor_8 = None
        slice_tensor_9: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_468, 1, 352, 384);  mul_468 = None
        add_tensor_8: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_7, slice_tensor_9);  add_tensor_7 = slice_tensor_9 = None
        slice_tensor_10: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_486, 1, 352, 384);  mul_486 = None
        add_tensor_9: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_8, slice_tensor_10);  add_tensor_8 = slice_tensor_10 = None
        slice_tensor_11: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_504, 1, 352, 384);  mul_504 = None
        add_tensor_10: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_9, slice_tensor_11);  add_tensor_9 = slice_tensor_11 = None
        slice_tensor_12: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_522, 1, 352, 384);  mul_522 = None
        add_tensor_11: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_10, slice_tensor_12);  add_tensor_10 = slice_tensor_12 = None
        slice_tensor_13: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_540, 1, 352, 384);  mul_540 = None
        add_tensor_12: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_11, slice_tensor_13);  add_tensor_11 = slice_tensor_13 = None
        slice_tensor_14: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_558, 1, 352, 384);  mul_558 = None
        add_tensor_13: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_12, slice_tensor_14);  add_tensor_12 = slice_tensor_14 = None
        slice_tensor_15: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_576, 1, 352, 384);  mul_576 = None
        add_tensor_14: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_13, slice_tensor_15);  add_tensor_13 = slice_tensor_15 = None
        slice_tensor_16: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_594, 1, 352, 384);  mul_594 = None
        add_tensor_15: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_14, slice_tensor_16);  add_tensor_14 = slice_tensor_16 = None
        slice_tensor_17: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_612, 1, 352, 384);  mul_612 = None
        add_tensor_16: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_15, slice_tensor_17);  add_tensor_15 = slice_tensor_17 = None
        slice_tensor_18: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_630, 1, 352, 384);  mul_630 = None
        add_tensor_17: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_16, slice_tensor_18);  add_tensor_16 = slice_tensor_18 = None
        slice_tensor_19: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_648, 1, 352, 384);  mul_648 = None
        add_tensor_18: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_17, slice_tensor_19);  add_tensor_17 = slice_tensor_19 = None
        le_scalar: "b8[64, 384, 14, 14]" = torch.ops.aten.le.Scalar(arg390_1, 0);  arg390_1 = None
        where_self: "f32[64, 384, 14, 14]" = torch.ops.aten.where.self(le_scalar, full, getitem_216);  le_scalar = full = getitem_216 = None
        sum_dim_int_list: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[64, 384, 14, 14]" = torch.ops.aten.sub.Tensor(arg388_1, arg684_1);  arg388_1 = arg684_1 = None
        mul_tensor: "f32[64, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 7.971938775510203e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 7.971938775510203e-05)
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(arg389_1, arg389_1)
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(arg389_1, arg96_1);  arg96_1 = None
        unsqueeze_default_6: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[64, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[64, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[64, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[64, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg389_1);  sum_dim_int_list_1 = arg389_1 = None
        slice_tensor_20: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 352, 384);  mul_tensor_7 = None
        add_tensor_19: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_18, slice_tensor_20);  add_tensor_18 = slice_tensor_20 = None
        return (mul_tensor_8, add_tensor_19)



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
