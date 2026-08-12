"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train
Pattern hash: 9be9812ec7f1
Shape hash: 66cb4e6d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4, 1024, 14, 14]", arg1_1: "bf16[4, 992, 14, 14]", arg2_1: "bf16[4, 960, 14, 14]", arg3_1: "bf16[4, 928, 14, 14]", arg4_1: "bf16[4, 896, 14, 14]", arg5_1: "bf16[4, 864, 14, 14]", arg6_1: "bf16[4, 832, 14, 14]", arg7_1: "bf16[4, 800, 14, 14]", arg8_1: "bf16[4, 768, 14, 14]", arg9_1: "bf16[4, 736, 14, 14]", arg10_1: "bf16[4, 704, 14, 14]", arg11_1: "bf16[4, 672, 14, 14]", arg12_1: "bf16[4, 640, 14, 14]", arg13_1: "bf16[4, 608, 14, 14]", arg14_1: "bf16[4, 576, 14, 14]", arg15_1: "bf16[4, 544, 14, 14]", arg16_1: "bf16[4, 512, 14, 14]", arg17_1: "bf16[4, 480, 14, 14]", arg18_1: "bf16[4, 448, 14, 14]", arg19_1: "bf16[4, 416, 14, 14]", arg20_1: "bf16[4, 384, 14, 14]", arg21_1: "bf16[4, 352, 14, 14]", arg22_1: "bf16[4, 320, 14, 14]", arg23_1: "bf16[]", arg24_1: "bf16[4, 320, 14, 14]", arg25_1: "bf16[4, 320, 14, 14]", arg26_1: "f32[1, 320, 1, 1]", arg27_1: "f32[320]", arg28_1: "f32[320]"):
        # No stacktrace found for following nodes
        slice_1: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 288, 320);  arg0_1 = None
        slice_2: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 288, 320);  arg1_1 = None
        add: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_1, slice_2);  slice_1 = slice_2 = None
        slice_3: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 288, 320);  arg2_1 = None
        add_1: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add, slice_3);  add = slice_3 = None
        slice_4: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg3_1, 1, 288, 320);  arg3_1 = None
        add_2: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_1, slice_4);  add_1 = slice_4 = None
        slice_5: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg4_1, 1, 288, 320);  arg4_1 = None
        add_3: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_2, slice_5);  add_2 = slice_5 = None
        slice_6: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg5_1, 1, 288, 320);  arg5_1 = None
        add_4: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_3, slice_6);  add_3 = slice_6 = None
        slice_7: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg6_1, 1, 288, 320);  arg6_1 = None
        add_5: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_4, slice_7);  add_4 = slice_7 = None
        slice_8: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg7_1, 1, 288, 320);  arg7_1 = None
        add_6: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_5, slice_8);  add_5 = slice_8 = None
        slice_9: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg8_1, 1, 288, 320);  arg8_1 = None
        add_7: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_6, slice_9);  add_6 = slice_9 = None
        slice_10: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg9_1, 1, 288, 320);  arg9_1 = None
        add_8: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_7, slice_10);  add_7 = slice_10 = None
        slice_11: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg10_1, 1, 288, 320);  arg10_1 = None
        add_9: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_8, slice_11);  add_8 = slice_11 = None
        slice_12: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg11_1, 1, 288, 320);  arg11_1 = None
        add_10: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_9, slice_12);  add_9 = slice_12 = None
        slice_13: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg12_1, 1, 288, 320);  arg12_1 = None
        add_11: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_10, slice_13);  add_10 = slice_13 = None
        slice_14: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg13_1, 1, 288, 320);  arg13_1 = None
        add_12: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_11, slice_14);  add_11 = slice_14 = None
        slice_15: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg14_1, 1, 288, 320);  arg14_1 = None
        add_13: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_12, slice_15);  add_12 = slice_15 = None
        slice_16: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg15_1, 1, 288, 320);  arg15_1 = None
        add_14: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_13, slice_16);  add_13 = slice_16 = None
        slice_17: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg16_1, 1, 288, 320);  arg16_1 = None
        add_15: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_14, slice_17);  add_14 = slice_17 = None
        slice_18: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg17_1, 1, 288, 320);  arg17_1 = None
        add_16: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_15, slice_18);  add_15 = slice_18 = None
        slice_19: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg18_1, 1, 288, 320);  arg18_1 = None
        add_17: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_16, slice_19);  add_16 = slice_19 = None
        slice_20: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg19_1, 1, 288, 320);  arg19_1 = None
        add_18: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_17, slice_20);  add_17 = slice_20 = None
        slice_21: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg20_1, 1, 288, 320);  arg20_1 = None
        add_19: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_18, slice_21);  add_18 = slice_21 = None
        slice_22: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg21_1, 1, 288, 320);  arg21_1 = None
        add_20: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_19, slice_22);  add_19 = slice_22 = None
        le: "b8[4, 320, 14, 14]" = torch.ops.aten.le.Scalar(arg22_1, 0);  arg22_1 = None
        where: "bf16[4, 320, 14, 14]" = torch.ops.aten.where.self(le, arg23_1, arg24_1);  le = arg23_1 = arg24_1 = None
        convert_element_type: "f32[4, 320, 14, 14]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        sum_1: "f32[320]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 2, 3])
        convert_element_type_1: "f32[4, 320, 14, 14]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.float32);  arg25_1 = None
        sub: "f32[4, 320, 14, 14]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg26_1);  convert_element_type_1 = arg26_1 = None
        mul: "f32[4, 320, 14, 14]" = torch.ops.aten.mul.Tensor(convert_element_type, sub)
        sum_2: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul, [0, 2, 3]);  mul = None
        mul_1: "f32[320]" = torch.ops.aten.mul.Tensor(sum_1, 0.0012755102040816326)
        unsqueeze: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_1, 0);  mul_1 = None
        unsqueeze_1: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_2: "f32[320]" = torch.ops.aten.mul.Tensor(sum_2, 0.0012755102040816326)
        mul_3: "f32[320]" = torch.ops.aten.mul.Tensor(arg27_1, arg27_1)
        mul_4: "f32[320]" = torch.ops.aten.mul.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        unsqueeze_3: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_4, 0);  mul_4 = None
        unsqueeze_4: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_5: "f32[320]" = torch.ops.aten.mul.Tensor(arg27_1, arg28_1);  arg28_1 = None
        unsqueeze_6: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_7: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_6: "f32[4, 320, 14, 14]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_5);  sub = unsqueeze_5 = None
        sub_1: "f32[4, 320, 14, 14]" = torch.ops.aten.sub.Tensor(convert_element_type, mul_6);  convert_element_type = mul_6 = None
        sub_2: "f32[4, 320, 14, 14]" = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_2);  sub_1 = unsqueeze_2 = None
        mul_7: "f32[4, 320, 14, 14]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_8);  sub_2 = unsqueeze_8 = None
        mul_8: "f32[320]" = torch.ops.aten.mul.Tensor(sum_2, arg27_1);  sum_2 = arg27_1 = None
        convert_element_type_2: "bf16[4, 320, 14, 14]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        slice_23: "bf16[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(convert_element_type_2, 1, 288, 320)
        add_21: "bf16[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_20, slice_23);  add_20 = slice_23 = None
        return (sum_1, mul_8, convert_element_type_2, add_21)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
