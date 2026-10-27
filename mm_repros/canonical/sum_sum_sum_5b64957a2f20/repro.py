"""
Standalone repro captured via capture_hook.
Label: mobilevit_s
Pattern hash: 5b64957a2f20
Shape hash: 72d3dde0
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
    def forward(self, arg0_1: "bf16[512, 4, 16, 64]", arg1_1: "bf16[512, 4, 16, 64]", arg2_1: "bf16[512, 4, 16, 64]", arg3_1: "bf16[720, 240]", arg4_1: "bf16[8192, 240]", arg5_1: "f32[240]", arg6_1: "bf16[512, 16, 240]", arg7_1: "f32[512, 16, 1]", arg8_1: "f32[512, 16, 1]", arg9_1: "bf16[512, 16, 240]", arg10_1: "bf16[240, 480]", arg11_1: "bf16[8192, 480]", arg12_1: "bf16[8192, 480]", arg13_1: "bf16[480, 240]", arg14_1: "bf16[8192, 240]", arg15_1: "f32[240]", arg16_1: "bf16[512, 16, 240]", arg17_1: "f32[512, 16, 1]", arg18_1: "f32[512, 16, 1]", arg19_1: "bf16[240, 240]", arg20_1: "bf16[8192, 240]", arg21_1: "bf16[512, 4, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18):
        # No stacktrace found for following nodes
        constant_pad_nd: "bf16[512, 4, 16, 60]" = torch.ops.aten.constant_pad_nd.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        constant_pad_nd_1: "bf16[512, 4, 16, 60]" = torch.ops.aten.constant_pad_nd.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        constant_pad_nd_2: "bf16[512, 4, 16, 60]" = torch.ops.aten.constant_pad_nd.default(arg2_1, _shape_param_2);  arg2_1 = _shape_param_2 = None
        cat: "bf16[1536, 4, 16, 60]" = torch.ops.aten.cat.default([constant_pad_nd_2, constant_pad_nd_1, constant_pad_nd]);  constant_pad_nd_2 = constant_pad_nd_1 = constant_pad_nd = None
        view: "bf16[3, 512, 4, 16, 60]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        permute: "bf16[512, 16, 3, 4, 60]" = torch.ops.aten.permute.default(view, [1, 3, 0, 2, 4]);  view = None
        clone: "bf16[512, 16, 3, 4, 60]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[512, 16, 720]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        view_2: "bf16[8192, 720]" = torch.ops.aten.view.default(view_1, _shape_param_5);  view_1 = _shape_param_5 = None
        mm: "bf16[8192, 240]" = torch.ops.aten.mm.default(view_2, arg3_1);  arg3_1 = None
        permute_1: "bf16[720, 8192]" = torch.ops.aten.permute.default(view_2, [1, 0])
        mm_1: "bf16[720, 240]" = torch.ops.aten.mm.default(permute_1, arg4_1);  permute_1 = arg4_1 = None
        sum_1: "f32[1, 720]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32);  view_2 = None
        view_3: "f32[720]" = torch.ops.aten.view.default(sum_1, _shape_param_6);  sum_1 = _shape_param_6 = None
        convert_element_type: "bf16[720]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        view_4: "bf16[512, 16, 240]" = torch.ops.aten.view.default(mm, _shape_param_7);  mm = _shape_param_7 = None
        convert_element_type_1: "f32[512, 16, 240]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        convert_element_type_2: "f32[720, 240]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_3: "f32[720]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        mul: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(convert_element_type_1, arg5_1);  arg5_1 = None
        mul_1: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul, 240)
        sum_2: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        convert_element_type_4: "f32[512, 16, 240]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        sub: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(convert_element_type_4, arg7_1);  convert_element_type_4 = arg7_1 = None
        mul_2: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub, arg8_1);  sub = None
        mul_3: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_3: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True);  mul_3 = None
        mul_4: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_2, sum_3);  sum_3 = None
        sub_1: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(mul_1, sum_2);  mul_1 = sum_2 = None
        sub_2: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(arg8_1, 240);  arg8_1 = None
        mul_5: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_2);  mul_2 = None
        sum_4: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_5: "f32[240]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [0, 1]);  convert_element_type_1 = None
        convert_element_type_5: "bf16[512, 16, 240]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        add: "bf16[512, 16, 240]" = torch.ops.aten.add.Tensor(arg9_1, convert_element_type_5);  arg9_1 = convert_element_type_5 = None
        view_5: "bf16[8192, 240]" = torch.ops.aten.view.default(add, _shape_param_8);  _shape_param_8 = None
        mm_2: "bf16[8192, 480]" = torch.ops.aten.mm.default(view_5, arg10_1);  arg10_1 = None
        permute_2: "bf16[240, 8192]" = torch.ops.aten.permute.default(view_5, [1, 0])
        mm_3: "bf16[240, 480]" = torch.ops.aten.mm.default(permute_2, arg11_1);  permute_2 = arg11_1 = None
        sum_6: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32);  view_5 = None
        view_6: "f32[240]" = torch.ops.aten.view.default(sum_6, _shape_param_9);  sum_6 = _shape_param_9 = None
        convert_element_type_6: "bf16[240]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        view_7: "bf16[512, 16, 480]" = torch.ops.aten.view.default(mm_2, _shape_param_10);  mm_2 = _shape_param_10 = None
        convert_element_type_7: "f32[240, 480]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_8: "f32[240]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        convert_element_type_9: "f32[512, 16, 480]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        view_8: "bf16[512, 16, 480]" = torch.ops.aten.view.default(arg12_1, _shape_param_11);  arg12_1 = _shape_param_11 = None
        convert_element_type_10: "f32[512, 16, 480]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        sigmoid: "f32[512, 16, 480]" = torch.ops.aten.sigmoid.default(convert_element_type_10)
        mul_7: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(convert_element_type_9, sigmoid);  convert_element_type_9 = None
        sub_3: "f32[512, 16, 480]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_8: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(convert_element_type_10, sub_3);  convert_element_type_10 = sub_3 = None
        add_1: "f32[512, 16, 480]" = torch.ops.aten.add.Tensor(mul_8, 1);  mul_8 = None
        mul_9: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(mul_7, add_1);  mul_7 = add_1 = None
        convert_element_type_11: "bf16[512, 16, 480]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        view_9: "bf16[8192, 480]" = torch.ops.aten.view.default(convert_element_type_11, _shape_param_12);  convert_element_type_11 = _shape_param_12 = None
        mm_4: "bf16[8192, 240]" = torch.ops.aten.mm.default(view_9, arg13_1);  arg13_1 = None
        permute_3: "bf16[480, 8192]" = torch.ops.aten.permute.default(view_9, [1, 0])
        mm_5: "bf16[480, 240]" = torch.ops.aten.mm.default(permute_3, arg14_1);  permute_3 = arg14_1 = None
        sum_7: "f32[1, 480]" = torch.ops.aten.sum.dim_IntList(view_9, [0], True, dtype = torch.float32);  view_9 = None
        view_10: "f32[480]" = torch.ops.aten.view.default(sum_7, _shape_param_13);  sum_7 = _shape_param_13 = None
        convert_element_type_12: "bf16[480]" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        view_11: "bf16[512, 16, 240]" = torch.ops.aten.view.default(mm_4, _shape_param_14);  mm_4 = _shape_param_14 = None
        convert_element_type_13: "f32[512, 16, 240]" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        convert_element_type_14: "f32[480, 240]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_15: "f32[480]" = torch.ops.prims.convert_element_type.default(convert_element_type_12, torch.float32);  convert_element_type_12 = None
        mul_10: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(convert_element_type_13, arg15_1);  arg15_1 = None
        mul_11: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_10, 240)
        sum_8: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_10, [2], True)
        convert_element_type_16: "f32[512, 16, 240]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.float32);  arg16_1 = None
        sub_4: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(convert_element_type_16, arg17_1);  convert_element_type_16 = arg17_1 = None
        mul_12: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub_4, arg18_1);  sub_4 = None
        mul_13: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_10, mul_12);  mul_10 = None
        sum_9: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_13, [2], True);  mul_13 = None
        mul_14: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_12, sum_9);  sum_9 = None
        sub_5: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(mul_11, sum_8);  mul_11 = sum_8 = None
        sub_6: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(sub_5, mul_14);  sub_5 = mul_14 = None
        div_1: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(arg18_1, 240);  arg18_1 = None
        mul_15: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(div_1, sub_6);  div_1 = sub_6 = None
        mul_16: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(convert_element_type_13, mul_12);  mul_12 = None
        sum_10: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_16, [0, 1]);  mul_16 = None
        sum_11: "f32[240]" = torch.ops.aten.sum.dim_IntList(convert_element_type_13, [0, 1]);  convert_element_type_13 = None
        convert_element_type_17: "bf16[512, 16, 240]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        add_2: "bf16[512, 16, 240]" = torch.ops.aten.add.Tensor(add, convert_element_type_17);  add = convert_element_type_17 = None
        view_12: "bf16[8192, 240]" = torch.ops.aten.view.default(add_2, _shape_param_15);  _shape_param_15 = None
        mm_6: "bf16[8192, 240]" = torch.ops.aten.mm.default(view_12, arg19_1);  arg19_1 = None
        permute_4: "bf16[240, 8192]" = torch.ops.aten.permute.default(view_12, [1, 0])
        mm_7: "bf16[240, 240]" = torch.ops.aten.mm.default(permute_4, arg20_1);  permute_4 = arg20_1 = None
        sum_12: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_12, [0], True, dtype = torch.float32);  view_12 = None
        view_13: "f32[240]" = torch.ops.aten.view.default(sum_12, _shape_param_16);  sum_12 = _shape_param_16 = None
        convert_element_type_18: "bf16[240]" = torch.ops.prims.convert_element_type.default(view_13, torch.bfloat16);  view_13 = None
        view_14: "bf16[512, 16, 240]" = torch.ops.aten.view.default(mm_6, _shape_param_17);  mm_6 = _shape_param_17 = None
        convert_element_type_19: "f32[240, 240]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_20: "f32[240]" = torch.ops.prims.convert_element_type.default(convert_element_type_18, torch.float32);  convert_element_type_18 = None
        view_15: "bf16[512, 16, 4, 60]" = torch.ops.aten.view.default(view_14, _shape_param_18);  view_14 = _shape_param_18 = None
        permute_5: "bf16[512, 4, 16, 60]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        slice_scatter: "bf16[512, 4, 16, 64]" = torch.ops.aten.slice_scatter.default(arg21_1, permute_5, -1, 0, 60);  arg21_1 = permute_5 = None
        return (convert_element_type_2, convert_element_type_3, sum_4, sum_5, convert_element_type_7, convert_element_type_8, convert_element_type_14, convert_element_type_15, sum_10, sum_11, add_2, convert_element_type_19, convert_element_type_20, slice_scatter)



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
