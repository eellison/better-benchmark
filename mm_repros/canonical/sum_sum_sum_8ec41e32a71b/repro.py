"""
Standalone repro captured via capture_hook.
Label: vit_base_patch14_dinov2.lvd142m
Pattern hash: 8ec41e32a71b
Shape hash: e6b34d63
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
    def forward(self, arg0_1: "bf16[128, 12, 1370, 64]", arg1_1: "bf16[128, 12, 1370, 64]", arg2_1: "bf16[128, 12, 1370, 64]", arg3_1: "bf16[2304, 768]", arg4_1: "bf16[175360, 768]", arg5_1: "f32[768]", arg6_1: "f32[128, 1370, 768]", arg7_1: "f32[128, 1370, 1]", arg8_1: "f32[128, 1370, 768]", arg9_1: "bf16[175360, 768]", arg10_1: "f32[768]", arg11_1: "bf16[768, 3072]", arg12_1: "bf16[175360, 3072]", arg13_1: "bf16[175360, 3072]", arg14_1: "bf16[3072, 768]", arg15_1: "bf16[175360, 768]", arg16_1: "f32[768]", arg17_1: "f32[128, 1370, 768]", arg18_1: "f32[128, 1370, 1]", arg19_1: "bf16[175360, 768]", arg20_1: "f32[768]", arg21_1: "bf16[768, 768]", arg22_1: "bf16[128, 12, 1370, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21):
        # No stacktrace found for following nodes
        cat: "bf16[384, 12, 1370, 64]" = torch.ops.aten.cat.default([arg0_1, arg1_1, arg2_1]);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[3, 128, 12, 1370, 64]" = torch.ops.aten.view.default(cat, _shape_param_0);  cat = _shape_param_0 = None
        permute: "bf16[128, 1370, 3, 12, 64]" = torch.ops.aten.permute.default(view, [1, 3, 0, 2, 4]);  view = None
        clone: "bf16[128, 1370, 3, 12, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[128, 1370, 2304]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "bf16[175360, 2304]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        mm: "bf16[175360, 768]" = torch.ops.aten.mm.default(view_2, arg3_1);  arg3_1 = None
        permute_1: "bf16[2304, 175360]" = torch.ops.aten.permute.default(view_2, [1, 0])
        mm_1: "bf16[2304, 768]" = torch.ops.aten.mm.default(permute_1, arg4_1);  permute_1 = arg4_1 = None
        sum_1: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32);  view_2 = None
        view_3: "f32[2304]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type: "bf16[2304]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        view_4: "bf16[128, 1370, 768]" = torch.ops.aten.view.default(mm, _shape_param_4);  mm = _shape_param_4 = None
        convert_element_type_1: "f32[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        convert_element_type_2: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_3: "f32[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        mul: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, arg5_1);  arg5_1 = None
        mul_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_2: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = None
        sum_3: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg6_1, sum_3);  sum_3 = None
        sub: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_2);  mul_1 = sum_2 = None
        sub_1: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg7_1, sub_1);  arg7_1 = sub_1 = None
        mul_5: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, arg6_1);  arg6_1 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [0, 1]);  convert_element_type_1 = None
        add: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(arg8_1, mul_4);  arg8_1 = mul_4 = None
        view_5: "bf16[128, 1370, 768]" = torch.ops.aten.view.default(arg9_1, _shape_param_5);  arg9_1 = _shape_param_5 = None
        mul_6: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(add, view_5);  view_5 = None
        mul_7: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(add, arg10_1);  arg10_1 = None
        convert_element_type_4: "bf16[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        sum_6: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1], True, dtype = torch.float32);  mul_6 = None
        view_6: "f32[768]" = torch.ops.aten.view.default(sum_6, _shape_param_6);  sum_6 = _shape_param_6 = None
        view_7: "bf16[175360, 768]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_7);  convert_element_type_4 = _shape_param_7 = None
        mm_2: "bf16[175360, 3072]" = torch.ops.aten.mm.default(view_7, arg11_1);  arg11_1 = None
        permute_2: "bf16[768, 175360]" = torch.ops.aten.permute.default(view_7, [1, 0])
        mm_3: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_2, arg12_1);  permute_2 = arg12_1 = None
        sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True, dtype = torch.float32);  view_7 = None
        view_8: "f32[768]" = torch.ops.aten.view.default(sum_7, _shape_param_8);  sum_7 = _shape_param_8 = None
        convert_element_type_5: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        view_9: "bf16[128, 1370, 3072]" = torch.ops.aten.view.default(mm_2, _shape_param_9);  mm_2 = _shape_param_9 = None
        convert_element_type_6: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_7: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        convert_element_type_8: "f32[128, 1370, 3072]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        view_10: "bf16[128, 1370, 3072]" = torch.ops.aten.view.default(arg13_1, _shape_param_10);  arg13_1 = _shape_param_10 = None
        convert_element_type_9: "f32[128, 1370, 3072]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        mul_8: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 0.7071067811865476)
        erf: "f32[128, 1370, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_1: "f32[128, 1370, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_10: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_9, convert_element_type_9)
        mul_11: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(mul_10, -0.5);  mul_10 = None
        exp: "f32[128, 1370, 3072]" = torch.ops.aten.exp.default(mul_11);  mul_11 = None
        mul_12: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_13: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_9, mul_12);  convert_element_type_9 = mul_12 = None
        add_2: "f32[128, 1370, 3072]" = torch.ops.aten.add.Tensor(mul_9, mul_13);  mul_9 = mul_13 = None
        mul_14: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_8, add_2);  convert_element_type_8 = add_2 = None
        convert_element_type_10: "bf16[128, 1370, 3072]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        view_11: "bf16[175360, 3072]" = torch.ops.aten.view.default(convert_element_type_10, _shape_param_11);  convert_element_type_10 = _shape_param_11 = None
        mm_4: "bf16[175360, 768]" = torch.ops.aten.mm.default(view_11, arg14_1);  arg14_1 = None
        permute_3: "bf16[3072, 175360]" = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_5: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_3, arg15_1);  permute_3 = arg15_1 = None
        sum_8: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True, dtype = torch.float32);  view_11 = None
        view_12: "f32[3072]" = torch.ops.aten.view.default(sum_8, _shape_param_12);  sum_8 = _shape_param_12 = None
        convert_element_type_11: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        view_13: "bf16[128, 1370, 768]" = torch.ops.aten.view.default(mm_4, _shape_param_13);  mm_4 = _shape_param_13 = None
        convert_element_type_12: "f32[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        convert_element_type_13: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_14: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        mul_15: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, arg16_1);  arg16_1 = None
        mul_16: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_15, 768)
        sum_9: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True)
        mul_17: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_15, arg17_1);  mul_15 = None
        sum_10: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_17, [2], True);  mul_17 = None
        mul_18: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg17_1, sum_10);  sum_10 = None
        sub_2: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(mul_16, sum_9);  mul_16 = sum_9 = None
        sub_3: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(sub_2, mul_18);  sub_2 = mul_18 = None
        mul_19: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg18_1, sub_3);  arg18_1 = sub_3 = None
        mul_20: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, arg17_1);  arg17_1 = None
        sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_20, [0, 1]);  mul_20 = None
        sum_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_12, [0, 1]);  convert_element_type_12 = None
        add_3: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(add, mul_19);  add = mul_19 = None
        view_14: "bf16[128, 1370, 768]" = torch.ops.aten.view.default(arg19_1, _shape_param_14);  arg19_1 = _shape_param_14 = None
        mul_21: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(add_3, view_14);  view_14 = None
        mul_22: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(add_3, arg20_1);  arg20_1 = None
        convert_element_type_15: "bf16[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        sum_13: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_21, [0, 1], True, dtype = torch.float32);  mul_21 = None
        view_15: "f32[768]" = torch.ops.aten.view.default(sum_13, _shape_param_15);  sum_13 = _shape_param_15 = None
        view_16: "bf16[175360, 768]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_16);  convert_element_type_15 = _shape_param_16 = None
        mm_6: "bf16[175360, 768]" = torch.ops.aten.mm.default(view_16, arg21_1);  arg21_1 = None
        permute_4: "bf16[768, 175360]" = torch.ops.aten.permute.default(view_16, [1, 0])
        permute_5: "bf16[128, 1370, 12, 64]" = torch.ops.aten.permute.default(arg22_1, [0, 2, 1, 3]);  arg22_1 = None
        view_17: "bf16[128, 1370, 768]" = torch.ops.aten.view.default(permute_5, _shape_param_17);  permute_5 = _shape_param_17 = None
        view_18: "bf16[175360, 768]" = torch.ops.aten.view.default(view_17, _shape_param_18);  view_17 = _shape_param_18 = None
        mm_7: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_4, view_18);  permute_4 = view_18 = None
        sum_14: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_16, [0], True, dtype = torch.float32);  view_16 = None
        view_19: "f32[768]" = torch.ops.aten.view.default(sum_14, _shape_param_19);  sum_14 = _shape_param_19 = None
        convert_element_type_16: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_19, torch.bfloat16);  view_19 = None
        view_20: "bf16[128, 1370, 768]" = torch.ops.aten.view.default(mm_6, _shape_param_20);  mm_6 = _shape_param_20 = None
        convert_element_type_17: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_18: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_16, torch.float32);  convert_element_type_16 = None
        view_21: "bf16[128, 1370, 12, 64]" = torch.ops.aten.view.default(view_20, _shape_param_21);  view_20 = _shape_param_21 = None
        permute_6: "bf16[128, 12, 1370, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        return (convert_element_type_2, convert_element_type_3, sum_4, sum_5, view_6, convert_element_type_6, convert_element_type_7, convert_element_type_13, convert_element_type_14, sum_11, sum_12, add_3, view_15, convert_element_type_17, convert_element_type_18, permute_6)



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
