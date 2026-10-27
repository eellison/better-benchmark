"""
Standalone repro captured via capture_hook.
Label: DistillGPT2
Pattern hash: dde4cf169b5d
Shape hash: 087e5a68
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
    def forward(self, arg0_1: "bf16[32, 12, 512, 64]", arg1_1: "bf16[32, 12, 512, 64]", arg2_1: "bf16[32, 12, 512, 64]", arg3_1: "bf16[2304, 768]", arg4_1: "bf16[768, 16384]", arg5_1: "f32[768]", arg6_1: "f32[32, 512, 768]", arg7_1: "f32[32, 512, 1]", arg8_1: "f32[32, 512, 768]", arg9_1: "b8[32, 512, 768]", arg10_1: "bf16[768, 3072]", arg11_1: "bf16[3072, 16384]", arg12_1: "bf16[16384, 3072]", arg13_1: "bf16[3072, 768]", arg14_1: "bf16[768, 16384]", arg15_1: "f32[768]", arg16_1: "f32[32, 512, 768]", arg17_1: "f32[32, 512, 1]", arg18_1: "b8[32, 512, 768]", arg19_1: "bf16[768, 768]", arg20_1: "bf16[32, 12, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18):
        # No stacktrace found for following nodes
        permute: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        permute_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
        view_1: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_1, _shape_param_1);  permute_1 = _shape_param_1 = None
        permute_2: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        view_2: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_2, _shape_param_2);  permute_2 = _shape_param_2 = None
        cat: "bf16[32, 512, 2304]" = torch.ops.aten.cat.default([view, view_2, view_1], 2);  view = view_2 = view_1 = None
        view_3: "bf16[16384, 2304]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_3, arg3_1);  arg3_1 = None
        mm_1: "bf16[768, 2304]" = torch.ops.aten.mm.default(arg4_1, view_3);  arg4_1 = None
        sum_1: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32);  view_3 = None
        view_4: "f32[2304]" = torch.ops.aten.view.default(sum_1, _shape_param_4);  sum_1 = _shape_param_4 = None
        convert_element_type: "bf16[2304]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_1: "f32[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        convert_element_type_2: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_3: "f32[768, 2304]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        view_5: "f32[32, 512, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_5);  convert_element_type_2 = _shape_param_5 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_5, arg5_1);  arg5_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_2: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = None
        sum_3: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg6_1, sum_3);  sum_3 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_2);  mul_1 = sum_2 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg7_1, sub_1);  arg7_1 = sub_1 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_5, arg6_1);  arg6_1 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_5, [0, 1]);  view_5 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg8_1, mul_4);  arg8_1 = mul_4 = None
        convert_element_type_4: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16)
        convert_element_type_5: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        mul_6: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_7: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, mul_6);  convert_element_type_4 = mul_6 = None
        view_6: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_7, _shape_param_6);  mul_7 = _shape_param_6 = None
        mm_2: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_6, arg10_1);  arg10_1 = None
        mm_3: "bf16[3072, 768]" = torch.ops.aten.mm.default(arg11_1, view_6);  arg11_1 = None
        sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_6, [0], True, dtype = torch.float32);  view_6 = None
        view_7: "f32[768]" = torch.ops.aten.view.default(sum_6, _shape_param_7);  sum_6 = _shape_param_7 = None
        convert_element_type_6: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        convert_element_type_7: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        convert_element_type_8: "f32[16384, 3072]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_9: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        view_8: "f32[32, 512, 3072]" = torch.ops.aten.view.default(convert_element_type_8, _shape_param_8);  convert_element_type_8 = _shape_param_8 = None
        view_9: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg12_1, _shape_param_9);  arg12_1 = _shape_param_9 = None
        mul_8: "bf16[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_9, 0.5)
        mul_9: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_8, mul_8);  mul_8 = None
        convert_element_type_10: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32)
        pow_1: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_10, 3.0)
        mul_10: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_1: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_9, mul_10);  view_9 = mul_10 = None
        mul_11: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_1, 0.7978845608028654);  add_1 = None
        tanh: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_11);  mul_11 = None
        add_2: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_12: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_8, add_2);  view_8 = add_2 = None
        convert_element_type_11: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        mul_13: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_2: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_13);  mul_13 = None
        mul_14: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_9, sub_2);  mul_9 = sub_2 = None
        mul_15: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_14, 0.7978845608028654);  mul_14 = None
        convert_element_type_12: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16)
        mul_16: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_15, 0.044715);  mul_15 = None
        pow_2: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_10, 2.0);  convert_element_type_10 = None
        mul_17: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_18: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        convert_element_type_13: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        add_3: "bf16[32, 512, 3072]" = torch.ops.aten.add.Tensor(convert_element_type_12, convert_element_type_13);  convert_element_type_12 = convert_element_type_13 = None
        mul_19: "bf16[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.5);  convert_element_type_11 = None
        add_4: "bf16[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_3, mul_19);  add_3 = mul_19 = None
        view_10: "bf16[16384, 3072]" = torch.ops.aten.view.default(add_4, _shape_param_10);  add_4 = _shape_param_10 = None
        mm_4: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_10, arg13_1);  arg13_1 = None
        mm_5: "bf16[768, 3072]" = torch.ops.aten.mm.default(arg14_1, view_10);  arg14_1 = None
        sum_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_10, [0], True, dtype = torch.float32);  view_10 = None
        view_11: "f32[3072]" = torch.ops.aten.view.default(sum_7, _shape_param_11);  sum_7 = _shape_param_11 = None
        convert_element_type_14: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_11, torch.bfloat16);  view_11 = None
        convert_element_type_15: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        convert_element_type_16: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_17: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        view_12: "f32[32, 512, 768]" = torch.ops.aten.view.default(convert_element_type_16, _shape_param_12);  convert_element_type_16 = _shape_param_12 = None
        mul_20: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_12, arg15_1);  arg15_1 = None
        mul_21: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, 768)
        sum_8: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True)
        mul_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg16_1);  mul_20 = None
        sum_9: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [2], True);  mul_22 = None
        mul_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg16_1, sum_9);  sum_9 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_21, sum_8);  mul_21 = sum_8 = None
        sub_4: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_3, mul_23);  sub_3 = mul_23 = None
        mul_24: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg17_1, sub_4);  arg17_1 = sub_4 = None
        mul_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_12, arg16_1);  arg16_1 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_25, [0, 1]);  mul_25 = None
        sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_12, [0, 1]);  view_12 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, mul_24);  add = mul_24 = None
        convert_element_type_18: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16)
        convert_element_type_19: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        mul_26: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_27: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_18, mul_26);  convert_element_type_18 = mul_26 = None
        view_13: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_27, _shape_param_13);  mul_27 = _shape_param_13 = None
        mm_6: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_13, arg19_1);  arg19_1 = None
        permute_3: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(arg20_1, [0, 2, 1, 3]);  arg20_1 = None
        view_14: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_3, _shape_param_14);  permute_3 = _shape_param_14 = None
        view_15: "bf16[16384, 768]" = torch.ops.aten.view.default(view_14, _shape_param_15);  view_14 = _shape_param_15 = None
        permute_4: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_15, [1, 0]);  view_15 = None
        mm_7: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_4, view_13);  permute_4 = None
        sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_13, [0], True, dtype = torch.float32);  view_13 = None
        view_16: "f32[768]" = torch.ops.aten.view.default(sum_12, _shape_param_16);  sum_12 = _shape_param_16 = None
        convert_element_type_20: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_16, torch.bfloat16);  view_16 = None
        convert_element_type_21: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32);  convert_element_type_20 = None
        convert_element_type_22: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        view_17: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_6, _shape_param_17);  mm_6 = _shape_param_17 = None
        view_18: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_17, _shape_param_18);  view_17 = _shape_param_18 = None
        permute_5: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        return (convert_element_type_1, convert_element_type_3, sum_4, sum_5, convert_element_type_7, convert_element_type_9, convert_element_type_15, convert_element_type_17, sum_10, sum_11, add_5, convert_element_type_21, convert_element_type_22, permute_5)



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
