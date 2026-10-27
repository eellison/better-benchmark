"""
Standalone repro captured via capture_hook.
Label: DistillGPT2
Pattern hash: 7cbf0c21266b
Shape hash: d0477538
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
    def forward(self, arg0_1: "bf16[32, 512, 50257]", arg1_1: "bf16[50257, 768]", arg2_1: "f32[32, 512, 768]", arg3_1: "f32[768]", arg4_1: "f32[32, 512, 768]", arg5_1: "f32[32, 512, 1]", arg6_1: "b8[32, 512, 768]", arg7_1: "bf16[768, 3072]", arg8_1: "bf16[3072, 16384]", arg9_1: "bf16[16384, 3072]", arg10_1: "bf16[3072, 768]", arg11_1: "bf16[768, 16384]", arg12_1: "f32[768]", arg13_1: "f32[32, 512, 768]", arg14_1: "f32[32, 512, 1]", arg15_1: "b8[32, 512, 768]", arg16_1: "bf16[768, 768]", arg17_1: "bf16[32, 12, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16):
        # No stacktrace found for following nodes
        view: "bf16[16384, 50257]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        constant_pad_nd: "bf16[16384, 50264]" = torch.ops.aten.constant_pad_nd.default(view, _shape_param_1);  _shape_param_1 = None
        constant_pad_nd_1: "bf16[50264, 768]" = torch.ops.aten.constant_pad_nd.default(arg1_1, _shape_param_2);  arg1_1 = _shape_param_2 = None
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        view_1: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add, arg3_1);  arg3_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = None
        sum_2: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg4_1, sum_2);  sum_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg5_1, sub_1);  arg5_1 = sub_1 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add, arg4_1);  arg4_1 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(add, [0, 1]);  add = None
        convert_element_type_1: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        convert_element_type_2: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        mul_6: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_7: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_6);  convert_element_type_1 = mul_6 = None
        view_2: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_7, _shape_param_4);  mul_7 = _shape_param_4 = None
        mm_1: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_2, arg7_1);  arg7_1 = None
        mm_2: "bf16[3072, 768]" = torch.ops.aten.mm.default(arg8_1, view_2);  arg8_1 = None
        sum_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32);  view_2 = None
        view_3: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_5);  sum_5 = _shape_param_5 = None
        convert_element_type_3: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_4: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        convert_element_type_5: "f32[16384, 3072]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_6: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        view_4: "f32[32, 512, 3072]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_6);  convert_element_type_5 = _shape_param_6 = None
        view_5: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg9_1, _shape_param_7);  arg9_1 = _shape_param_7 = None
        mul_8: "bf16[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_5, 0.5)
        mul_9: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_4, mul_8);  mul_8 = None
        convert_element_type_7: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32)
        pow_1: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_7, 3.0)
        mul_10: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_1: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_5, mul_10);  view_5 = mul_10 = None
        mul_11: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_1, 0.7978845608028654);  add_1 = None
        tanh: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_11);  mul_11 = None
        add_2: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_12: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_4, add_2);  view_4 = add_2 = None
        convert_element_type_8: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        mul_13: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_2: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_13);  mul_13 = None
        mul_14: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_9, sub_2);  mul_9 = sub_2 = None
        mul_15: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_14, 0.7978845608028654);  mul_14 = None
        convert_element_type_9: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16)
        mul_16: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_15, 0.044715);  mul_15 = None
        pow_2: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_7, 2.0);  convert_element_type_7 = None
        mul_17: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_18: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        convert_element_type_10: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        add_3: "bf16[32, 512, 3072]" = torch.ops.aten.add.Tensor(convert_element_type_9, convert_element_type_10);  convert_element_type_9 = convert_element_type_10 = None
        mul_19: "bf16[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 0.5);  convert_element_type_8 = None
        add_4: "bf16[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_3, mul_19);  add_3 = mul_19 = None
        view_6: "bf16[16384, 3072]" = torch.ops.aten.view.default(add_4, _shape_param_8);  add_4 = _shape_param_8 = None
        mm_3: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_6, arg10_1);  arg10_1 = None
        mm_4: "bf16[768, 3072]" = torch.ops.aten.mm.default(arg11_1, view_6);  arg11_1 = None
        sum_6: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_6, [0], True, dtype = torch.float32);  view_6 = None
        view_7: "f32[3072]" = torch.ops.aten.view.default(sum_6, _shape_param_9);  sum_6 = _shape_param_9 = None
        convert_element_type_11: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        convert_element_type_12: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        convert_element_type_13: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_14: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        view_8: "f32[32, 512, 768]" = torch.ops.aten.view.default(convert_element_type_13, _shape_param_10);  convert_element_type_13 = _shape_param_10 = None
        mul_20: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_8, arg12_1);  arg12_1 = None
        mul_21: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, 768)
        sum_7: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True)
        mul_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg13_1);  mul_20 = None
        sum_8: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [2], True);  mul_22 = None
        mul_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg13_1, sum_8);  sum_8 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_21, sum_7);  mul_21 = sum_7 = None
        sub_4: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_3, mul_23);  sub_3 = mul_23 = None
        mul_24: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg14_1, sub_4);  arg14_1 = sub_4 = None
        mul_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_8, arg13_1);  arg13_1 = None
        sum_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_25, [0, 1]);  mul_25 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_8, [0, 1]);  view_8 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, mul_24);  mul_4 = mul_24 = None
        convert_element_type_15: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16)
        convert_element_type_16: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.bfloat16);  arg15_1 = None
        mul_26: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_27: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_15, mul_26);  convert_element_type_15 = mul_26 = None
        view_9: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_27, _shape_param_11);  mul_27 = _shape_param_11 = None
        mm_5: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_9, arg16_1);  arg16_1 = None
        permute: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(arg17_1, [0, 2, 1, 3]);  arg17_1 = None
        view_10: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute, _shape_param_12);  permute = _shape_param_12 = None
        view_11: "bf16[16384, 768]" = torch.ops.aten.view.default(view_10, _shape_param_13);  view_10 = _shape_param_13 = None
        permute_1: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_11, [1, 0]);  view_11 = None
        mm_6: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_1, view_9);  permute_1 = None
        sum_11: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_9, [0], True, dtype = torch.float32);  view_9 = None
        view_12: "f32[768]" = torch.ops.aten.view.default(sum_11, _shape_param_14);  sum_11 = _shape_param_14 = None
        convert_element_type_17: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        convert_element_type_18: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_17, torch.float32);  convert_element_type_17 = None
        convert_element_type_19: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        view_13: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_5, _shape_param_15);  mm_5 = _shape_param_15 = None
        view_14: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_13, _shape_param_16);  view_13 = _shape_param_16 = None
        permute_2: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        return (view, sum_3, sum_4, convert_element_type_4, convert_element_type_6, convert_element_type_12, convert_element_type_14, sum_9, sum_10, add_5, convert_element_type_18, convert_element_type_19, permute_2)



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
