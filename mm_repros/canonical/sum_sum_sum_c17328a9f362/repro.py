"""
Standalone repro captured via capture_hook.
Label: nvidia_deeprecommender
Pattern hash: c17328a9f362
Shape hash: c0ff2382
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
    def forward(self, arg0_1: "bf16[256, 197951]", arg1_1: "bf16[256, 197951]", arg2_1: "bf16[197951, 512]", arg3_1: "bf16[256, 512]", arg4_1: "bf16[256, 512]", arg5_1: "bf16[512, 512]", arg6_1: "bf16[256, 512]", arg7_1: "bf16[256, 512]", arg8_1: "bf16[512, 1024]", arg9_1: "bf16[256, 1024]", arg10_1: "b8[256, 1024]", arg11_1: "bf16[256, 1024]", arg12_1: "bf16[1024, 512]", arg13_1: "bf16[256, 512]", arg14_1: "bf16[256, 512]", arg15_1: "bf16[512, 512]", arg16_1: "bf16[256, 512]", arg17_1: "bf16[256, 512]", arg18_1: "bf16[256, 197951]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        convert_element_type: "f32[256, 197951]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[256, 197951]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        le: "b8[256, 197951]" = torch.ops.aten.le.Scalar(convert_element_type_1, 0)
        mul: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(convert_element_type, 1)
        mul_1: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(mul, 1.7580993408473766);  mul = None
        mul_2: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        exp: "f32[256, 197951]" = torch.ops.aten.exp.default(mul_2);  mul_2 = None
        mul_3: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(mul_1, exp);  mul_1 = exp = None
        mul_4: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.0507009873554805);  convert_element_type = None
        where: "f32[256, 197951]" = torch.ops.aten.where.self(le, mul_3, mul_4);  le = mul_3 = mul_4 = None
        convert_element_type_2: "bf16[256, 197951]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        constant_pad_nd: "bf16[256, 197952]" = torch.ops.aten.constant_pad_nd.default(convert_element_type_2, [0, 1, 0, 0])
        constant_pad_nd_1: "bf16[197952, 512]" = torch.ops.aten.constant_pad_nd.default(arg2_1, [0, 0, 0, 1]);  arg2_1 = None
        mm: "bf16[256, 512]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        permute: "bf16[197951, 256]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0])
        constant_pad_nd_2: "bf16[197952, 256]" = torch.ops.aten.constant_pad_nd.default(permute, [0, 0, 0, 1]);  permute = None
        mm_1: "bf16[197952, 512]" = torch.ops.aten.mm.default(constant_pad_nd_2, arg3_1);  constant_pad_nd_2 = arg3_1 = None
        slice_1: "bf16[197951, 512]" = torch.ops.aten.slice.Tensor(mm_1, 0, 0, -1);  mm_1 = None
        sum_1: "f32[1, 197951]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0], True, dtype = torch.float32);  convert_element_type_2 = None
        view: "f32[197951]" = torch.ops.aten.view.default(sum_1, _shape_param_0);  sum_1 = _shape_param_0 = None
        convert_element_type_3: "bf16[197951]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_4: "f32[197951, 512]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        convert_element_type_5: "f32[197951]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        convert_element_type_6: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_7: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        le_1: "b8[256, 512]" = torch.ops.aten.le.Scalar(convert_element_type_7, 0)
        mul_5: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1)
        mul_6: "f32[256, 512]" = torch.ops.aten.mul.Tensor(mul_5, 1.7580993408473766);  mul_5 = None
        mul_7: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1);  convert_element_type_7 = None
        exp_1: "f32[256, 512]" = torch.ops.aten.exp.default(mul_7);  mul_7 = None
        mul_8: "f32[256, 512]" = torch.ops.aten.mul.Tensor(mul_6, exp_1);  mul_6 = exp_1 = None
        mul_9: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.0507009873554805);  convert_element_type_6 = None
        where_1: "f32[256, 512]" = torch.ops.aten.where.self(le_1, mul_8, mul_9);  le_1 = mul_8 = mul_9 = None
        convert_element_type_8: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        mm_2: "bf16[256, 512]" = torch.ops.aten.mm.default(convert_element_type_8, arg5_1);  arg5_1 = None
        permute_1: "bf16[512, 256]" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0])
        mm_3: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_1, arg6_1);  permute_1 = arg6_1 = None
        sum_2: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_8, [0], True, dtype = torch.float32);  convert_element_type_8 = None
        view_1: "f32[512]" = torch.ops.aten.view.default(sum_2, _shape_param_1);  sum_2 = _shape_param_1 = None
        convert_element_type_9: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_10: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_11: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
        convert_element_type_12: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_13: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        le_2: "b8[256, 512]" = torch.ops.aten.le.Scalar(convert_element_type_13, 0)
        mul_10: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1)
        mul_11: "f32[256, 512]" = torch.ops.aten.mul.Tensor(mul_10, 1.7580993408473766);  mul_10 = None
        mul_12: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1);  convert_element_type_13 = None
        exp_2: "f32[256, 512]" = torch.ops.aten.exp.default(mul_12);  mul_12 = None
        mul_13: "f32[256, 512]" = torch.ops.aten.mul.Tensor(mul_11, exp_2);  mul_11 = exp_2 = None
        mul_14: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.0507009873554805);  convert_element_type_12 = None
        where_2: "f32[256, 512]" = torch.ops.aten.where.self(le_2, mul_13, mul_14);  le_2 = mul_13 = mul_14 = None
        convert_element_type_14: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where_2, torch.bfloat16);  where_2 = None
        mm_4: "bf16[256, 1024]" = torch.ops.aten.mm.default(convert_element_type_14, arg8_1);  arg8_1 = None
        permute_2: "bf16[512, 256]" = torch.ops.aten.permute.default(convert_element_type_14, [1, 0])
        mm_5: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_2, arg9_1);  permute_2 = arg9_1 = None
        sum_3: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_14, [0], True, dtype = torch.float32);  convert_element_type_14 = None
        view_2: "f32[512]" = torch.ops.aten.view.default(sum_3, _shape_param_2);  sum_3 = _shape_param_2 = None
        convert_element_type_15: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_16: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_17: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_15, torch.float32);  convert_element_type_15 = None
        convert_element_type_18: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        mul_15: "bf16[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 5.000000000000001);  convert_element_type_18 = None
        mul_16: "bf16[256, 1024]" = torch.ops.aten.mul.Tensor(mm_4, mul_15);  mm_4 = mul_15 = None
        convert_element_type_19: "f32[256, 1024]" = torch.ops.prims.convert_element_type.default(mul_16, torch.float32);  mul_16 = None
        convert_element_type_20: "f32[256, 1024]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.float32);  arg11_1 = None
        le_3: "b8[256, 1024]" = torch.ops.aten.le.Scalar(convert_element_type_20, 0)
        mul_17: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1)
        mul_18: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(mul_17, 1.7580993408473766);  mul_17 = None
        mul_19: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1);  convert_element_type_20 = None
        exp_3: "f32[256, 1024]" = torch.ops.aten.exp.default(mul_19);  mul_19 = None
        mul_20: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(mul_18, exp_3);  mul_18 = exp_3 = None
        mul_21: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.0507009873554805);  convert_element_type_19 = None
        where_3: "f32[256, 1024]" = torch.ops.aten.where.self(le_3, mul_20, mul_21);  le_3 = mul_20 = mul_21 = None
        convert_element_type_21: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(where_3, torch.bfloat16);  where_3 = None
        mm_6: "bf16[256, 512]" = torch.ops.aten.mm.default(convert_element_type_21, arg12_1);  arg12_1 = None
        permute_3: "bf16[1024, 256]" = torch.ops.aten.permute.default(convert_element_type_21, [1, 0])
        mm_7: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_3, arg13_1);  permute_3 = arg13_1 = None
        sum_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_21, [0], True, dtype = torch.float32);  convert_element_type_21 = None
        view_3: "f32[1024]" = torch.ops.aten.view.default(sum_4, _shape_param_3);  sum_4 = _shape_param_3 = None
        convert_element_type_22: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_23: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_24: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        convert_element_type_25: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_26: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.float32);  arg14_1 = None
        le_4: "b8[256, 512]" = torch.ops.aten.le.Scalar(convert_element_type_26, 0)
        mul_22: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1)
        mul_23: "f32[256, 512]" = torch.ops.aten.mul.Tensor(mul_22, 1.7580993408473766);  mul_22 = None
        mul_24: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1);  convert_element_type_26 = None
        exp_4: "f32[256, 512]" = torch.ops.aten.exp.default(mul_24);  mul_24 = None
        mul_25: "f32[256, 512]" = torch.ops.aten.mul.Tensor(mul_23, exp_4);  mul_23 = exp_4 = None
        mul_26: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.0507009873554805);  convert_element_type_25 = None
        where_4: "f32[256, 512]" = torch.ops.aten.where.self(le_4, mul_25, mul_26);  le_4 = mul_25 = mul_26 = None
        convert_element_type_27: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where_4, torch.bfloat16);  where_4 = None
        mm_8: "bf16[256, 512]" = torch.ops.aten.mm.default(convert_element_type_27, arg15_1);  arg15_1 = None
        permute_4: "bf16[512, 256]" = torch.ops.aten.permute.default(convert_element_type_27, [1, 0])
        mm_9: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_4, arg16_1);  permute_4 = arg16_1 = None
        sum_5: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_27, [0], True, dtype = torch.float32);  convert_element_type_27 = None
        view_4: "f32[512]" = torch.ops.aten.view.default(sum_5, _shape_param_4);  sum_5 = _shape_param_4 = None
        convert_element_type_28: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_29: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_30: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_28, torch.float32);  convert_element_type_28 = None
        convert_element_type_31: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        convert_element_type_32: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        le_5: "b8[256, 512]" = torch.ops.aten.le.Scalar(convert_element_type_32, 0)
        mul_27: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1)
        mul_28: "f32[256, 512]" = torch.ops.aten.mul.Tensor(mul_27, 1.7580993408473766);  mul_27 = None
        mul_29: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1);  convert_element_type_32 = None
        exp_5: "f32[256, 512]" = torch.ops.aten.exp.default(mul_29);  mul_29 = None
        mul_30: "f32[256, 512]" = torch.ops.aten.mul.Tensor(mul_28, exp_5);  mul_28 = exp_5 = None
        mul_31: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.0507009873554805);  convert_element_type_31 = None
        where_5: "f32[256, 512]" = torch.ops.aten.where.self(le_5, mul_30, mul_31);  le_5 = mul_30 = mul_31 = None
        convert_element_type_33: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where_5, torch.bfloat16);  where_5 = None
        permute_5: "bf16[512, 256]" = torch.ops.aten.permute.default(convert_element_type_33, [1, 0])
        constant_pad_nd_3: "bf16[256, 197952]" = torch.ops.aten.constant_pad_nd.default(arg18_1, [0, 1, 0, 0]);  arg18_1 = None
        mm_10: "bf16[512, 197952]" = torch.ops.aten.mm.default(permute_5, constant_pad_nd_3);  permute_5 = constant_pad_nd_3 = None
        slice_2: "bf16[512, 197951]" = torch.ops.aten.slice.Tensor(mm_10, 1, 0, -1);  mm_10 = None
        sum_6: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_33, [0], True, dtype = torch.float32);  convert_element_type_33 = None
        view_5: "f32[512]" = torch.ops.aten.view.default(sum_6, _shape_param_5);  sum_6 = _shape_param_5 = None
        convert_element_type_34: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        convert_element_type_35: "f32[512, 197951]" = torch.ops.prims.convert_element_type.default(slice_2, torch.float32);  slice_2 = None
        convert_element_type_36: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_34, torch.float32);  convert_element_type_34 = None
        return (convert_element_type_4, convert_element_type_5, convert_element_type_10, convert_element_type_11, convert_element_type_16, convert_element_type_17, convert_element_type_23, convert_element_type_24, convert_element_type_29, convert_element_type_30, convert_element_type_35, convert_element_type_36)



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
