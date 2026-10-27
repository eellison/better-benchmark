"""
Standalone repro captured via capture_hook.
Label: GPT2ForSequenceClassification
Pattern hash: 627107c794e0
Shape hash: 325d112d
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "b8[8, 2]", arg3_1: "b8[8, 1]", arg4_1: "bf16[8, 2]", arg5_1: "bf16[8, 2]", arg6_1: "i64[8]", arg7_1: "i64[8]", arg8_1: "bf16[8192, 768]", arg9_1: "bf16[2, 768]", arg10_1: "f32[768]", arg11_1: "f32[8, 1024, 768]", arg12_1: "f32[8, 1024, 1]", arg13_1: "b8[8, 1024, 768]", arg14_1: "bf16[768, 3072]", arg15_1: "bf16[3072, 8192]", arg16_1: "bf16[8192, 3072]", arg17_1: "bf16[3072, 768]", arg18_1: "bf16[768, 8192]", arg19_1: "f32[768]", arg20_1: "f32[8, 1024, 768]", arg21_1: "f32[8, 1024, 1]", arg22_1: "b8[8, 1024, 768]", arg23_1: "bf16[768, 768]", arg24_1: "bf16[8, 12, 1024, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where: "f32[8, 2]" = torch.ops.aten.where.self(arg2_1, scalar_tensor_1, scalar_tensor);  arg2_1 = scalar_tensor_1 = scalar_tensor = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8, 1]" = torch.ops.aten.where.self(arg3_1, div, full);  arg3_1 = div = full = None
        mul: "f32[8, 2]" = torch.ops.aten.mul.Tensor(where, where_1);  where = where_1 = None
        convert_element_type: "bf16[8, 2]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[8, 2]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        convert_element_type_2: "f32[8, 2]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        amax: "f32[8, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [1], True)
        sub: "f32[8, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[8, 2]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[8, 2]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_3: "bf16[8, 2]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[8, 2]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp_1: "f32[8, 2]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_2: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[8, 2]" = torch.ops.aten.mul.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        sub_2: "f32[8, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[8, 2]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        add: "bf16[8, 2]" = torch.ops.aten.add.Tensor(arg5_1, convert_element_type_5);  arg5_1 = convert_element_type_5 = None
        full_1: "bf16[8, 1024, 2]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        index_put: "bf16[8, 1024, 2]" = torch.ops.aten.index_put.default(full_1, [arg6_1, arg7_1], add, True);  full_1 = arg6_1 = arg7_1 = add = None
        view: "bf16[8192, 2]" = torch.ops.aten.view.default(index_put, _shape_param_1);  index_put = _shape_param_1 = None
        permute: "bf16[2, 8192]" = torch.ops.aten.permute.default(view, [1, 0])
        mm: "bf16[2, 768]" = torch.ops.aten.mm.default(permute, arg8_1);  permute = arg8_1 = None
        mm_1: "bf16[8192, 768]" = torch.ops.aten.mm.default(view, arg9_1);  view = arg9_1 = None
        view_1: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_2);  mm_1 = _shape_param_2 = None
        convert_element_type_6: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        convert_element_type_7: "f32[2, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, arg10_1);  arg10_1 = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        sum_3: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg11_1);  mul_2 = None
        sum_4: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        mul_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg11_1, sum_4);  sum_4 = None
        sub_3: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_4: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_3, mul_5);  sub_3 = mul_5 = None
        mul_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg12_1, sub_4);  arg12_1 = sub_4 = None
        mul_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, arg11_1);  arg11_1 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None
        sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [0, 1]);  convert_element_type_6 = None
        convert_element_type_8: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16)
        convert_element_type_9: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        mul_8: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_9: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, mul_8);  convert_element_type_8 = mul_8 = None
        view_2: "bf16[8192, 768]" = torch.ops.aten.view.default(mul_9, _shape_param_3);  mul_9 = _shape_param_3 = None
        mm_2: "bf16[8192, 3072]" = torch.ops.aten.mm.default(view_2, arg14_1);  arg14_1 = None
        mm_3: "bf16[3072, 768]" = torch.ops.aten.mm.default(arg15_1, view_2);  arg15_1 = None
        sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32);  view_2 = None
        view_3: "f32[768]" = torch.ops.aten.view.default(sum_7, _shape_param_4);  sum_7 = _shape_param_4 = None
        convert_element_type_10: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_11: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_10, torch.float32);  convert_element_type_10 = None
        convert_element_type_12: "f32[8192, 3072]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_13: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        view_4: "f32[8, 1024, 3072]" = torch.ops.aten.view.default(convert_element_type_12, _shape_param_5);  convert_element_type_12 = _shape_param_5 = None
        view_5: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(arg16_1, _shape_param_6);  arg16_1 = _shape_param_6 = None
        mul_10: "bf16[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_5, 0.5)
        mul_11: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_4, mul_10);  mul_10 = None
        convert_element_type_14: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32)
        pow_1: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_14, 3.0)
        mul_12: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_1: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_5, mul_12);  view_5 = mul_12 = None
        mul_13: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_1, 0.7978845608028654);  add_1 = None
        tanh: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_13);  mul_13 = None
        add_2: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_14: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_4, add_2);  view_4 = add_2 = None
        convert_element_type_15: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        mul_15: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_5: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_15);  mul_15 = None
        mul_16: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_11, sub_5);  mul_11 = sub_5 = None
        mul_17: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_16, 0.7978845608028654);  mul_16 = None
        convert_element_type_16: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16)
        mul_18: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_17, 0.044715);  mul_17 = None
        pow_2: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_14, 2.0);  convert_element_type_14 = None
        mul_19: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_20: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        convert_element_type_17: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        add_3: "bf16[8, 1024, 3072]" = torch.ops.aten.add.Tensor(convert_element_type_16, convert_element_type_17);  convert_element_type_16 = convert_element_type_17 = None
        mul_21: "bf16[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 0.5);  convert_element_type_15 = None
        add_4: "bf16[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_3, mul_21);  add_3 = mul_21 = None
        view_6: "bf16[8192, 3072]" = torch.ops.aten.view.default(add_4, _shape_param_7);  add_4 = _shape_param_7 = None
        mm_4: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_6, arg17_1);  arg17_1 = None
        mm_5: "bf16[768, 3072]" = torch.ops.aten.mm.default(arg18_1, view_6);  arg18_1 = None
        sum_8: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_6, [0], True, dtype = torch.float32);  view_6 = None
        view_7: "f32[3072]" = torch.ops.aten.view.default(sum_8, _shape_param_8);  sum_8 = _shape_param_8 = None
        convert_element_type_18: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        convert_element_type_19: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_18, torch.float32);  convert_element_type_18 = None
        convert_element_type_20: "f32[8192, 768]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_21: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        view_8: "f32[8, 1024, 768]" = torch.ops.aten.view.default(convert_element_type_20, _shape_param_9);  convert_element_type_20 = _shape_param_9 = None
        mul_22: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_8, arg19_1);  arg19_1 = None
        mul_23: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_22, 768)
        sum_9: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [2], True)
        mul_24: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg20_1);  mul_22 = None
        sum_10: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_24, [2], True);  mul_24 = None
        mul_25: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg20_1, sum_10);  sum_10 = None
        sub_6: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_23, sum_9);  mul_23 = sum_9 = None
        sub_7: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_6, mul_25);  sub_6 = mul_25 = None
        mul_26: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg21_1, sub_7);  arg21_1 = sub_7 = None
        mul_27: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_8, arg20_1);  arg20_1 = None
        sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_27, [0, 1]);  mul_27 = None
        sum_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_8, [0, 1]);  view_8 = None
        add_5: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_6, mul_26);  mul_6 = mul_26 = None
        convert_element_type_22: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16)
        convert_element_type_23: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        mul_28: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_29: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, mul_28);  convert_element_type_22 = mul_28 = None
        view_9: "bf16[8192, 768]" = torch.ops.aten.view.default(mul_29, _shape_param_10);  mul_29 = _shape_param_10 = None
        mm_6: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_9, arg23_1);  arg23_1 = None
        permute_1: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(arg24_1, [0, 2, 1, 3]);  arg24_1 = None
        view_10: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(permute_1, _shape_param_11);  permute_1 = _shape_param_11 = None
        view_11: "bf16[8192, 768]" = torch.ops.aten.view.default(view_10, _shape_param_12);  view_10 = _shape_param_12 = None
        permute_2: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_11, [1, 0]);  view_11 = None
        mm_7: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_2, view_9);  permute_2 = None
        sum_13: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_9, [0], True, dtype = torch.float32);  view_9 = None
        view_12: "f32[768]" = torch.ops.aten.view.default(sum_13, _shape_param_13);  sum_13 = _shape_param_13 = None
        convert_element_type_24: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        convert_element_type_25: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_24, torch.float32);  convert_element_type_24 = None
        convert_element_type_26: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        view_13: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_6, _shape_param_14);  mm_6 = _shape_param_14 = None
        view_14: "bf16[8, 1024, 12, 64]" = torch.ops.aten.view.default(view_13, _shape_param_15);  view_13 = _shape_param_15 = None
        permute_3: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        return (convert_element_type_7, sum_5, sum_6, convert_element_type_11, convert_element_type_13, convert_element_type_19, convert_element_type_21, sum_11, sum_12, add_5, convert_element_type_25, convert_element_type_26, permute_3)



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
