"""
Standalone repro captured via capture_hook.
Label: MegatronBertForCausalLM
Pattern hash: c43bbe19d108
Shape hash: 66910ba5
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
    def forward(self, arg0_1: "bf16[16, 16, 512, 64]", arg1_1: "bf16[16, 16, 512, 64]", arg2_1: "bf16[16, 16, 512, 64]", arg3_1: "bf16[1024, 1024]", arg4_1: "bf16[8192, 1024]", arg5_1: "bf16[1024, 1024]", arg6_1: "bf16[1024, 1024]", arg7_1: "f32[1024]", arg8_1: "f32[16, 512, 1024]", arg9_1: "f32[16, 512, 1]", arg10_1: "f32[16, 512, 1024]", arg11_1: "b8[16, 512, 1024]", arg12_1: "bf16[1024, 4096]", arg13_1: "bf16[8192, 4096]", arg14_1: "bf16[8192, 4096]", arg15_1: "bf16[4096, 1024]", arg16_1: "bf16[8192, 1024]", arg17_1: "f32[1024]", arg18_1: "f32[16, 512, 1024]", arg19_1: "f32[16, 512, 1]", arg20_1: "b8[16, 512, 1024]", arg21_1: "bf16[1024, 1024]", arg22_1: "bf16[8192, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22):
        # No stacktrace found for following nodes
        permute: "bf16[16, 512, 16, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        permute_1: "bf16[16, 512, 16, 64]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
        permute_2: "bf16[16, 512, 16, 64]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        clone: "bf16[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        view_1: "bf16[8192, 1024]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_1, arg3_1);  arg3_1 = None
        permute_3: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_1, [1, 0])
        mm_1: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_3, arg4_1);  permute_3 = None
        sum_1: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32);  view_1 = None
        view_2: "f32[1024]" = torch.ops.aten.view.default(sum_1, _shape_param_2);  sum_1 = _shape_param_2 = None
        convert_element_type: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        view_3: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        convert_element_type_1: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        convert_element_type_2: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_3: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        view_4: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(permute_1, _shape_param_4);  permute_1 = _shape_param_4 = None
        clone_1: "bf16[16, 512, 1024]" = torch.ops.aten.clone.default(view_4, memory_format = torch.contiguous_format);  view_4 = None
        view_5: "bf16[8192, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        mm_2: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_5, arg5_1);  arg5_1 = None
        permute_4: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_5, [1, 0])
        mm_3: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_4, arg4_1);  permute_4 = None
        sum_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32);  view_5 = None
        view_6: "f32[1024]" = torch.ops.aten.view.default(sum_2, _shape_param_6);  sum_2 = _shape_param_6 = None
        convert_element_type_4: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        view_7: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm_2, _shape_param_7);  mm_2 = _shape_param_7 = None
        convert_element_type_5: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        add: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_5);  convert_element_type_1 = convert_element_type_5 = None
        convert_element_type_6: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_7: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        clone_2: "bf16[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_8: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(clone_2, _shape_param_8);  clone_2 = _shape_param_8 = None
        view_9: "bf16[8192, 1024]" = torch.ops.aten.view.default(view_8, _shape_param_9);  view_8 = _shape_param_9 = None
        mm_4: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_9, arg6_1);  arg6_1 = None
        permute_5: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_9, [1, 0])
        mm_5: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_5, arg4_1);  permute_5 = arg4_1 = None
        sum_3: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_9, [0], True, dtype = torch.float32);  view_9 = None
        view_10: "f32[1024]" = torch.ops.aten.view.default(sum_3, _shape_param_10);  sum_3 = _shape_param_10 = None
        convert_element_type_8: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        view_11: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm_4, _shape_param_11);  mm_4 = _shape_param_11 = None
        convert_element_type_9: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        add_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add, convert_element_type_9);  add = convert_element_type_9 = None
        convert_element_type_10: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_11: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        mul: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_1, arg7_1);  arg7_1 = None
        mul_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        sum_4: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, arg8_1);  mul = None
        sum_5: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg8_1, sum_5);  sum_5 = None
        sub: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1, sum_4);  mul_1 = sum_4 = None
        sub_1: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg9_1, sub_1);  arg9_1 = sub_1 = None
        mul_5: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_1, arg8_1);  arg8_1 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_7: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_1, [0, 1]);  add_1 = None
        add_2: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(arg10_1, mul_4);  arg10_1 = mul_4 = None
        convert_element_type_12: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16)
        convert_element_type_13: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        mul_6: "bf16[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_7: "bf16[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, mul_6);  convert_element_type_12 = mul_6 = None
        view_12: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_7, _shape_param_12);  mul_7 = _shape_param_12 = None
        mm_6: "bf16[8192, 4096]" = torch.ops.aten.mm.default(view_12, arg12_1);  arg12_1 = None
        permute_6: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_12, [1, 0])
        mm_7: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_6, arg13_1);  permute_6 = arg13_1 = None
        sum_8: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_12, [0], True, dtype = torch.float32);  view_12 = None
        view_13: "f32[1024]" = torch.ops.aten.view.default(sum_8, _shape_param_13);  sum_8 = _shape_param_13 = None
        convert_element_type_14: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_13, torch.bfloat16);  view_13 = None
        view_14: "bf16[16, 512, 4096]" = torch.ops.aten.view.default(mm_6, _shape_param_14);  mm_6 = _shape_param_14 = None
        convert_element_type_15: "f32[1024, 4096]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_16: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        convert_element_type_17: "f32[16, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        view_15: "bf16[16, 512, 4096]" = torch.ops.aten.view.default(arg14_1, _shape_param_15);  arg14_1 = _shape_param_15 = None
        convert_element_type_18: "f32[16, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_15, torch.float32);  view_15 = None
        mul_8: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.7071067811865476)
        erf: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_3: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
        mul_10: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_18, convert_element_type_18)
        mul_11: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_10, -0.5);  mul_10 = None
        exp: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_11);  mul_11 = None
        mul_12: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_13: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_18, mul_12);  convert_element_type_18 = mul_12 = None
        add_4: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_9, mul_13);  mul_9 = mul_13 = None
        mul_14: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_17, add_4);  convert_element_type_17 = add_4 = None
        convert_element_type_19: "bf16[16, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        view_16: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_16);  convert_element_type_19 = _shape_param_16 = None
        mm_8: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_16, arg15_1);  arg15_1 = None
        permute_7: "bf16[4096, 8192]" = torch.ops.aten.permute.default(view_16, [1, 0])
        mm_9: "bf16[4096, 1024]" = torch.ops.aten.mm.default(permute_7, arg16_1);  permute_7 = arg16_1 = None
        sum_9: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_16, [0], True, dtype = torch.float32);  view_16 = None
        view_17: "f32[4096]" = torch.ops.aten.view.default(sum_9, _shape_param_17);  sum_9 = _shape_param_17 = None
        convert_element_type_20: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_17, torch.bfloat16);  view_17 = None
        view_18: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm_8, _shape_param_18);  mm_8 = _shape_param_18 = None
        convert_element_type_21: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_18, torch.float32);  view_18 = None
        convert_element_type_22: "f32[4096, 1024]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_23: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32);  convert_element_type_20 = None
        mul_15: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_21, arg17_1);  arg17_1 = None
        mul_16: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_15, 1024)
        sum_10: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True)
        mul_17: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_15, arg18_1);  mul_15 = None
        sum_11: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_17, [2], True);  mul_17 = None
        mul_18: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg18_1, sum_11);  sum_11 = None
        sub_2: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_16, sum_10);  mul_16 = sum_10 = None
        sub_3: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_2, mul_18);  sub_2 = mul_18 = None
        mul_19: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg19_1, sub_3);  arg19_1 = sub_3 = None
        mul_20: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_21, arg18_1);  arg18_1 = None
        sum_12: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_20, [0, 1]);  mul_20 = None
        sum_13: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_21, [0, 1]);  convert_element_type_21 = None
        add_5: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_2, mul_19);  add_2 = mul_19 = None
        convert_element_type_24: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16)
        convert_element_type_25: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        mul_21: "bf16[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_22: "bf16[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, mul_21);  convert_element_type_24 = mul_21 = None
        view_19: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_22, _shape_param_19);  mul_22 = _shape_param_19 = None
        mm_10: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_19, arg21_1);  arg21_1 = None
        permute_8: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_19, [1, 0])
        mm_11: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_8, arg22_1);  permute_8 = arg22_1 = None
        sum_14: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_19, [0], True, dtype = torch.float32);  view_19 = None
        view_20: "f32[1024]" = torch.ops.aten.view.default(sum_14, _shape_param_20);  sum_14 = _shape_param_20 = None
        convert_element_type_26: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_20, torch.bfloat16);  view_20 = None
        view_21: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm_10, _shape_param_21);  mm_10 = _shape_param_21 = None
        convert_element_type_27: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_28: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_26, torch.float32);  convert_element_type_26 = None
        view_22: "bf16[16, 512, 16, 64]" = torch.ops.aten.view.default(view_21, _shape_param_22);  view_21 = _shape_param_22 = None
        permute_9: "bf16[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        clone_3: "bf16[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        return (convert_element_type_2, convert_element_type_3, convert_element_type_6, convert_element_type_7, convert_element_type_10, convert_element_type_11, sum_6, sum_7, convert_element_type_15, convert_element_type_16, convert_element_type_22, convert_element_type_23, sum_12, sum_13, add_5, convert_element_type_27, convert_element_type_28, clone_3)



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
