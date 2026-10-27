"""
Standalone repro captured via capture_hook.
Label: M2M100ForConditionalGeneration
Pattern hash: 809ffe97dcf1
Shape hash: ed04ebae
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
    def forward(self, arg0_1: "f32[64, 128, 1024]", arg1_1: "b8[64, 128, 1024]", arg2_1: "bf16[1024, 4096]", arg3_1: "bf16[8192, 4096]", arg4_1: "b8[64, 128, 4096]", arg5_1: "bf16[4096, 1024]", arg6_1: "bf16[8192, 1024]", arg7_1: "f32[1024]", arg8_1: "f32[64, 128, 1024]", arg9_1: "f32[64, 128, 1]", arg10_1: "b8[64, 128, 1024]", arg11_1: "bf16[1024, 1024]", arg12_1: "bf16[8192, 1024]", arg13_1: "bf16[1024, 128, 128]", arg14_1: "bf16[1024, 64, 128]", arg15_1: "b8[64, 16, 128, 128]", arg16_1: "bf16[64, 16, 128, 128]", arg17_1: "bf16[1024, 64, 128]", arg18_1: "bf16[1024, 128, 64]", arg19_1: "bf16[1024, 1024]", arg20_1: "bf16[8192, 1024]", arg21_1: "bf16[1024, 1024]", arg22_1: "bf16[1024, 1024]", arg23_1: "bf16[8192, 1024]", arg24_1: "f32[1024]", arg25_1: "bf16[8192, 1024]", arg26_1: "b8[64, 128, 1024]", arg27_1: "f32[64, 128, 1024]", arg28_1: "f32[64, 128, 1]", arg29_1: "f32[64, 128, 1]", arg30_1: "bf16[1024, 1024]", arg31_1: "bf16[64, 16, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16)
        convert_element_type_1: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        mm: "bf16[8192, 4096]" = torch.ops.aten.mm.default(view, arg2_1);  arg2_1 = None
        permute: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view, [1, 0])
        mm_1: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute, arg3_1);  permute = arg3_1 = None
        sum_1: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32);  view = None
        view_1: "f32[1024]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        convert_element_type_2: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        view_2: "bf16[64, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type_3: "f32[1024, 4096]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_4: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[64, 128, 4096]" = torch.ops.aten.where.self(arg4_1, full, view_2);  arg4_1 = full = view_2 = None
        view_3: "bf16[8192, 4096]" = torch.ops.aten.view.default(where, _shape_param_3);  where = _shape_param_3 = None
        mm_2: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_3, arg5_1);  arg5_1 = None
        permute_1: "bf16[4096, 8192]" = torch.ops.aten.permute.default(view_3, [1, 0])
        mm_3: "bf16[4096, 1024]" = torch.ops.aten.mm.default(permute_1, arg6_1);  permute_1 = arg6_1 = None
        sum_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32);  view_3 = None
        view_4: "f32[4096]" = torch.ops.aten.view.default(sum_2, _shape_param_4);  sum_2 = _shape_param_4 = None
        convert_element_type_5: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        view_5: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm_2, _shape_param_5);  mm_2 = _shape_param_5 = None
        convert_element_type_6: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        convert_element_type_7: "f32[4096, 1024]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_8: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        mul_2: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_6, arg7_1);  arg7_1 = None
        mul_3: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_2, 1024)
        sum_3: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        mul_4: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_2, arg8_1);  mul_2 = None
        sum_4: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        mul_5: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(arg8_1, sum_4);  sum_4 = None
        sub: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_1: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(sub, mul_5);  sub = mul_5 = None
        mul_6: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(arg9_1, sub_1);  arg9_1 = sub_1 = None
        mul_7: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_6, arg8_1);  arg8_1 = None
        sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [0, 1]);  convert_element_type_6 = None
        add: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(arg0_1, mul_6);  arg0_1 = mul_6 = None
        convert_element_type_9: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16)
        convert_element_type_10: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        mul_8: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_9: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_9, mul_8);  convert_element_type_9 = mul_8 = None
        view_6: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_9, _shape_param_6);  mul_9 = _shape_param_6 = None
        mm_4: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_6, arg11_1);  arg11_1 = None
        permute_2: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_6, [1, 0])
        mm_5: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_2, arg12_1);  permute_2 = arg12_1 = None
        sum_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_6, [0], True, dtype = torch.float32);  view_6 = None
        view_7: "f32[1024]" = torch.ops.aten.view.default(sum_7, _shape_param_7);  sum_7 = _shape_param_7 = None
        convert_element_type_11: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        view_8: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm_4, _shape_param_8);  mm_4 = _shape_param_8 = None
        convert_element_type_12: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_13: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        view_9: "bf16[64, 128, 16, 64]" = torch.ops.aten.view.default(view_8, _shape_param_9);  view_8 = _shape_param_9 = None
        permute_3: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        clone: "bf16[64, 16, 128, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_10: "bf16[1024, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_10);  clone = _shape_param_10 = None
        bmm: "bf16[1024, 128, 64]" = torch.ops.aten.bmm.default(arg13_1, view_10);  arg13_1 = None
        bmm_1: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_10, arg14_1);  view_10 = arg14_1 = None
        view_11: "bf16[64, 16, 128, 64]" = torch.ops.aten.view.default(bmm, _shape_param_11);  bmm = _shape_param_11 = None
        view_12: "bf16[64, 16, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_12);  bmm_1 = _shape_param_12 = None
        convert_element_type_14: "bf16[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.bfloat16);  arg15_1 = None
        mul_10: "bf16[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_11: "bf16[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_12, mul_10);  view_12 = mul_10 = None
        convert_element_type_15: "f32[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_11, torch.float32);  mul_11 = None
        convert_element_type_16: "f32[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.float32);  arg16_1 = None
        mul_12: "f32[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_15, convert_element_type_16);  convert_element_type_15 = None
        sum_8: "f32[64, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_12, [-1], True)
        neg: "f32[64, 16, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_16);  convert_element_type_16 = None
        fma: "f32[64, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_8, mul_12);  neg = sum_8 = mul_12 = None
        convert_element_type_17: "bf16[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_13: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_13);  convert_element_type_17 = _shape_param_13 = None
        bmm_2: "bf16[1024, 64, 128]" = torch.ops.aten.bmm.default(arg17_1, view_13);  arg17_1 = None
        bmm_3: "bf16[1024, 128, 64]" = torch.ops.aten.bmm.default(view_13, arg18_1);  view_13 = arg18_1 = None
        view_14: "bf16[64, 16, 64, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_14);  bmm_2 = _shape_param_14 = None
        view_15: "bf16[64, 16, 128, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_15);  bmm_3 = _shape_param_15 = None
        mul_13: "bf16[64, 16, 64, 128]" = torch.ops.aten.mul.Scalar(view_14, 0.3535533905932738);  view_14 = None
        permute_4: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(mul_13, [0, 1, 3, 2]);  mul_13 = None
        mul_14: "bf16[64, 16, 128, 64]" = torch.ops.aten.mul.Scalar(view_15, 0.3535533905932738);  view_15 = None
        permute_5: "bf16[64, 128, 16, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        clone_1: "bf16[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_16: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_16);  clone_1 = _shape_param_16 = None
        permute_6: "bf16[64, 128, 16, 64]" = torch.ops.aten.permute.default(permute_4, [0, 2, 1, 3]);  permute_4 = None
        view_17: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(permute_6, _shape_param_17);  permute_6 = _shape_param_17 = None
        view_18: "bf16[8192, 1024]" = torch.ops.aten.view.default(view_16, _shape_param_18);  view_16 = _shape_param_18 = None
        mm_6: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_18, arg19_1);  arg19_1 = None
        permute_7: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_7: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_7, arg20_1);  permute_7 = None
        sum_9: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True, dtype = torch.float32);  view_18 = None
        view_19: "f32[1024]" = torch.ops.aten.view.default(sum_9, _shape_param_19);  sum_9 = _shape_param_19 = None
        convert_element_type_18: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_19, torch.bfloat16);  view_19 = None
        view_20: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm_6, _shape_param_20);  mm_6 = _shape_param_20 = None
        convert_element_type_19: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_20, torch.float32);  view_20 = None
        convert_element_type_20: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_21: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_18, torch.float32);  convert_element_type_18 = None
        clone_2: "bf16[64, 128, 1024]" = torch.ops.aten.clone.default(view_17, memory_format = torch.contiguous_format);  view_17 = None
        view_21: "bf16[8192, 1024]" = torch.ops.aten.view.default(clone_2, _shape_param_21);  clone_2 = _shape_param_21 = None
        mm_8: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_21, arg21_1);  arg21_1 = None
        permute_8: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_21, [1, 0])
        mm_9: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_8, arg20_1);  permute_8 = arg20_1 = None
        sum_10: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_21, [0], True, dtype = torch.float32);  view_21 = None
        view_22: "f32[1024]" = torch.ops.aten.view.default(sum_10, _shape_param_22);  sum_10 = _shape_param_22 = None
        convert_element_type_22: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_22, torch.bfloat16);  view_22 = None
        view_23: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm_8, _shape_param_23);  mm_8 = _shape_param_23 = None
        convert_element_type_23: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_23, torch.float32);  view_23 = None
        add_1: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_19, convert_element_type_23);  convert_element_type_19 = convert_element_type_23 = None
        convert_element_type_24: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_25: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        permute_9: "bf16[64, 128, 16, 64]" = torch.ops.aten.permute.default(mul_14, [0, 2, 1, 3]);  mul_14 = None
        clone_3: "bf16[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_24: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(clone_3, _shape_param_24);  clone_3 = _shape_param_24 = None
        view_25: "bf16[8192, 1024]" = torch.ops.aten.view.default(view_24, _shape_param_25);  view_24 = _shape_param_25 = None
        mm_10: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_25, arg22_1);  arg22_1 = None
        permute_10: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_25, [1, 0])
        mm_11: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_10, arg23_1);  permute_10 = arg23_1 = None
        sum_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_25, [0], True, dtype = torch.float32);  view_25 = None
        view_26: "f32[1024]" = torch.ops.aten.view.default(sum_11, _shape_param_26);  sum_11 = _shape_param_26 = None
        convert_element_type_26: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_26, torch.bfloat16);  view_26 = None
        view_27: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm_10, _shape_param_27);  mm_10 = _shape_param_27 = None
        convert_element_type_27: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        convert_element_type_28: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_29: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_26, torch.float32);  convert_element_type_26 = None
        mul_15: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_27, arg24_1);  arg24_1 = None
        mul_16: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_15, 1024)
        sum_12: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True)
        view_28: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(arg25_1, _shape_param_28);  arg25_1 = _shape_param_28 = None
        mul_17: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(arg26_1, view_28);  view_28 = None
        mul_18: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_17, 1.1111111111111112);  mul_17 = None
        add_2: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(arg27_1, mul_18);  arg27_1 = mul_18 = None
        sub_2: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add_2, arg28_1);  add_2 = arg28_1 = None
        mul_19: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_2, arg29_1);  sub_2 = None
        mul_20: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_15, mul_19);  mul_15 = None
        sum_13: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True);  mul_20 = None
        mul_21: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_19, sum_13);  sum_13 = None
        sub_3: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_16, sum_12);  mul_16 = sum_12 = None
        sub_4: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_3, mul_21);  sub_3 = mul_21 = None
        div: "f32[64, 128, 1]" = torch.ops.aten.div.Tensor(arg29_1, 1024);  arg29_1 = None
        mul_22: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(div, sub_4);  div = sub_4 = None
        mul_23: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_27, mul_19);  mul_19 = None
        sum_14: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_23, [0, 1]);  mul_23 = None
        sum_15: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_27, [0, 1]);  convert_element_type_27 = None
        add_3: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add, mul_22);  add = mul_22 = None
        convert_element_type_30: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16)
        convert_element_type_31: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.bfloat16);  arg26_1 = None
        mul_24: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_25: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_30, mul_24);  convert_element_type_30 = mul_24 = None
        view_29: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_25, _shape_param_29);  mul_25 = _shape_param_29 = None
        mm_12: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_29, arg30_1);  arg30_1 = None
        permute_11: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_29, [1, 0])
        permute_12: "bf16[64, 128, 16, 64]" = torch.ops.aten.permute.default(arg31_1, [0, 2, 1, 3]);  arg31_1 = None
        view_30: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(permute_12, _shape_param_30);  permute_12 = _shape_param_30 = None
        view_31: "bf16[8192, 1024]" = torch.ops.aten.view.default(view_30, _shape_param_31);  view_30 = _shape_param_31 = None
        mm_13: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_11, view_31);  permute_11 = view_31 = None
        sum_16: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_29, [0], True, dtype = torch.float32);  view_29 = None
        view_32: "f32[1024]" = torch.ops.aten.view.default(sum_16, _shape_param_32);  sum_16 = _shape_param_32 = None
        convert_element_type_32: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_32, torch.bfloat16);  view_32 = None
        view_33: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm_12, _shape_param_33);  mm_12 = _shape_param_33 = None
        convert_element_type_33: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_34: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_32, torch.float32);  convert_element_type_32 = None
        view_34: "bf16[64, 128, 16, 64]" = torch.ops.aten.view.default(view_33, _shape_param_34);  view_33 = _shape_param_34 = None
        permute_13: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        return (convert_element_type_3, convert_element_type_4, convert_element_type_7, convert_element_type_8, sum_5, sum_6, convert_element_type_12, convert_element_type_13, convert_element_type_20, convert_element_type_21, add_1, convert_element_type_24, convert_element_type_25, convert_element_type_28, convert_element_type_29, sum_14, sum_15, add_3, convert_element_type_33, convert_element_type_34, permute_13)



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
