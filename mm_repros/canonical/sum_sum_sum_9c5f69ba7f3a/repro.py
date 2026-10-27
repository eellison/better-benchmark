"""
Standalone repro captured via capture_hook.
Label: M2M100ForConditionalGeneration
Pattern hash: 9c5f69ba7f3a
Shape hash: 90e21286
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
    def forward(self, arg0_1: "f32[64, 128, 1024]", arg1_1: "b8[64, 128, 1024]", arg2_1: "bf16[1024, 4096]", arg3_1: "bf16[8192, 4096]", arg4_1: "b8[64, 128, 4096]", arg5_1: "bf16[4096, 1024]", arg6_1: "bf16[8192, 1024]", arg7_1: "f32[1024]", arg8_1: "bf16[8192, 1024]", arg9_1: "b8[64, 128, 1024]", arg10_1: "f32[64, 128, 1024]", arg11_1: "f32[64, 128, 1]", arg12_1: "f32[64, 128, 1]", arg13_1: "bf16[1024, 1024]", arg14_1: "bf16[8192, 1024]", arg15_1: "bf16[1024, 128, 128]", arg16_1: "bf16[1024, 64, 128]", arg17_1: "b8[64, 16, 128, 128]", arg18_1: "bf16[64, 16, 128, 128]", arg19_1: "bf16[1024, 64, 128]", arg20_1: "bf16[1024, 128, 64]", arg21_1: "bf16[1024, 1024]", arg22_1: "bf16[8192, 1024]", arg23_1: "bf16[1024, 1024]", arg24_1: "bf16[1024, 1024]", arg25_1: "f32[1024]", arg26_1: "f32[64, 128, 1]", arg27_1: "f32[64, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28):
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
        view_6: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(arg8_1, _shape_param_6);  arg8_1 = _shape_param_6 = None
        mul_4: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(arg9_1, view_6);  view_6 = None
        mul_5: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None
        add: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(arg10_1, mul_5);  mul_5 = None
        sub: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add, arg11_1);  add = arg11_1 = None
        mul_6: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, arg12_1);  sub = None
        mul_7: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_2, mul_6);  mul_2 = None
        sum_4: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_7, [2], True);  mul_7 = None
        mul_8: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_6, sum_4);  sum_4 = None
        sub_1: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_2: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_1, mul_8);  sub_1 = mul_8 = None
        div: "f32[64, 128, 1]" = torch.ops.aten.div.Tensor(arg12_1, 1024);  arg12_1 = None
        mul_9: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_10: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_6, mul_6);  mul_6 = None
        sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_10, [0, 1]);  mul_10 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [0, 1]);  convert_element_type_6 = None
        add_1: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(arg0_1, mul_9);  arg0_1 = mul_9 = None
        convert_element_type_9: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16)
        convert_element_type_10: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        mul_11: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_12: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_9, mul_11);  convert_element_type_9 = mul_11 = None
        view_7: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_12, _shape_param_7);  mul_12 = _shape_param_7 = None
        mm_4: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_7, arg13_1);  arg13_1 = None
        permute_2: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_7, [1, 0])
        mm_5: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_2, arg14_1);  permute_2 = arg14_1 = None
        sum_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True, dtype = torch.float32);  view_7 = None
        view_8: "f32[1024]" = torch.ops.aten.view.default(sum_7, _shape_param_8);  sum_7 = _shape_param_8 = None
        convert_element_type_11: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        view_9: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm_4, _shape_param_9);  mm_4 = _shape_param_9 = None
        convert_element_type_12: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_13: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        view_10: "bf16[64, 128, 16, 64]" = torch.ops.aten.view.default(view_9, _shape_param_10);  view_9 = _shape_param_10 = None
        permute_3: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone: "bf16[64, 16, 128, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_11: "bf16[1024, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_11);  clone = _shape_param_11 = None
        bmm: "bf16[1024, 128, 64]" = torch.ops.aten.bmm.default(arg15_1, view_11);  arg15_1 = None
        bmm_1: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_11, arg16_1);  view_11 = arg16_1 = None
        view_12: "bf16[64, 16, 128, 64]" = torch.ops.aten.view.default(bmm, _shape_param_12);  bmm = _shape_param_12 = None
        view_13: "bf16[64, 16, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_13);  bmm_1 = _shape_param_13 = None
        convert_element_type_14: "bf16[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        mul_13: "bf16[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_14: "bf16[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_13, mul_13);  view_13 = mul_13 = None
        convert_element_type_15: "f32[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_14, torch.float32);  mul_14 = None
        convert_element_type_16: "f32[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.float32);  arg18_1 = None
        mul_15: "f32[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_15, convert_element_type_16);  convert_element_type_15 = None
        sum_8: "f32[64, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [-1], True)
        neg: "f32[64, 16, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_16);  convert_element_type_16 = None
        fma: "f32[64, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_8, mul_15);  neg = sum_8 = mul_15 = None
        convert_element_type_17: "bf16[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_14: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_14);  convert_element_type_17 = _shape_param_14 = None
        bmm_2: "bf16[1024, 64, 128]" = torch.ops.aten.bmm.default(arg19_1, view_14);  arg19_1 = None
        bmm_3: "bf16[1024, 128, 64]" = torch.ops.aten.bmm.default(view_14, arg20_1);  view_14 = arg20_1 = None
        view_15: "bf16[64, 16, 64, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_15);  bmm_2 = _shape_param_15 = None
        view_16: "bf16[64, 16, 128, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_16);  bmm_3 = _shape_param_16 = None
        mul_16: "bf16[64, 16, 64, 128]" = torch.ops.aten.mul.Scalar(view_15, 0.3535533905932738);  view_15 = None
        permute_4: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(mul_16, [0, 1, 3, 2]);  mul_16 = None
        mul_17: "bf16[64, 16, 128, 64]" = torch.ops.aten.mul.Scalar(view_16, 0.3535533905932738);  view_16 = None
        permute_5: "bf16[64, 128, 16, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        clone_1: "bf16[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_17: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_17);  clone_1 = _shape_param_17 = None
        permute_6: "bf16[64, 128, 16, 64]" = torch.ops.aten.permute.default(permute_4, [0, 2, 1, 3]);  permute_4 = None
        view_18: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(permute_6, _shape_param_18);  permute_6 = _shape_param_18 = None
        view_19: "bf16[8192, 1024]" = torch.ops.aten.view.default(view_17, _shape_param_19);  view_17 = _shape_param_19 = None
        mm_6: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_19, arg21_1);  arg21_1 = None
        permute_7: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_19, [1, 0])
        mm_7: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_7, arg22_1);  permute_7 = None
        sum_9: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_19, [0], True, dtype = torch.float32);  view_19 = None
        view_20: "f32[1024]" = torch.ops.aten.view.default(sum_9, _shape_param_20);  sum_9 = _shape_param_20 = None
        convert_element_type_18: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_20, torch.bfloat16);  view_20 = None
        view_21: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm_6, _shape_param_21);  mm_6 = _shape_param_21 = None
        convert_element_type_19: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        convert_element_type_20: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_21: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_18, torch.float32);  convert_element_type_18 = None
        clone_2: "bf16[64, 128, 1024]" = torch.ops.aten.clone.default(view_18, memory_format = torch.contiguous_format);  view_18 = None
        view_22: "bf16[8192, 1024]" = torch.ops.aten.view.default(clone_2, _shape_param_22);  clone_2 = _shape_param_22 = None
        mm_8: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_22, arg23_1);  arg23_1 = None
        permute_8: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_22, [1, 0])
        mm_9: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_8, arg22_1);  permute_8 = None
        sum_10: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True, dtype = torch.float32);  view_22 = None
        view_23: "f32[1024]" = torch.ops.aten.view.default(sum_10, _shape_param_23);  sum_10 = _shape_param_23 = None
        convert_element_type_22: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_23, torch.bfloat16);  view_23 = None
        view_24: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm_8, _shape_param_24);  mm_8 = _shape_param_24 = None
        convert_element_type_23: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_24, torch.float32);  view_24 = None
        add_2: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_19, convert_element_type_23);  convert_element_type_19 = convert_element_type_23 = None
        convert_element_type_24: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_25: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        permute_9: "bf16[64, 128, 16, 64]" = torch.ops.aten.permute.default(mul_17, [0, 2, 1, 3]);  mul_17 = None
        clone_3: "bf16[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_25: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(clone_3, _shape_param_25);  clone_3 = _shape_param_25 = None
        view_26: "bf16[8192, 1024]" = torch.ops.aten.view.default(view_25, _shape_param_26);  view_25 = _shape_param_26 = None
        mm_10: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_26, arg24_1);  arg24_1 = None
        permute_10: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_26, [1, 0])
        mm_11: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_10, arg22_1);  permute_10 = arg22_1 = None
        sum_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_26, [0], True, dtype = torch.float32);  view_26 = None
        view_27: "f32[1024]" = torch.ops.aten.view.default(sum_11, _shape_param_27);  sum_11 = _shape_param_27 = None
        convert_element_type_26: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_27, torch.bfloat16);  view_27 = None
        view_28: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm_10, _shape_param_28);  mm_10 = _shape_param_28 = None
        convert_element_type_27: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        add_3: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_2, convert_element_type_27);  add_2 = convert_element_type_27 = None
        convert_element_type_28: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_29: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_26, torch.float32);  convert_element_type_26 = None
        mul_18: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_3, arg25_1);  arg25_1 = None
        mul_19: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_18, 1024)
        sum_12: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True)
        sub_3: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(arg10_1, arg26_1);  arg10_1 = arg26_1 = None
        mul_20: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_3, arg27_1);  sub_3 = None
        mul_21: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_18, mul_20);  mul_18 = None
        sum_13: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_21, [2], True);  mul_21 = None
        mul_22: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_20, sum_13);  sum_13 = None
        sub_4: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_19, sum_12);  mul_19 = sum_12 = None
        sub_5: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_4, mul_22);  sub_4 = mul_22 = None
        div_1: "f32[64, 128, 1]" = torch.ops.aten.div.Tensor(arg27_1, 1024);  arg27_1 = None
        mul_23: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_24: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_3, mul_20);  mul_20 = None
        sum_14: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_24, [0, 1]);  mul_24 = None
        sum_15: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_3, [0, 1]);  add_3 = None
        add_4: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_1, mul_23);  add_1 = mul_23 = None
        return (convert_element_type_3, convert_element_type_4, convert_element_type_7, convert_element_type_8, sum_5, sum_6, convert_element_type_12, convert_element_type_13, convert_element_type_20, convert_element_type_21, convert_element_type_24, convert_element_type_25, convert_element_type_28, convert_element_type_29, sum_14, sum_15, add_4)



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
