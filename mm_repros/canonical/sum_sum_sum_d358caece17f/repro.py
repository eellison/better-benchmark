"""
Standalone repro captured via capture_hook.
Label: YituTechConvBert
Pattern hash: d358caece17f
Shape hash: 954381a3
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
    def forward(self, arg0_1: "bf16[32, 6, 512, 64]", arg1_1: "bf16[32, 6, 512, 64]", arg2_1: "bf16[32, 6, 512, 64]", arg3_1: "f32[98304, 9, 1]", arg4_1: "bf16[98304, 64, 1]", arg5_1: "i64[9, 512, 1, 1]", arg6_1: "i64[1, 1]", arg7_1: "bf16[384, 768]", arg8_1: "bf16[16384, 768]", arg9_1: "f32[32, 512, 768]", arg10_1: "bf16[32, 384, 512]", arg11_1: "f32[384, 1]", arg12_1: "bf16[32, 512, 384]", arg13_1: "bf16[384, 768]", arg14_1: "bf16[32, 768, 512]", arg15_1: "bf16[384, 768]", arg16_1: "bf16[384, 768]", arg17_1: "f32[768]", arg18_1: "f32[32, 512, 768]", arg19_1: "f32[32, 512, 1]", arg20_1: "b8[32, 512, 768]", arg21_1: "bf16[768, 3072]", arg22_1: "bf16[16384, 3072]", arg23_1: "bf16[16384, 3072]", arg24_1: "bf16[3072, 768]", arg25_1: "bf16[16384, 768]", arg26_1: "f32[768]", arg27_1: "f32[32, 512, 768]", arg28_1: "f32[32, 512, 1]", arg29_1: "b8[32, 512, 768]", arg30_1: "bf16[768, 768]", arg31_1: "bf16[16384, 768]", arg32_1: "bf16[16384, 384]", arg33_1: "bf16[98304, 9, 64]", arg34_1: "f32[54]", arg35_1: "bf16[16384, 54]", arg36_1: "f32[98304, 1, 1]", arg37_1: "f32[98304, 1, 1]", arg38_1: "bf16[16384, 384]", arg39_1: "bf16[54, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42):
        # No stacktrace found for following nodes
        permute: "bf16[32, 512, 6, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        permute_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
        permute_2: "bf16[32, 512, 6, 64]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        convert_element_type: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        expand: "bf16[98304, 9, 1]" = torch.ops.aten.expand.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        permute_3: "bf16[98304, 1, 9]" = torch.ops.aten.permute.default(expand, [0, 2, 1]);  expand = None
        convert_element_type_1: "f32[98304, 64, 1]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        convert_element_type_2: "f32[98304, 1, 9]" = torch.ops.prims.convert_element_type.default(permute_3, torch.float32);  permute_3 = None
        mul: "f32[98304, 64, 9]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_2);  convert_element_type_1 = convert_element_type_2 = None
        convert_element_type_3: "bf16[98304, 64, 9]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        view: "bf16[32, 512, 384, 9]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_1);  convert_element_type_3 = _shape_param_1 = None
        view_1: "bf16[32, 512, 3456]" = torch.ops.aten.view.default(view, _shape_param_2);  view = _shape_param_2 = None
        permute_4: "bf16[32, 3456, 512]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        convert_element_type_4: "f32[32, 3456, 512]" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None
        view_2: "f32[32, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_3);  convert_element_type_4 = _shape_param_3 = None
        permute_5: "f32[32, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_2, [0, 1, 2, 4, 3, 5]);  view_2 = None
        full: "f32[32, 384, 520, 1]" = torch.ops.aten.full.default(_shape_param_4, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        index_put: "f32[32, 384, 520, 1]" = torch.ops.aten.index_put.default(full, [None, None, arg5_1, arg6_1], permute_5, True);  arg5_1 = arg6_1 = permute_5 = None
        constant_pad_nd: "f32[32, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(index_put, _shape_param_5, 0.0);  index_put = _shape_param_5 = None
        convert_element_type_5: "bf16[32, 384, 512, 1]" = torch.ops.prims.convert_element_type.default(constant_pad_nd, torch.bfloat16);  constant_pad_nd = None
        squeeze: "bf16[32, 384, 512]" = torch.ops.aten.squeeze.dim(convert_element_type_5, -1);  convert_element_type_5 = None
        permute_6: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(squeeze, [0, 2, 1]);  squeeze = None
        clone: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_3: "bf16[16384, 384]" = torch.ops.aten.view.default(clone, _shape_param_6);  clone = _shape_param_6 = None
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_3, arg7_1);  arg7_1 = None
        permute_7: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_3, [1, 0])
        mm_1: "bf16[384, 768]" = torch.ops.aten.mm.default(permute_7, arg8_1);  permute_7 = None
        sum_1: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32);  view_3 = None
        view_4: "f32[384]" = torch.ops.aten.view.default(sum_1, _shape_param_7);  sum_1 = _shape_param_7 = None
        convert_element_type_6: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        view_5: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_8);  mm = _shape_param_8 = None
        convert_element_type_7: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg9_1, convert_element_type_7);  arg9_1 = convert_element_type_7 = None
        convert_element_type_8: "f32[384, 768]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_9: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        add_1: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(arg10_1, arg11_1);  arg10_1 = arg11_1 = None
        convert_element_type_10: "bf16[32, 384, 512]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        permute_8: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(convert_element_type_10, [0, 2, 1]);  convert_element_type_10 = None
        mul_1: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(arg12_1, permute_8);  arg12_1 = permute_8 = None
        clone_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_6: "bf16[32, 512, 384]" = torch.ops.aten.view.default(clone_1, _shape_param_9);  clone_1 = _shape_param_9 = None
        view_7: "bf16[32, 512, 384]" = torch.ops.aten.view.default(permute_1, _shape_param_10);  permute_1 = _shape_param_10 = None
        clone_2: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_8: "bf16[32, 512, 384]" = torch.ops.aten.view.default(clone_2, _shape_param_11);  clone_2 = _shape_param_11 = None
        add_2: "bf16[32, 512, 384]" = torch.ops.aten.add.Tensor(mul_1, view_8);  mul_1 = view_8 = None
        view_9: "bf16[16384, 384]" = torch.ops.aten.view.default(add_2, _shape_param_12);  add_2 = _shape_param_12 = None
        mm_2: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_9, arg13_1);  arg13_1 = None
        permute_9: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_9, [1, 0])
        mm_3: "bf16[384, 768]" = torch.ops.aten.mm.default(permute_9, arg8_1);  permute_9 = None
        sum_2: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_9, [0], True, dtype = torch.float32);  view_9 = None
        view_10: "f32[384]" = torch.ops.aten.view.default(sum_2, _shape_param_13);  sum_2 = _shape_param_13 = None
        convert_element_type_11: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        view_11: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_14);  mm_2 = _shape_param_14 = None
        convert_element_type_12: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, convert_element_type_12);  add = convert_element_type_12 = None
        convert_element_type_13: "f32[384, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_14: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        convert_element_type_15: "f32[32, 768, 512]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.float32);  arg14_1 = None
        permute_10: "f32[32, 512, 768]" = torch.ops.aten.permute.default(convert_element_type_15, [0, 2, 1]);  convert_element_type_15 = None
        add_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_3, permute_10);  add_3 = permute_10 = None
        view_12: "bf16[16384, 384]" = torch.ops.aten.view.default(view_6, _shape_param_15);  view_6 = _shape_param_15 = None
        mm_4: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_12, arg15_1);  arg15_1 = None
        permute_11: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_12, [1, 0])
        mm_5: "bf16[384, 768]" = torch.ops.aten.mm.default(permute_11, arg8_1);  permute_11 = None
        sum_3: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_12, [0], True, dtype = torch.float32);  view_12 = None
        view_13: "f32[384]" = torch.ops.aten.view.default(sum_3, _shape_param_16);  sum_3 = _shape_param_16 = None
        convert_element_type_16: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_13, torch.bfloat16);  view_13 = None
        view_14: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_4, _shape_param_17);  mm_4 = _shape_param_17 = None
        convert_element_type_17: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_4, convert_element_type_17);  add_4 = convert_element_type_17 = None
        convert_element_type_18: "f32[384, 768]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_19: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_16, torch.float32);  convert_element_type_16 = None
        clone_3: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(view_7, memory_format = torch.contiguous_format);  view_7 = None
        view_15: "bf16[16384, 384]" = torch.ops.aten.view.default(clone_3, _shape_param_18);  clone_3 = _shape_param_18 = None
        mm_6: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_15, arg16_1);  arg16_1 = None
        permute_12: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_15, [1, 0])
        mm_7: "bf16[384, 768]" = torch.ops.aten.mm.default(permute_12, arg8_1);  permute_12 = arg8_1 = None
        sum_4: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_15, [0], True, dtype = torch.float32);  view_15 = None
        view_16: "f32[384]" = torch.ops.aten.view.default(sum_4, _shape_param_19);  sum_4 = _shape_param_19 = None
        convert_element_type_20: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_16, torch.bfloat16);  view_16 = None
        view_17: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_6, _shape_param_20);  mm_6 = _shape_param_20 = None
        convert_element_type_21: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_21);  add_5 = convert_element_type_21 = None
        convert_element_type_22: "f32[384, 768]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_23: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32);  convert_element_type_20 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_6, arg17_1);  arg17_1 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        sum_5: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg18_1);  mul_2 = None
        sum_6: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg18_1, sum_6);  sum_6 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_3, sum_5);  mul_3 = sum_5 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub, mul_5);  sub = mul_5 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg19_1, sub_1);  arg19_1 = sub_1 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_6, arg18_1);  arg18_1 = None
        sum_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None
        sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_6, [0, 1]);  add_6 = None
        convert_element_type_24: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16)
        convert_element_type_25: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        mul_8: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_9: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_24, mul_8);  convert_element_type_24 = mul_8 = None
        view_18: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_9, _shape_param_21);  mul_9 = _shape_param_21 = None
        mm_8: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_18, arg21_1);  arg21_1 = None
        permute_13: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_9: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_13, arg22_1);  permute_13 = arg22_1 = None
        sum_9: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True, dtype = torch.float32);  view_18 = None
        view_19: "f32[768]" = torch.ops.aten.view.default(sum_9, _shape_param_22);  sum_9 = _shape_param_22 = None
        convert_element_type_26: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_19, torch.bfloat16);  view_19 = None
        view_20: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_8, _shape_param_23);  mm_8 = _shape_param_23 = None
        convert_element_type_27: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_28: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_26, torch.float32);  convert_element_type_26 = None
        convert_element_type_29: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_20, torch.float32);  view_20 = None
        view_21: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg23_1, _shape_param_24);  arg23_1 = _shape_param_24 = None
        convert_element_type_30: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_10: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.7071067811865476)
        erf: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
        add_7: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_11: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
        mul_12: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_30, convert_element_type_30)
        mul_13: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, -0.5);  mul_12 = None
        exp: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_13);  mul_13 = None
        mul_14: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_15: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_30, mul_14);  convert_element_type_30 = mul_14 = None
        add_8: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_11, mul_15);  mul_11 = mul_15 = None
        mul_16: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_29, add_8);  convert_element_type_29 = add_8 = None
        convert_element_type_31: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        view_22: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_25);  convert_element_type_31 = _shape_param_25 = None
        mm_10: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_22, arg24_1);  arg24_1 = None
        permute_14: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_22, [1, 0])
        mm_11: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_14, arg25_1);  permute_14 = arg25_1 = None
        sum_10: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True, dtype = torch.float32);  view_22 = None
        view_23: "f32[3072]" = torch.ops.aten.view.default(sum_10, _shape_param_26);  sum_10 = _shape_param_26 = None
        convert_element_type_32: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_23, torch.bfloat16);  view_23 = None
        view_24: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_10, _shape_param_27);  mm_10 = _shape_param_27 = None
        convert_element_type_33: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_24, torch.float32);  view_24 = None
        add_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_6, convert_element_type_33);  mul_6 = convert_element_type_33 = None
        convert_element_type_34: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_35: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_32, torch.float32);  convert_element_type_32 = None
        mul_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_9, arg26_1);  arg26_1 = None
        mul_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, 768)
        sum_11: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_17, [2], True)
        mul_19: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg27_1);  mul_17 = None
        sum_12: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_19, [2], True);  mul_19 = None
        mul_20: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg27_1, sum_12);  sum_12 = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_18, sum_11);  mul_18 = sum_11 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_2, mul_20);  sub_2 = mul_20 = None
        mul_21: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg28_1, sub_3);  arg28_1 = sub_3 = None
        mul_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_9, arg27_1);  arg27_1 = None
        sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_22, [0, 1]);  mul_22 = None
        sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_9, [0, 1]);  add_9 = None
        convert_element_type_36: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16)
        convert_element_type_37: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        mul_23: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_24: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_36, mul_23);  convert_element_type_36 = mul_23 = None
        view_25: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_24, _shape_param_28);  mul_24 = _shape_param_28 = None
        mm_12: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_25, arg30_1);  arg30_1 = None
        permute_15: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_25, [1, 0])
        mm_13: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_15, arg31_1);  permute_15 = arg31_1 = None
        sum_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_25, [0], True, dtype = torch.float32);  view_25 = None
        view_26: "f32[768]" = torch.ops.aten.view.default(sum_15, _shape_param_29);  sum_15 = _shape_param_29 = None
        convert_element_type_38: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_26, torch.bfloat16);  view_26 = None
        view_27: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_12, _shape_param_30);  mm_12 = _shape_param_30 = None
        convert_element_type_39: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_40: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_38, torch.float32);  convert_element_type_38 = None
        view_28: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_27, _shape_param_31);  view_27 = _shape_param_31 = None
        slice_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_28, 2, 0, 6)
        slice_2: "bf16[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_28, 2, 6, 12);  view_28 = None
        view_29: "bf16[16384, 384]" = torch.ops.aten.view.default(slice_2, _shape_param_32);  slice_2 = _shape_param_32 = None
        permute_16: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(slice_1, [0, 2, 1, 3]);  slice_1 = None
        clone_4: "bf16[32, 6, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_30: "bf16[32, 512, 384]" = torch.ops.aten.view.default(arg32_1, _shape_param_33);  arg32_1 = _shape_param_33 = None
        view_31: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view_30, _shape_param_34);  _shape_param_34 = None
        permute_17: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
        clone_5: "bf16[16384, 384]" = torch.ops.aten.clone.default(view_29, memory_format = torch.contiguous_format);  view_29 = None
        view_32: "bf16[98304, 64, 1]" = torch.ops.aten.view.default(clone_5, _shape_param_35);  clone_5 = _shape_param_35 = None
        bmm: "bf16[98304, 9, 1]" = torch.ops.aten.bmm.default(arg33_1, view_32);  arg33_1 = None
        convert_element_type_41: "bf16[54]" = torch.ops.prims.convert_element_type.default(arg34_1, torch.bfloat16);  arg34_1 = None
        view_33: "bf16[32, 512, 54]" = torch.ops.aten.view.default(arg35_1, _shape_param_36);  arg35_1 = _shape_param_36 = None
        add_10: "bf16[32, 512, 54]" = torch.ops.aten.add.Tensor(view_33, convert_element_type_41);  view_33 = convert_element_type_41 = None
        view_34: "bf16[98304, 9, 1]" = torch.ops.aten.view.default(add_10, _shape_param_37);  add_10 = _shape_param_37 = None
        convert_element_type_42: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(view_34, torch.float32);  view_34 = None
        sub_4: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(convert_element_type_42, arg36_1);  convert_element_type_42 = arg36_1 = None
        exp_1: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        div: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp_1, arg37_1);  exp_1 = arg37_1 = None
        convert_element_type_43: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
        mul_25: "f32[98304, 9, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_43, div);  convert_element_type_43 = None
        sum_16: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_25, [1], True)
        neg: "f32[98304, 9, 1]" = torch.ops.aten.neg.default(div)
        fma: "f32[98304, 9, 1]" = torch.ops.prims.fma.default(neg, sum_16, mul_25);  neg = sum_16 = mul_25 = None
        convert_element_type_44: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_35: "bf16[32, 512, 54]" = torch.ops.aten.view.default(convert_element_type_44, _shape_param_38);  convert_element_type_44 = _shape_param_38 = None
        sum_17: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_35, [0, 1], True, dtype = torch.float32)
        view_36: "f32[54]" = torch.ops.aten.view.default(sum_17, _shape_param_39);  sum_17 = _shape_param_39 = None
        convert_element_type_45: "bf16[54]" = torch.ops.prims.convert_element_type.default(view_36, torch.bfloat16);  view_36 = None
        view_37: "bf16[16384, 54]" = torch.ops.aten.view.default(view_35, _shape_param_40);  view_35 = _shape_param_40 = None
        permute_18: "bf16[54, 16384]" = torch.ops.aten.permute.default(view_37, [1, 0])
        mm_14: "bf16[54, 384]" = torch.ops.aten.mm.default(permute_18, arg38_1);  permute_18 = arg38_1 = None
        mm_15: "bf16[16384, 384]" = torch.ops.aten.mm.default(view_37, arg39_1);  view_37 = arg39_1 = None
        view_38: "bf16[32, 512, 384]" = torch.ops.aten.view.default(mm_15, _shape_param_41);  mm_15 = _shape_param_41 = None
        convert_element_type_46: "f32[54, 384]" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        convert_element_type_47: "f32[54]" = torch.ops.prims.convert_element_type.default(convert_element_type_45, torch.float32);  convert_element_type_45 = None
        mul_26: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(view_38, view_30);  view_30 = None
        permute_19: "bf16[32, 384, 512]" = torch.ops.aten.permute.default(mul_26, [0, 2, 1]);  mul_26 = None
        sum_18: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_19, [0, 2], True, dtype = torch.float32)
        view_39: "f32[384, 1]" = torch.ops.aten.view.default(sum_18, _shape_param_42);  sum_18 = _shape_param_42 = None
        return (full, convert_element_type_8, convert_element_type_9, convert_element_type_13, convert_element_type_14, convert_element_type_18, convert_element_type_19, convert_element_type_22, convert_element_type_23, sum_7, sum_8, convert_element_type_27, convert_element_type_28, convert_element_type_34, convert_element_type_35, mul_21, sum_13, sum_14, convert_element_type_39, convert_element_type_40, clone_4, permute_17, view_32, div, view_38, convert_element_type_46, convert_element_type_47, permute_19, view_39)



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
