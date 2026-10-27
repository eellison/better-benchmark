"""
Standalone repro captured via capture_hook.
Label: AllenaiLongformerBase
Pattern hash: a35a7cbea9d9
Shape hash: 6f3f4cf4
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
    def forward(self, arg0_1: "f32[8, 1024, 768]", arg1_1: "f32[768]", arg2_1: "f32[8, 1024, 768]", arg3_1: "f32[8, 1024, 1]", arg4_1: "b8[8, 1024, 768]", arg5_1: "bf16[768, 3072]", arg6_1: "bf16[8192, 3072]", arg7_1: "bf16[8192, 3072]", arg8_1: "bf16[3072, 768]", arg9_1: "bf16[8192, 768]", arg10_1: "f32[768]", arg11_1: "f32[8, 1024, 768]", arg12_1: "f32[8, 1024, 1]", arg13_1: "b8[8, 1024, 768]", arg14_1: "bf16[8192, 768]", arg15_1: "bf16[768, 768]", arg16_1: "bf16[384, 768, 256]", arg17_1: "bf16[384, 64, 768]", arg18_1: "b8[8, 1024, 12, 513]", arg19_1: "b8[8, 1024]", arg20_1: "bf16[8, 1024, 12, 513]", arg21_1: "f32[8, 1024, 12, 1]", arg22_1: "f32[8, 1024, 12, 1]", arg23_1: "bf16[1, 256, 1, 257]", arg24_1: "bf16[1, 256, 1, 257]", arg25_1: "bf16[288, 64, 512]", arg26_1: "bf16[288, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49):
        # No stacktrace found for following nodes
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = None
        sum_2: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg2_1, sum_2);  arg2_1 = sum_2 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg3_1, sub_1);  arg3_1 = sub_1 = None
        convert_element_type: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        convert_element_type_1: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        mul_5: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_6: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_5);  convert_element_type = mul_5 = None
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(mul_6, _shape_param_0);  mul_6 = _shape_param_0 = None
        mm: "bf16[8192, 3072]" = torch.ops.aten.mm.default(view, arg5_1);  arg5_1 = None
        permute: "bf16[768, 8192]" = torch.ops.aten.permute.default(view, [1, 0])
        mm_1: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute, arg6_1);  permute = arg6_1 = None
        sum_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32);  view = None
        view_1: "f32[768]" = torch.ops.aten.view.default(sum_3, _shape_param_1);  sum_3 = _shape_param_1 = None
        convert_element_type_2: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        view_2: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type_3: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_4: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        convert_element_type_5: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        view_3: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(arg7_1, _shape_param_3);  arg7_1 = _shape_param_3 = None
        convert_element_type_6: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_7: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476)
        erf: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_9: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_6, convert_element_type_6)
        mul_10: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_9, -0.5);  mul_9 = None
        exp: "f32[8, 1024, 3072]" = torch.ops.aten.exp.default(mul_10);  mul_10 = None
        mul_11: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_12: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_6, mul_11);  convert_element_type_6 = mul_11 = None
        add_1: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_8, mul_12);  mul_8 = mul_12 = None
        mul_13: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_5, add_1);  convert_element_type_5 = add_1 = None
        convert_element_type_7: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        view_4: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_4);  convert_element_type_7 = _shape_param_4 = None
        mm_2: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_4, arg8_1);  arg8_1 = None
        permute_1: "bf16[3072, 8192]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_3: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_1, arg9_1);  permute_1 = arg9_1 = None
        sum_4: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[3072]" = torch.ops.aten.view.default(sum_4, _shape_param_5);  sum_4 = _shape_param_5 = None
        convert_element_type_8: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_6);  mm_2 = _shape_param_6 = None
        convert_element_type_9: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        add_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_4, convert_element_type_9);  mul_4 = convert_element_type_9 = None
        convert_element_type_10: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_11: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        mul_14: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_2, arg10_1);  arg10_1 = None
        mul_15: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_14, 768)
        sum_5: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_14, [2], True)
        mul_16: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg11_1);  mul_14 = None
        sum_6: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_16, [2], True);  mul_16 = None
        mul_17: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg11_1, sum_6);  sum_6 = None
        sub_2: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_15, sum_5);  mul_15 = sum_5 = None
        sub_3: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_2, mul_17);  sub_2 = mul_17 = None
        mul_18: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg12_1, sub_3);  arg12_1 = sub_3 = None
        mul_19: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_2, arg11_1);  arg11_1 = None
        sum_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_19, [0, 1]);  mul_19 = None
        sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_2, [0, 1]);  add_2 = None
        convert_element_type_12: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16)
        convert_element_type_13: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        mul_20: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_21: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, mul_20);  convert_element_type_12 = mul_20 = None
        sum_9: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_21, [0, 1], True, dtype = torch.float32)
        view_7: "f32[768]" = torch.ops.aten.view.default(sum_9, _shape_param_7);  sum_9 = _shape_param_7 = None
        convert_element_type_14: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        view_8: "bf16[8192, 768]" = torch.ops.aten.view.default(mul_21, _shape_param_8);  mul_21 = _shape_param_8 = None
        permute_2: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_4: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_2, arg14_1);  permute_2 = arg14_1 = None
        mm_5: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_8, arg15_1);  view_8 = arg15_1 = None
        view_9: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_5, _shape_param_9);  mm_5 = _shape_param_9 = None
        convert_element_type_15: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_16: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        permute_3: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(view_9, [1, 0, 2]);  view_9 = None
        view_10: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(permute_3, _shape_param_10);  permute_3 = _shape_param_10 = None
        permute_4: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_10, [1, 0, 2, 3]);  view_10 = None
        permute_5: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_4, [0, 2, 1, 3]);  permute_4 = None
        clone: "bf16[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_11: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(clone, _shape_param_11);  clone = _shape_param_11 = None
        view_12: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_11, _shape_param_12);  view_11 = _shape_param_12 = None
        permute_6: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_12, [0, 1, 2, 4, 3]);  view_12 = None
        view_13: "bf16[384, 256, 64]" = torch.ops.aten.view.default(permute_6, _shape_param_13);  permute_6 = _shape_param_13 = None
        bmm: "bf16[384, 768, 64]" = torch.ops.aten.bmm.default(arg16_1, view_13);  arg16_1 = None
        bmm_1: "bf16[384, 256, 768]" = torch.ops.aten.bmm.default(view_13, arg17_1);  view_13 = arg17_1 = None
        view_14: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm, _shape_param_14);  bmm = _shape_param_14 = None
        view_15: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_1, _shape_param_15);  bmm_1 = _shape_param_15 = None
        squeeze: "bf16[96, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_14, 4);  view_14 = None
        squeeze_1: "bf16[96, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_15, 4);  view_15 = None
        full: "bf16[96, 4, 256, 769]" = torch.ops.aten.full.default(_shape_param_16, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_16 = None
        slice_scatter: "bf16[96, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full, squeeze_1, 3, 0, -1);  squeeze_1 = None
        view_16: "bf16[96, 4, 196864]" = torch.ops.aten.view.default(slice_scatter, _shape_param_17);  slice_scatter = _shape_param_17 = None
        full_1: "bf16[96, 4, 197120]" = torch.ops.aten.full.default(_shape_param_18, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_18 = None
        slice_scatter_1: "bf16[96, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_1, view_16, 2, 0, -256);  view_16 = None
        view_17: "bf16[96, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_1, _shape_param_19);  slice_scatter_1 = _shape_param_19 = None
        constant_pad_nd: "bf16[96, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_17, _shape_param_20);  view_17 = _shape_param_20 = None
        view_18: "bf16[18874368]" = torch.ops.aten.view.default(squeeze, [-1]);  squeeze = None
        view_19: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_21);  constant_pad_nd = _shape_param_21 = None
        permute_7: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
        convert_element_type_17: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        mul_22: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_23: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_7, mul_22);  permute_7 = mul_22 = None
        clone_1: "bf16[8, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_23, memory_format = torch.contiguous_format);  mul_23 = None
        convert_element_type_18: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_1, torch.float32);  clone_1 = None
        unsqueeze: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, 2);  arg19_1 = None
        unsqueeze_1: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 3);  unsqueeze = None
        full_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_1, full_2, convert_element_type_18);  convert_element_type_18 = None
        convert_element_type_19: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.float32);  arg20_1 = None
        clone_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_19, memory_format = torch.contiguous_format);  convert_element_type_19 = None
        sub_4: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_2, arg21_1);  clone_2 = arg21_1 = None
        exp_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        div: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_1, arg22_1);  exp_1 = arg22_1 = None
        mul_24: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where, div);  where = None
        sum_10: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_24, [-1], True)
        neg: "f32[8, 1024, 12, 513]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[8, 1024, 12, 513]" = torch.ops.prims.fma.default(neg, sum_10, mul_24);  neg = sum_10 = mul_24 = None
        convert_element_type_20: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        permute_8: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_20, [0, 2, 1, 3]);  convert_element_type_20 = None
        clone_3: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_20: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_3, _shape_param_22);  clone_3 = _shape_param_22 = None
        view_21: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_20, _shape_param_23);  view_20 = _shape_param_23 = None
        permute_9: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        clone_4: "bf16[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format)
        copy: "bf16[8, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_9, clone_4);  permute_9 = clone_4 = None
        permute_10: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(copy, [0, 2, 1, 3]);  copy = None
        view_22: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_10, _shape_param_24);  permute_10 = _shape_param_24 = None
        view_23: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_22, _shape_param_25);  view_22 = _shape_param_25 = None
        permute_11: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        slice_1: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_11, 1, -256, 9223372036854775807)
        slice_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1, 3, -257, 9223372036854775807)
        clone_5: "bf16[8, 256, 12, 257]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format)
        full_3: "bf16[8, 256, 12, 257]" = torch.ops.aten.full.default(_shape_param_26, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_26 = None
        copy_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2, full_3);  slice_2 = None
        slice_scatter_2: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1, copy_1, 3, -257, 9223372036854775807);  slice_1 = copy_1 = None
        slice_scatter_3: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_11, slice_scatter_2, 1, -256, 9223372036854775807);  permute_11 = slice_scatter_2 = None
        permute_12: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_3, [0, 2, 1, 3]);  slice_scatter_3 = None
        view_24: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_12, _shape_param_27);  permute_12 = _shape_param_27 = None
        full_4: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(arg23_1, _shape_param_28);  arg23_1 = _shape_param_28 = None
        convert_element_type_21: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
        where_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_21, full_4, clone_5);  clone_5 = None
        full_5: "bf16[8, 256, 12, 513]" = torch.ops.aten.full.default(_shape_param_29, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_29 = None
        slice_scatter_4: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_5, where_1, 3, -257, 9223372036854775807);  where_1 = None
        full_6: "bf16[8, 1024, 12, 513]" = torch.ops.aten.full.default(_shape_param_30, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_30 = None
        slice_scatter_5: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_6, slice_scatter_4, 1, -256, 9223372036854775807);  slice_scatter_4 = None
        permute_13: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_5, [0, 2, 1, 3]);  slice_scatter_5 = None
        clone_6: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_25: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_6, _shape_param_31);  clone_6 = _shape_param_31 = None
        add_3: "bf16[96, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_24, view_25);  view_24 = view_25 = None
        view_26: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(add_3, _shape_param_32);  add_3 = _shape_param_32 = None
        permute_14: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        slice_3: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_14, 1, 0, 256)
        slice_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 257)
        clone_7: "bf16[8, 256, 12, 257]" = torch.ops.aten.clone.default(slice_4, memory_format = torch.contiguous_format)
        copy_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_4, full_3);  slice_4 = None
        slice_scatter_6: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_2, 3, 0, 257);  slice_3 = copy_2 = None
        slice_scatter_7: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_14, slice_scatter_6, 1, 0, 256);  permute_14 = slice_scatter_6 = None
        permute_15: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_7, [0, 2, 1, 3]);  slice_scatter_7 = None
        view_27: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_15, _shape_param_33);  permute_15 = _shape_param_33 = None
        expand_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(arg24_1, _shape_param_34);  arg24_1 = _shape_param_34 = None
        convert_element_type_22: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
        where_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_22, full_4, clone_7);  clone_7 = None
        slice_scatter_8: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_5, where_2, 3, 0, 257);  where_2 = None
        slice_scatter_9: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_6, slice_scatter_8, 1, 0, 256);  slice_scatter_8 = None
        permute_16: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_9, [0, 2, 1, 3]);  slice_scatter_9 = None
        clone_8: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_28: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_8, _shape_param_35);  clone_8 = _shape_param_35 = None
        add_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_27, view_28);  view_27 = view_28 = None
        select: "bf16[96, 256, 513]" = torch.ops.aten.select.int(add_4, 1, 0)
        slice_5: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select, 1, 1, 256)
        slice_6: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_5, 2, 1, 256)
        clone_9: "bf16[96, 255, 255]" = torch.ops.aten.clone.default(slice_6, memory_format = torch.contiguous_format)
        full_7: "bf16[96, 255, 255]" = torch.ops.aten.full.default(_shape_param_36, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_36 = None
        copy_3: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_6, full_7);  slice_6 = None
        slice_scatter_10: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_5, copy_3, 2, 1, 256);  slice_5 = copy_3 = None
        slice_scatter_11: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select, slice_scatter_10, 1, 1, 256);  select = slice_scatter_10 = None
        select_scatter: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_4, slice_scatter_11, 1, 0);  add_4 = slice_scatter_11 = None
        full_8: "bf16[96, 255, 513]" = torch.ops.aten.full.default(_shape_param_37, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_37 = None
        slice_scatter_12: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(full_8, clone_9, 2, -255, 9223372036854775807);  clone_9 = None
        full_9: "bf16[96, 512, 513]" = torch.ops.aten.full.default(_shape_param_38, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_38 = None
        slice_scatter_13: "bf16[96, 512, 513]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_12, 1, 0, 255);  slice_scatter_12 = None
        full_10: "bf16[96, 3, 512, 513]" = torch.ops.aten.full.default(_shape_param_39, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_39 = None
        select_scatter_1: "bf16[96, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_10, slice_scatter_13, 1, 0);  slice_scatter_13 = None
        slice_7: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_8: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 256)
        clone_10: "bf16[96, 3, 256, 256]" = torch.ops.aten.clone.default(slice_8, memory_format = torch.contiguous_format)
        full_11: "bf16[96, 3, 256, 256]" = torch.ops.aten.full.default(_shape_param_40, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_40 = None
        copy_4: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_8, full_11);  slice_8 = None
        slice_scatter_14: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_4, 3, 0, 256);  slice_7 = copy_4 = None
        slice_scatter_15: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_14, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_14 = None
        full_12: "bf16[96, 3, 256, 513]" = torch.ops.aten.full.default(_shape_param_41, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_41 = None
        slice_scatter_16: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_12, clone_10, 3, 257, 9223372036854775807);  clone_10 = None
        slice_scatter_17: "bf16[96, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_10, slice_scatter_16, 2, -257, -1);  slice_scatter_16 = None
        add_5: "bf16[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_1, slice_scatter_17);  select_scatter_1 = slice_scatter_17 = None
        select_1: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, -1)
        slice_9: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807)
        clone_11: "bf16[96, 256, 257]" = torch.ops.aten.clone.default(slice_9, memory_format = torch.contiguous_format)
        full_13: "bf16[96, 256, 257]" = torch.ops.aten.full.default(_shape_param_42, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_42 = None
        copy_5: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_9, full_13);  slice_9 = None
        slice_scatter_18: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_1, copy_5, 2, 256, 9223372036854775807);  select_1 = copy_5 = None
        select_scatter_2: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_15, slice_scatter_18, 1, -1);  slice_scatter_15 = slice_scatter_18 = None
        full_14: "bf16[96, 256, 513]" = torch.ops.aten.full.default(_shape_param_43, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_43 = None
        slice_scatter_19: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(full_14, clone_11, 2, 0, 257);  clone_11 = None
        slice_scatter_20: "bf16[96, 512, 513]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_19, 1, 256, 9223372036854775807);  slice_scatter_19 = None
        select_scatter_3: "bf16[96, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_10, slice_scatter_20, 1, -1);  slice_scatter_20 = None
        add_6: "bf16[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_5, select_scatter_3);  add_5 = select_scatter_3 = None
        slice_10: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 0, -1);  select_scatter_2 = None
        slice_11: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_10, 3, 256, 9223372036854775807);  slice_10 = None
        clone_12: "bf16[96, 3, 256, 257]" = torch.ops.aten.clone.default(slice_11, memory_format = torch.contiguous_format);  slice_11 = None
        slice_scatter_21: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_12, clone_12, 3, 0, 257);  clone_12 = None
        slice_scatter_22: "bf16[96, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_10, slice_scatter_21, 2, 0, 256);  slice_scatter_21 = None
        add_7: "bf16[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_6, slice_scatter_22);  add_6 = slice_scatter_22 = None
        view_29: "bf16[96, 3, 513, 512]" = torch.ops.aten.view.default(add_7, _shape_param_44);  add_7 = _shape_param_44 = None
        constant_pad_nd_1: "bf16[96, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_29, [0, 0, 0, -1]);  view_29 = None
        view_30: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_1, _shape_param_45);  constant_pad_nd_1 = _shape_param_45 = None
        permute_17: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_30, [0, 1, 2, 4, 3]);  view_30 = None
        view_31: "bf16[288, 512, 512]" = torch.ops.aten.view.default(permute_17, _shape_param_46);  permute_17 = _shape_param_46 = None
        bmm_2: "bf16[288, 64, 512]" = torch.ops.aten.bmm.default(arg25_1, view_31);  arg25_1 = None
        bmm_3: "bf16[288, 512, 64]" = torch.ops.aten.bmm.default(view_31, arg26_1);  view_31 = arg26_1 = None
        view_32: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_2, _shape_param_47);  bmm_2 = _shape_param_47 = None
        permute_18: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 1, 4, 3, 2]);  view_32 = None
        view_33: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_3, _shape_param_48);  bmm_3 = _shape_param_48 = None
        permute_19: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_18, [0, 1, 3, 4, 2]);  permute_18 = None
        squeeze_2: "bf16[96, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_19, 4);  permute_19 = None
        squeeze_3: "bf16[96, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_33, 4);  view_33 = None
        clone_13: "bf16[96, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        view_34: "bf16[9437184]" = torch.ops.aten.view.default(clone_13, _shape_param_49);  clone_13 = _shape_param_49 = None
        view_35: "bf16[9437184]" = torch.ops.aten.view.default(squeeze_3, [-1]);  squeeze_3 = None
        return (convert_element_type_3, convert_element_type_4, convert_element_type_10, convert_element_type_11, mul_18, sum_7, sum_8, convert_element_type_15, convert_element_type_16, full, full_1, view_18, unsqueeze_1, full_2, full_3, full_4, convert_element_type_21, full_5, full_6, convert_element_type_22, full_7, full_8, full_9, full_10, full_11, full_12, full_13, full_14, view_34, view_35)



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
