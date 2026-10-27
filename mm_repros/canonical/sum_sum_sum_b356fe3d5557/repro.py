"""
Standalone repro captured via capture_hook.
Label: MegatronBertForCausalLM
Pattern hash: b356fe3d5557
Shape hash: bf99c910
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[16, 513]", arg3_1: "bf16[16, 512, 29056]", arg4_1: "f32[8192, 1]", arg5_1: "f32[8192, 1]", arg6_1: "bf16[16, 512, 29056]", arg7_1: "bf16[29056, 1024]", arg8_1: "f32[1024]", arg9_1: "bf16[8192, 1024]", arg10_1: "f32[16, 512, 1]", arg11_1: "f32[16, 512, 1]", arg12_1: "bf16[1024, 1024]", arg13_1: "bf16[8192, 1024]", arg14_1: "f32[1024]", arg15_1: "f32[16, 512, 1024]", arg16_1: "f32[16, 512, 1]", arg17_1: "b8[16, 512, 1024]", arg18_1: "bf16[1024, 4096]", arg19_1: "bf16[8192, 4096]", arg20_1: "bf16[8192, 4096]", arg21_1: "bf16[4096, 1024]", arg22_1: "bf16[8192, 1024]", arg23_1: "f32[1024]", arg24_1: "f32[16, 512, 1024]", arg25_1: "f32[16, 512, 1]", arg26_1: "b8[16, 512, 1024]", arg27_1: "bf16[1024, 1024]", arg28_1: "bf16[8192, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        slice_1: "i64[16, 512]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 1, 9223372036854775807);  arg2_1 = None
        clone: "i64[16, 512]" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None
        view: "i64[8192]" = torch.ops.aten.view.default(clone, [-1]);  clone = None
        unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[29056]" = torch.ops.prims.iota.default(29056, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 29056]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[8192, 29056]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[8192, 29056]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[8192, 29056]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[8192, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = full_1 = None
        mul: "f32[8192, 29056]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "f32[16, 512, 29056]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        view_2: "f32[8192, 29056]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        sub: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(view_2, arg4_1);  view_2 = arg4_1 = None
        sub_1: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        exp: "f32[8192, 29056]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        mul_1: "f32[8192, 29056]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        view_3: "f32[16, 512, 29056]" = torch.ops.aten.view.default(sub_2, _shape_param_3);  sub_2 = _shape_param_3 = None
        convert_element_type_1: "bf16[16, 512, 29056]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        add: "bf16[16, 512, 29056]" = torch.ops.aten.add.Tensor(arg6_1, convert_element_type_1);  arg6_1 = convert_element_type_1 = None
        view_4: "bf16[8192, 29056]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_4, arg7_1);  arg7_1 = None
        sum_2: "f32[1, 29056]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32)
        view_5: "f32[29056]" = torch.ops.aten.view.default(sum_2, _shape_param_5);  sum_2 = _shape_param_5 = None
        convert_element_type_2: "bf16[29056]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm, _shape_param_6);  mm = _shape_param_6 = None
        convert_element_type_3: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_4: "f32[29056]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        mul_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_3, arg8_1);  arg8_1 = None
        mul_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_2, 1024)
        sum_3: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        view_7: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(arg9_1, _shape_param_7);  arg9_1 = _shape_param_7 = None
        convert_element_type_5: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        mul_4: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.5)
        mul_5: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.7071067811865476)
        erf: "f32[16, 512, 1024]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_4, add_1);  mul_4 = None
        convert_element_type_6: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_7: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        sub_3: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_7, arg10_1);  convert_element_type_7 = arg10_1 = None
        mul_7: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_3, arg11_1);  sub_3 = None
        mul_8: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_2, mul_7);  mul_2 = None
        sum_4: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True);  mul_8 = None
        mul_9: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_7, sum_4);  sum_4 = None
        sub_4: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_5: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_4, mul_9);  sub_4 = mul_9 = None
        div_1: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(arg11_1, 1024);  arg11_1 = None
        mul_10: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_11: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_3, mul_7);  mul_7 = None
        sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_11, [0, 1]);  mul_11 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0, 1]);  convert_element_type_3 = None
        convert_element_type_8: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        convert_element_type_9: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        mul_12: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_13: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_5, convert_element_type_5)
        mul_14: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_13, -0.5);  mul_13 = None
        exp_1: "f32[16, 512, 1024]" = torch.ops.aten.exp.default(mul_14);  mul_14 = None
        mul_15: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_16: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_5, mul_15);  convert_element_type_5 = mul_15 = None
        add_2: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_12, mul_16);  mul_12 = mul_16 = None
        mul_17: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_9, add_2);  convert_element_type_9 = add_2 = None
        convert_element_type_10: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        view_8: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_10, _shape_param_8);  convert_element_type_10 = _shape_param_8 = None
        mm_1: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_8, arg12_1);  arg12_1 = None
        permute: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_2: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute, arg13_1);  permute = arg13_1 = None
        sum_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_9: "f32[1024]" = torch.ops.aten.view.default(sum_7, _shape_param_9);  sum_7 = _shape_param_9 = None
        convert_element_type_11: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm_1, _shape_param_10);  mm_1 = _shape_param_10 = None
        convert_element_type_12: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        convert_element_type_13: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_14: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        mul_18: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, arg14_1);  arg14_1 = None
        mul_19: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_18, 1024)
        sum_8: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True)
        mul_20: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_18, arg15_1);  mul_18 = None
        sum_9: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True);  mul_20 = None
        mul_21: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg15_1, sum_9);  sum_9 = None
        sub_6: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_19, sum_8);  mul_19 = sum_8 = None
        sub_7: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_6, mul_21);  sub_6 = mul_21 = None
        mul_22: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg16_1, sub_7);  arg16_1 = sub_7 = None
        mul_23: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, arg15_1);  arg15_1 = None
        sum_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_23, [0, 1]);  mul_23 = None
        sum_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_12, [0, 1]);  convert_element_type_12 = None
        convert_element_type_15: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16)
        convert_element_type_16: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        mul_24: "bf16[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_25: "bf16[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_15, mul_24);  convert_element_type_15 = mul_24 = None
        view_11: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_25, _shape_param_11);  mul_25 = _shape_param_11 = None
        mm_3: "bf16[8192, 4096]" = torch.ops.aten.mm.default(view_11, arg18_1);  arg18_1 = None
        permute_1: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_4: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_1, arg19_1);  permute_1 = arg19_1 = None
        sum_12: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True, dtype = torch.float32);  view_11 = None
        view_12: "f32[1024]" = torch.ops.aten.view.default(sum_12, _shape_param_12);  sum_12 = _shape_param_12 = None
        convert_element_type_17: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        view_13: "bf16[16, 512, 4096]" = torch.ops.aten.view.default(mm_3, _shape_param_13);  mm_3 = _shape_param_13 = None
        convert_element_type_18: "f32[1024, 4096]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_19: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_17, torch.float32);  convert_element_type_17 = None
        convert_element_type_20: "f32[16, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        view_14: "bf16[16, 512, 4096]" = torch.ops.aten.view.default(arg20_1, _shape_param_14);  arg20_1 = _shape_param_14 = None
        convert_element_type_21: "f32[16, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        mul_26: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.7071067811865476)
        erf_1: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_3: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_27: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
        mul_28: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_21, convert_element_type_21)
        mul_29: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_28, -0.5);  mul_28 = None
        exp_2: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_29);  mul_29 = None
        mul_30: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_31: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_21, mul_30);  convert_element_type_21 = mul_30 = None
        add_4: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_27, mul_31);  mul_27 = mul_31 = None
        mul_32: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_20, add_4);  convert_element_type_20 = add_4 = None
        convert_element_type_22: "bf16[16, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        view_15: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_22, _shape_param_15);  convert_element_type_22 = _shape_param_15 = None
        mm_5: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_15, arg21_1);  arg21_1 = None
        permute_2: "bf16[4096, 8192]" = torch.ops.aten.permute.default(view_15, [1, 0])
        mm_6: "bf16[4096, 1024]" = torch.ops.aten.mm.default(permute_2, arg22_1);  permute_2 = arg22_1 = None
        sum_13: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_15, [0], True, dtype = torch.float32);  view_15 = None
        view_16: "f32[4096]" = torch.ops.aten.view.default(sum_13, _shape_param_16);  sum_13 = _shape_param_16 = None
        convert_element_type_23: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_16, torch.bfloat16);  view_16 = None
        view_17: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm_5, _shape_param_17);  mm_5 = _shape_param_17 = None
        convert_element_type_24: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        convert_element_type_25: "f32[4096, 1024]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_26: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_23, torch.float32);  convert_element_type_23 = None
        mul_33: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, arg23_1);  arg23_1 = None
        mul_34: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_33, 1024)
        sum_14: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_33, [2], True)
        mul_35: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_33, arg24_1);  mul_33 = None
        sum_15: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_35, [2], True);  mul_35 = None
        mul_36: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg24_1, sum_15);  sum_15 = None
        sub_8: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_34, sum_14);  mul_34 = sum_14 = None
        sub_9: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_8, mul_36);  sub_8 = mul_36 = None
        mul_37: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg25_1, sub_9);  arg25_1 = sub_9 = None
        mul_38: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, arg24_1);  arg24_1 = None
        sum_16: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_38, [0, 1]);  mul_38 = None
        sum_17: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_24, [0, 1]);  convert_element_type_24 = None
        add_5: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_22, mul_37);  mul_22 = mul_37 = None
        convert_element_type_27: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16)
        convert_element_type_28: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.bfloat16);  arg26_1 = None
        mul_39: "bf16[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_40: "bf16[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_27, mul_39);  convert_element_type_27 = mul_39 = None
        view_18: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_40, _shape_param_18);  mul_40 = _shape_param_18 = None
        mm_7: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_18, arg27_1);  arg27_1 = None
        permute_3: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_8: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_3, arg28_1);  permute_3 = arg28_1 = None
        sum_18: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True, dtype = torch.float32);  view_18 = None
        view_19: "f32[1024]" = torch.ops.aten.view.default(sum_18, _shape_param_19);  sum_18 = _shape_param_19 = None
        convert_element_type_29: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_19, torch.bfloat16);  view_19 = None
        view_20: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm_7, _shape_param_20);  mm_7 = _shape_param_20 = None
        convert_element_type_30: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        convert_element_type_31: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_29, torch.float32);  convert_element_type_29 = None
        view_21: "bf16[16, 512, 16, 64]" = torch.ops.aten.view.default(view_20, _shape_param_21);  view_20 = _shape_param_21 = None
        permute_4: "bf16[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        clone_1: "bf16[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        return (view_4, convert_element_type_4, sum_5, sum_6, convert_element_type_13, convert_element_type_14, sum_10, sum_11, convert_element_type_18, convert_element_type_19, convert_element_type_25, convert_element_type_26, sum_16, sum_17, add_5, convert_element_type_30, convert_element_type_31, clone_1)



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
