"""
Standalone repro captured via capture_hook.
Label: XLNetLMHeadModel
Pattern hash: 4ceff1032b87
Shape hash: 8e004e83
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[16, 512]", arg3_1: "bf16[16, 512, 32000]", arg4_1: "f32[8192, 1]", arg5_1: "f32[8192, 1]", arg6_1: "bf16[16, 512, 32000]", arg7_1: "bf16[32000, 1024]", arg8_1: "b8[512, 16, 1024]", arg9_1: "f32[1024]", arg10_1: "f32[512, 16, 1024]", arg11_1: "f32[512, 16, 1]", arg12_1: "b8[512, 16, 1024]", arg13_1: "bf16[1024, 4096]", arg14_1: "bf16[8192, 4096]", arg15_1: "b8[512, 16, 4096]", arg16_1: "bf16[8192, 4096]", arg17_1: "bf16[4096, 1024]", arg18_1: "bf16[8192, 1024]", arg19_1: "f32[1024]", arg20_1: "f32[512, 16, 1024]", arg21_1: "f32[512, 16, 1]", arg22_1: "b8[512, 16, 1024]", arg23_1: "bf16[1024, 8192]", arg24_1: "bf16[1024, 1024]", arg25_1: "bf16[256, 64, 512]", arg26_1: "b8[16, 16, 512, 512]", arg27_1: "bf16[16, 16, 512, 512]", arg28_1: "f32[16, 16, 512, 1]", arg29_1: "f32[16, 16, 512, 1]", arg30_1: "b8[16, 16, 512, 1]", arg31_1: "f32[16, 16, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "i64[8192]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[32000]" = torch.ops.prims.iota.default(32000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 32000]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[8192, 32000]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[8192, 32000]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[8192, 32000]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[8192, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = full_1 = None
        mul: "f32[8192, 32000]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "bf16[8192, 32000]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[8192, 32000]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        view_2: "bf16[8192, 32000]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[8192, 32000]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[8192, 32000]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        sub_1: "f32[8192, 32000]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        convert_element_type_3: "bf16[8192, 32000]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[8192, 32000]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp: "f32[8192, 32000]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[8192, 32000]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[8192, 32000]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[8192, 32000]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_3: "bf16[16, 512, 32000]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        add: "bf16[16, 512, 32000]" = torch.ops.aten.add.Tensor(arg6_1, view_3);  arg6_1 = view_3 = None
        view_4: "bf16[8192, 32000]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_4, arg7_1);  arg7_1 = None
        sum_2: "f32[1, 32000]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32)
        view_5: "f32[32000]" = torch.ops.aten.view.default(sum_2, _shape_param_5);  sum_2 = _shape_param_5 = None
        convert_element_type_6: "bf16[32000]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm, _shape_param_6);  mm = _shape_param_6 = None
        convert_element_type_7: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_8: "f32[32000]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        permute: "f32[512, 16, 1024]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0, 2]);  convert_element_type_7 = None
        convert_element_type_9: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        mul_2: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_3: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(permute, mul_2);  permute = mul_2 = None
        clone: "f32[512, 16, 1024]" = torch.ops.aten.clone.default(mul_3, memory_format = torch.contiguous_format);  mul_3 = None
        mul_4: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(clone, arg9_1);  arg9_1 = None
        mul_5: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_4, 1024)
        sum_3: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True)
        mul_6: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_4, arg10_1);  mul_4 = None
        sum_4: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_6, [2], True);  mul_6 = None
        mul_7: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg10_1, sum_4);  sum_4 = None
        sub_3: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(mul_5, sum_3);  mul_5 = sum_3 = None
        sub_4: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(sub_3, mul_7);  sub_3 = mul_7 = None
        mul_8: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg11_1, sub_4);  arg11_1 = sub_4 = None
        mul_9: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(clone, arg10_1);  arg10_1 = None
        sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_9, [0, 1]);  mul_9 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(clone, [0, 1]);  clone = None
        convert_element_type_10: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16)
        convert_element_type_11: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        mul_10: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_11: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_10, mul_10);  convert_element_type_10 = mul_10 = None
        view_7: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_11, _shape_param_7);  mul_11 = _shape_param_7 = None
        mm_1: "bf16[8192, 4096]" = torch.ops.aten.mm.default(view_7, arg13_1);  arg13_1 = None
        permute_1: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_7, [1, 0])
        mm_2: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_1, arg14_1);  permute_1 = arg14_1 = None
        sum_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True, dtype = torch.float32);  view_7 = None
        view_8: "f32[1024]" = torch.ops.aten.view.default(sum_7, _shape_param_8);  sum_7 = _shape_param_8 = None
        convert_element_type_12: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        view_9: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(mm_1, _shape_param_9);  mm_1 = _shape_param_9 = None
        convert_element_type_13: "f32[1024, 4096]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_14: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_12, torch.float32);  convert_element_type_12 = None
        convert_element_type_15: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.bfloat16);  arg15_1 = None
        mul_12: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_13: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(view_9, mul_12);  view_9 = mul_12 = None
        convert_element_type_16: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_13, torch.float32);  mul_13 = None
        view_10: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(arg16_1, _shape_param_10);  arg16_1 = _shape_param_10 = None
        convert_element_type_17: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        mul_14: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 0.7071067811865476)
        erf: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_1: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_15: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_16: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_17, convert_element_type_17)
        mul_17: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_16, -0.5);  mul_16 = None
        exp_1: "f32[512, 16, 4096]" = torch.ops.aten.exp.default(mul_17);  mul_17 = None
        mul_18: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_19: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_17, mul_18);  convert_element_type_17 = mul_18 = None
        add_2: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(mul_15, mul_19);  mul_15 = mul_19 = None
        mul_20: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_16, add_2);  convert_element_type_16 = add_2 = None
        convert_element_type_18: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        view_11: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_18, _shape_param_11);  convert_element_type_18 = _shape_param_11 = None
        mm_3: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_11, arg17_1);  arg17_1 = None
        permute_2: "bf16[4096, 8192]" = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_4: "bf16[4096, 1024]" = torch.ops.aten.mm.default(permute_2, arg18_1);  permute_2 = arg18_1 = None
        sum_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True, dtype = torch.float32);  view_11 = None
        view_12: "f32[4096]" = torch.ops.aten.view.default(sum_8, _shape_param_12);  sum_8 = _shape_param_12 = None
        convert_element_type_19: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        view_13: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(mm_3, _shape_param_13);  mm_3 = _shape_param_13 = None
        convert_element_type_20: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        add_3: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_8, convert_element_type_20);  mul_8 = convert_element_type_20 = None
        convert_element_type_21: "f32[4096, 1024]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_22: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_19, torch.float32);  convert_element_type_19 = None
        mul_21: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_3, arg19_1);  arg19_1 = None
        mul_22: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_21, 1024)
        sum_9: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_21, [2], True)
        mul_23: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_21, arg20_1);  mul_21 = None
        sum_10: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_23, [2], True);  mul_23 = None
        mul_24: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg20_1, sum_10);  sum_10 = None
        sub_5: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(mul_22, sum_9);  mul_22 = sum_9 = None
        sub_6: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(sub_5, mul_24);  sub_5 = mul_24 = None
        mul_25: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg21_1, sub_6);  arg21_1 = sub_6 = None
        mul_26: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_3, arg20_1);  arg20_1 = None
        sum_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_26, [0, 1]);  mul_26 = None
        sum_12: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_3, [0, 1]);  add_3 = None
        convert_element_type_23: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16)
        convert_element_type_24: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        mul_27: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_28: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_23, mul_27);  convert_element_type_23 = mul_27 = None
        view_14: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(mul_28, _shape_param_14);  mul_28 = _shape_param_14 = None
        permute_3: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.permute.default(view_14, [0, 1, 3, 4, 2]);  view_14 = None
        view_15: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(permute_3, _shape_param_15);  permute_3 = _shape_param_15 = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_15, 0);  view_15 = None
        mm_5: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg23_1, squeeze);  arg23_1 = None
        unsqueeze_1: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_5, 0);  mm_5 = None
        mm_6: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, arg24_1);  squeeze = arg24_1 = None
        unsqueeze_2: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_6, 0);  mm_6 = None
        view_16: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_16);  unsqueeze_1 = _shape_param_16 = None
        permute_4: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_16, [3, 4, 2, 0, 1]);  view_16 = None
        view_17: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.view.default(unsqueeze_2, _shape_param_17);  unsqueeze_2 = _shape_param_17 = None
        permute_5: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(view_17, [0, 1, 4, 2, 3]);  view_17 = None
        permute_6: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_4, [2, 4, 3, 0, 1]);  permute_4 = None
        squeeze_1: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_6, 4);  permute_6 = None
        squeeze_2: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_1, 3);  squeeze_1 = None
        permute_7: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_5, [0, 1, 4, 3, 2]);  permute_5 = None
        squeeze_3: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_7, 4);  permute_7 = None
        convert_element_type_25: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_2, torch.float32);  squeeze_2 = None
        view_18: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(squeeze_3, _shape_param_18);  squeeze_3 = _shape_param_18 = None
        permute_8: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_18, [1, 2, 0, 4, 3]);  view_18 = None
        clone_1: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_19: "bf16[256, 512, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_19);  clone_1 = _shape_param_19 = None
        bmm: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_19, arg25_1);  arg25_1 = None
        view_20: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.view.default(bmm, _shape_param_20);  bmm = _shape_param_20 = None
        squeeze_4: "bf16[16, 16, 512, 512]" = torch.ops.aten.squeeze.dim(view_20, 4);  view_20 = None
        convert_element_type_26: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(squeeze_4, torch.float32);  squeeze_4 = None
        convert_element_type_27: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.float32);  arg26_1 = None
        mul_29: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
        mul_30: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_26, mul_29);  convert_element_type_26 = mul_29 = None
        mul_31: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(arg27_1, 0.125)
        convert_element_type_28: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_31, torch.float32);  mul_31 = None
        convert_element_type_29: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.float32);  arg27_1 = None
        mul_32: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1);  convert_element_type_29 = None
        sub_7: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_32, arg28_1);  mul_32 = arg28_1 = None
        mul_33: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_7, 0.125);  sub_7 = None
        sub_8: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_28, arg29_1);  convert_element_type_28 = arg29_1 = None
        where_3: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(arg30_1, mul_33, sub_8);  arg30_1 = mul_33 = sub_8 = None
        exp_2: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_3);  where_3 = None
        div_1: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, arg31_1);  exp_2 = arg31_1 = None
        mul_34: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_30, div_1);  mul_30 = None
        sum_13: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_34, [3], True)
        neg: "f32[16, 16, 512, 512]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma: "f32[16, 16, 512, 512]" = torch.ops.prims.fma.default(neg, sum_13, mul_34);  neg = sum_13 = mul_34 = None
        convert_element_type_30: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        mul_35: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.125);  convert_element_type_30 = None
        return (view_4, convert_element_type_8, sum_5, sum_6, convert_element_type_13, convert_element_type_14, convert_element_type_21, convert_element_type_22, mul_25, sum_11, sum_12, convert_element_type_25, view_19, mul_35)



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
