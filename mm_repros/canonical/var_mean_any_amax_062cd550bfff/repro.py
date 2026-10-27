"""
Standalone repro captured via capture_hook.
Label: BlenderbotForConditionalGeneration
Pattern hash: 062cd550bfff
Shape hash: 7f02d12a
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
    def forward(self, arg0_1: "bf16[16, 32, 128, 80]", arg1_1: "bf16[2560, 2560]", arg2_1: "bf16[2560]", arg3_1: "bf16[16, 128, 2560]", arg4_1: "bf16[2560]", arg5_1: "bf16[2560]", arg6_1: "bf16[2560, 2560]", arg7_1: "bf16[2560]", arg8_1: "bf16[16, 128, 2560]", arg9_1: "bf16[2560, 2560]", arg10_1: "bf16[2560]", arg11_1: "bf16[2560, 2560]", arg12_1: "bf16[2560]", arg13_1: "bf16[2560, 2560]", arg14_1: "bf16[2560]", arg15_1: "bf16[2560]", arg16_1: "bf16[2560]", arg17_1: "bf16[10240, 2560]", arg18_1: "bf16[10240]", arg19_1: "bf16[2560, 10240]", arg20_1: "bf16[2560]", arg21_1: "i64[16, 128]", arg22_1: "bf16[2560]", arg23_1: "bf16[2560]", arg24_1: "bf16[8008, 2560]", arg25_1: "bf16[1, 8008]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32):
        # No stacktrace found for following nodes
        permute: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        view_1: "bf16[2048, 2560]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg2_1, view_1, permute_1);  arg2_1 = view_1 = permute_1 = None
        view_2: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        add: "bf16[16, 128, 2560]" = torch.ops.aten.add.Tensor(arg3_1, view_2);  arg3_1 = view_2 = None
        convert_element_type: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1]" = var_mean[0]
        getitem_1: "f32[16, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_3: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        permute_2: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm_1: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg7_1, view_3, permute_2);  arg7_1 = view_3 = permute_2 = None
        view_4: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        view_5: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_3: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        mul_2: "bf16[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_3, 0.334370152488211);  permute_3 = None
        expand: "bf16[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_2, _shape_param_6);  mul_2 = _shape_param_6 = None
        clone: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_6: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone, _shape_param_7);  clone = _shape_param_7 = None
        view_7: "bf16[2048, 2560]" = torch.ops.aten.view.default(arg8_1, _shape_param_8);  _shape_param_8 = None
        permute_4: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg10_1, view_7, permute_4);  arg10_1 = view_7 = permute_4 = None
        view_8: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_2, _shape_param_9);  addmm_2 = _shape_param_9 = None
        view_9: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_8, _shape_param_10);  view_8 = _shape_param_10 = None
        permute_5: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        permute_6: "bf16[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_5, [0, 1, 3, 2]);  permute_5 = None
        mul_3: "bf16[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_6, 0.334370152488211);  permute_6 = None
        expand_1: "bf16[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_3, _shape_param_11);  mul_3 = _shape_param_11 = None
        clone_1: "bf16[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_10: "bf16[512, 80, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_12);  clone_1 = _shape_param_12 = None
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_6, view_10);  view_6 = view_10 = None
        view_11: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_13);  bmm = _shape_param_13 = None
        eq: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(view_11, -inf)
        logical_not: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[16, 32, 128, 128]" = torch.ops.aten.full.default(_shape_param_14, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_14 = None
        convert_element_type_2: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        amax: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub_1: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "bf16[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where: "bf16[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_3);  logical_not_1 = full = convert_element_type_3 = None
        expand_2: "bf16[16, 32, 128, 128]" = torch.ops.aten.expand.default(where, _shape_param_15);  where = _shape_param_15 = None
        view_12: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_2, _shape_param_16);  expand_2 = _shape_param_16 = None
        view_13: "bf16[2048, 2560]" = torch.ops.aten.view.default(arg8_1, _shape_param_17);  arg8_1 = _shape_param_17 = None
        permute_7: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg12_1, view_13, permute_7);  arg12_1 = view_13 = permute_7 = None
        view_14: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_3, _shape_param_18);  addmm_3 = _shape_param_18 = None
        view_15: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_14, _shape_param_19);  view_14 = _shape_param_19 = None
        permute_8: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        expand_3: "bf16[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_8, _shape_param_20);  permute_8 = _shape_param_20 = None
        clone_2: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_16: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone_2, _shape_param_21);  clone_2 = _shape_param_21 = None
        bmm_1: "bf16[512, 128, 80]" = torch.ops.aten.bmm.default(view_12, view_16);  view_12 = view_16 = None
        view_17: "bf16[16, 32, 128, 80]" = torch.ops.aten.view.default(bmm_1, _shape_param_22);  bmm_1 = _shape_param_22 = None
        permute_9: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        clone_3: "bf16[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_18: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(clone_3, _shape_param_23);  clone_3 = _shape_param_23 = None
        view_19: "bf16[2048, 2560]" = torch.ops.aten.view.default(view_18, _shape_param_24);  view_18 = _shape_param_24 = None
        permute_10: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_4: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg14_1, view_19, permute_10);  arg14_1 = view_19 = permute_10 = None
        view_20: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_4, _shape_param_25);  addmm_4 = _shape_param_25 = None
        add_3: "bf16[16, 128, 2560]" = torch.ops.aten.add.Tensor(add, view_20);  add = view_20 = None
        convert_element_type_4: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[16, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[16, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_3);  convert_element_type_4 = getitem_3 = None
        add_4: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_4: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_4, arg15_1);  mul_4 = arg15_1 = None
        add_5: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_5, arg16_1);  mul_5 = arg16_1 = None
        convert_element_type_5: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        view_21: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_26);  convert_element_type_5 = _shape_param_26 = None
        permute_11: "bf16[2560, 10240]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5: "bf16[2048, 10240]" = torch.ops.aten.addmm.default(arg18_1, view_21, permute_11);  arg18_1 = view_21 = permute_11 = None
        view_22: "bf16[16, 128, 10240]" = torch.ops.aten.view.default(addmm_5, _shape_param_27);  addmm_5 = _shape_param_27 = None
        convert_element_type_6: "f32[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(view_22, torch.float32);  view_22 = None
        mul_6: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_7: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476);  convert_element_type_6 = None
        erf: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_6: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_6, add_6);  mul_6 = add_6 = None
        convert_element_type_7: "bf16[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        view_23: "bf16[2048, 10240]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_28);  convert_element_type_7 = _shape_param_28 = None
        permute_12: "bf16[10240, 2560]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_6: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg20_1, view_23, permute_12);  arg20_1 = view_23 = permute_12 = None
        view_24: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_6, _shape_param_29);  addmm_6 = _shape_param_29 = None
        add_7: "bf16[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_3, view_24);  add_3 = view_24 = None
        convert_element_type_8: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem_4: "f32[16, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[16, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        view_25: "i64[2048]" = torch.ops.aten.view.default(arg21_1, [-1]);  arg21_1 = None
        ne: "b8[2048]" = torch.ops.aten.ne.Scalar(view_25, -100)
        sub_3: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_5);  convert_element_type_8 = getitem_5 = None
        add_8: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_9: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_9, arg22_1);  mul_9 = arg22_1 = None
        add_9: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_10, arg23_1);  mul_10 = arg23_1 = None
        convert_element_type_9: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        view_26: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_30);  convert_element_type_9 = _shape_param_30 = None
        permute_13: "bf16[2560, 8008]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm: "bf16[2048, 8008]" = torch.ops.aten.mm.default(view_26, permute_13);  view_26 = permute_13 = None
        view_27: "bf16[16, 128, 8008]" = torch.ops.aten.view.default(mm, _shape_param_31);  mm = _shape_param_31 = None
        add_10: "bf16[16, 128, 8008]" = torch.ops.aten.add.Tensor(view_27, arg25_1);  view_27 = arg25_1 = None
        view_28: "bf16[2048, 8008]" = torch.ops.aten.view.default(add_10, _shape_param_32);  _shape_param_32 = None
        convert_element_type_10: "f32[2048, 8008]" = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        amax_1: "f32[2048, 1]" = torch.ops.aten.amax.default(convert_element_type_10, [1], True)
        sub_4: "f32[2048, 8008]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_1);  convert_element_type_10 = amax_1 = None
        exp_1: "f32[2048, 8008]" = torch.ops.aten.exp.default(sub_4)
        sum_2: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [1], True);  exp_1 = None
        log: "f32[2048, 1]" = torch.ops.aten.log.default(sum_2);  sum_2 = None
        sub_5: "f32[2048, 8008]" = torch.ops.aten.sub.Tensor(sub_4, log);  sub_4 = log = None
        convert_element_type_11: "bf16[2048, 8008]" = torch.ops.prims.convert_element_type.default(sub_5, torch.bfloat16);  sub_5 = None
        ne_1: "b8[2048]" = torch.ops.aten.ne.Scalar(view_25, -100)
        full_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[2048]" = torch.ops.aten.where.self(ne_1, view_25, full_1);  ne_1 = full_1 = None
        unsqueeze: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "bf16[2048, 1]" = torch.ops.aten.gather.default(convert_element_type_11, 1, unsqueeze);  convert_element_type_11 = unsqueeze = None
        squeeze: "bf16[2048]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[2048]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[2048]" = torch.ops.aten.where.self(ne, neg, full_2);  ne = neg = full_2 = None
        sum_3: "bf16[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        ne_2: "b8[2048]" = torch.ops.aten.ne.Scalar(view_25, -100);  view_25 = None
        sum_4: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_12: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_4, torch.bfloat16);  sum_4 = None
        div_1: "bf16[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_12);  sum_3 = convert_element_type_12 = None
        return (add_10, div_1)



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
